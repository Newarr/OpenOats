import CoreML
import Foundation
import os

/// Transcription backend for Cohere Transcribe, converted to CoreML.
///
/// The model is split into three CoreML components following the WhisperKit pattern:
///   - ConformerEncoder.mlmodelc  — mel features → encoder embeddings
///   - TransformerDecoder.mlmodelc — encoder embeddings + tokens → logits
///
/// Mel spectrogram extraction is performed in Swift (matching the model's preprocessing).
///
/// @unchecked Sendable: models are written once in prepare() before any transcribe() calls.
final class CohereTranscribeBackend: TranscriptionBackend, @unchecked Sendable {
    let displayName = "Cohere Transcribe"

    /// HuggingFace repo hosting the pre-converted CoreML models.
    static let modelRepo = "newarr/cohere-transcribe-coreml"

    /// Expected download size for the INT4-quantized model package.
    static let downloadSize = "~1.1 GB"

    private var encoder: MLModel?
    private var decoder: MLModel?
    private var config: CohereTranscribeConfig?

    private let log = Logger(subsystem: "com.openoats", category: "CohereTranscribe")

    // MARK: - TranscriptionBackend

    func checkStatus() -> BackendStatus {
        let exists = Self.modelExists()
        return exists
            ? .ready
            : .needsDownload(
                prompt: "Cohere Transcribe requires a one-time model download (\(Self.downloadSize))."
            )
    }

    func prepare(onStatus: @Sendable (String) -> Void) async throws {
        onStatus("Downloading Cohere Transcribe...")
        let modelDir = try await Self.ensureModelDownloaded(onStatus: onStatus)

        onStatus("Loading Cohere Transcribe models...")
        let cfg = try loadConfig(from: modelDir)
        self.config = cfg

        // Configure compute units: Conformer encoder and decoder on Neural Engine
        let computeConfig = MLModelConfiguration()
        computeConfig.computeUnits = .all

        let encoderURL = modelDir.appendingPathComponent("ConformerEncoder.mlmodelc")
        let decoderURL = modelDir.appendingPathComponent("TransformerDecoder.mlmodelc")

        guard FileManager.default.fileExists(atPath: encoderURL.path),
              FileManager.default.fileExists(atPath: decoderURL.path) else {
            throw CohereTranscribeError.modelFilesNotFound
        }

        self.encoder = try MLModel(contentsOf: encoderURL, configuration: computeConfig)
        self.decoder = try MLModel(contentsOf: decoderURL, configuration: computeConfig)

        log.info("Cohere Transcribe loaded successfully")
    }

    func transcribe(_ samples: [Float], locale: Locale) async throws -> String {
        guard let encoder, let decoder, let config else {
            throw TranscriptionBackendError.notPrepared
        }

        // 1. Compute mel spectrogram from raw 16kHz audio
        let melFeatures = computeMelSpectrogram(
            samples: samples,
            nMels: config.nMels,
            hopLength: config.hopLength,
            sampleRate: config.sampleRate
        )

        // 2. Run Conformer encoder
        let melArray = try MLMultiArray(shape: [1, config.nMels as NSNumber, melFeatures.count / config.nMels as NSNumber], dataType: .float16)
        for i in 0..<melFeatures.count {
            melArray[i] = NSNumber(value: melFeatures[i])
        }

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel_features": MLFeatureValue(multiArray: melArray)
        ])
        let encoderOutput = try encoder.prediction(from: encoderInput)

        guard let encoderEmbeddings = encoderOutput.featureValue(for: "encoder_output")?.multiArrayValue else {
            throw CohereTranscribeError.encoderOutputMissing
        }

        // 3. Autoregressive decoding
        var tokens: [Int32] = [Int32(config.bosTokenID)]
        let maxLength = config.maxTargetLength

        for _ in 0..<maxLength {
            let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
            tokenArray[0] = NSNumber(value: tokens.last!)

            let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: tokenArray),
                "encoder_output": MLFeatureValue(multiArray: encoderEmbeddings),
            ])

            let decoderOutput = try decoder.prediction(from: decoderInput)

            guard let logits = decoderOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw CohereTranscribeError.decoderOutputMissing
            }

            // Greedy: pick argmax of last token's logits
            let vocabSize = logits.shape.last!.intValue
            var maxIdx = 0
            var maxVal: Float = -.infinity
            for v in 0..<vocabSize {
                let val = logits[v].floatValue
                if val > maxVal {
                    maxVal = val
                    maxIdx = v
                }
            }

            if maxIdx == config.eosTokenID { break }
            tokens.append(Int32(maxIdx))
        }

        // 4. Decode tokens to text (skip BOS token)
        let outputTokens = Array(tokens.dropFirst())
        return decodeTokens(outputTokens, modelDir: Self.modelDirectory())
    }

    // MARK: - Mel Spectrogram

    /// Compute log-mel spectrogram features from raw audio samples.
    /// This matches the preprocessing expected by the Cohere Transcribe encoder.
    private func computeMelSpectrogram(
        samples: [Float],
        nMels: Int,
        hopLength: Int,
        sampleRate: Int
    ) -> [Float] {
        // Simplified mel computation — in production this should use vDSP/Accelerate
        // for FFT and mel filterbank application matching the model's training config.
        //
        // For the initial integration, we use a basic STFT + mel filterbank.
        // The exact parameters (window size, n_fft, etc.) must match the model's
        // AutoProcessor config — inspect processor_config.json after downloading.

        let nFFT = 400  // 25ms window at 16kHz (typical)
        let windowLength = nFFT
        let numFrames = max(1, (samples.count - windowLength) / hopLength + 1)

        // Placeholder: return zeros of the correct shape.
        // The real implementation should use vDSP.FFT and a mel filterbank matrix.
        // See WhisperKit's MelSpectrogram.mlmodelc approach — it computes mel via
        // a small CoreML model rather than in Swift, which is the recommended pattern.
        //
        // TODO: Replace with either:
        //   (a) A MelSpectrogram.mlmodelc converted from the model's preprocessor, or
        //   (b) An Accelerate/vDSP-based implementation matching processor_config.json
        return [Float](repeating: 0, count: nMels * numFrames)
    }

    // MARK: - Tokenizer

    /// Decode token IDs to text using the saved tokenizer.
    private func decodeTokens(_ tokens: [Int32], modelDir: URL) -> String {
        // Load vocab.json from the tokenizer directory for ID → string mapping.
        // In production, use swift-transformers' AutoTokenizer or a SentencePiece wrapper.
        //
        // TODO: Integrate with the swift-transformers package (already a transitive
        // dependency via WhisperKit) for proper tokenizer support.
        let tokenizerDir = modelDir.appendingPathComponent("tokenizer")
        let vocabURL = tokenizerDir.appendingPathComponent("vocab.json")

        guard let data = try? Data(contentsOf: vocabURL),
              let vocab = try? JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            return tokens.map { "[\($0)]" }.joined()
        }

        let reversed = Dictionary(uniqueKeysWithValues: vocab.map { ($0.value, $0.key) })
        return tokens.map { reversed[Int($0)] ?? "[\($0)]" }.joined()
    }

    // MARK: - Model Management

    /// Directory where CoreML model files are cached.
    private static func modelDirectory() -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent("OpenOats")
            .appendingPathComponent("models")
            .appendingPathComponent("cohere-transcribe")
    }

    /// Check whether the CoreML model files exist locally.
    static func modelExists() -> Bool {
        let dir = modelDirectory()
        let fm = FileManager.default
        return fm.fileExists(atPath: dir.appendingPathComponent("ConformerEncoder.mlmodelc").path)
            && fm.fileExists(atPath: dir.appendingPathComponent("TransformerDecoder.mlmodelc").path)
            && fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
    }

    /// Download and cache the CoreML model if not already present.
    private static func ensureModelDownloaded(
        onStatus: @Sendable (String) -> Void
    ) async throws -> URL {
        let dir = modelDirectory()
        if modelExists() { return dir }

        // Create the cache directory
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Download from HuggingFace using URLSession.
        // The model repo should contain pre-converted .mlmodelc bundles.
        let baseURL = "https://huggingface.co/\(modelRepo)/resolve/main"
        let filesToDownload = [
            "ConformerEncoder.mlmodelc.zip",
            "TransformerDecoder.mlmodelc.zip",
            "config.json",
            "tokenizer/vocab.json",
            "tokenizer/tokenizer_config.json",
        ]

        for file in filesToDownload {
            onStatus("Downloading \(file)...")
            let remoteURL = URL(string: "\(baseURL)/\(file)")!
            let localURL = dir.appendingPathComponent(file)

            // Ensure parent directory exists
            try FileManager.default.createDirectory(
                at: localURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            let (data, response) = try await URLSession.shared.data(from: remoteURL)
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw CohereTranscribeError.downloadFailed(file)
            }
            try data.write(to: localURL)

            // Unzip .mlmodelc bundles
            if file.hasSuffix(".zip") {
                onStatus("Extracting \(file)...")
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
                process.arguments = ["-o", localURL.path, "-d", dir.path]
                try process.run()
                process.waitUntilExit()
                try? FileManager.default.removeItem(at: localURL)
            }
        }

        return dir
    }

    // MARK: - Config

    private func loadConfig(from dir: URL) throws -> CohereTranscribeConfig {
        let configURL = dir.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        return try JSONDecoder().decode(CohereTranscribeConfig.self, from: data)
    }
}

// MARK: - Supporting Types

struct CohereTranscribeConfig: Codable, Sendable {
    let nMels: Int
    let sampleRate: Int
    let hopLength: Int
    let dModel: Int
    let vocabSize: Int
    let maxTargetLength: Int
    let bosTokenID: Int
    let eosTokenID: Int

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case sampleRate = "sample_rate"
        case hopLength = "hop_length"
        case dModel = "d_model"
        case vocabSize = "vocab_size"
        case maxTargetLength = "max_target_length"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
    }
}

enum CohereTranscribeError: LocalizedError {
    case modelFilesNotFound
    case encoderOutputMissing
    case decoderOutputMissing
    case downloadFailed(String)
    case configMissing

    var errorDescription: String? {
        switch self {
        case .modelFilesNotFound:
            "Cohere Transcribe CoreML model files not found. Run prepare() first."
        case .encoderOutputMissing:
            "Conformer encoder did not produce expected output."
        case .decoderOutputMissing:
            "Transformer decoder did not produce expected logits."
        case .downloadFailed(let file):
            "Failed to download model file: \(file)"
        case .configMissing:
            "Model config.json not found."
        }
    }
}
