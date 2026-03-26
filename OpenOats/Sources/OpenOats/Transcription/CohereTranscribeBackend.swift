import Accelerate
import CoreML
import Foundation
import Tokenizers
import os

/// Transcription backend for Cohere Transcribe, converted to CoreML.
///
/// The model is split into CoreML components following the WhisperKit pattern:
///   - ConformerEncoder.mlmodelc  — mel features → encoder embeddings
///   - TransformerDecoder.mlmodelc — encoder embeddings + tokens → logits
///
/// Mel spectrogram is computed natively using Accelerate/vDSP with a
/// pre-exported filterbank matrix that exactly matches the model's training
/// preprocessing.
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
    private var tokenizer: (any Tokenizer)?

    /// Pre-computed mel filterbank matrix, shape: (nMels, nFFT/2+1), row-major.
    /// Exported from the model's AutoProcessor to guarantee exact match.
    private var melFilterbank: [Float]?

    /// Hann window of length nFFT, exported alongside the filterbank.
    private var hannWindow: [Float]?

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

        // Load pre-computed mel filterbank and window
        self.melFilterbank = try loadFloatBinary(
            from: modelDir.appendingPathComponent("mel_filterbank.bin")
        )
        self.hannWindow = try loadFloatBinary(
            from: modelDir.appendingPathComponent("hann_window.bin")
        )

        // Load tokenizer via swift-transformers
        let tokenizerDir = modelDir.appendingPathComponent("tokenizer")
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

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
        let melFeatures = computeMelSpectrogram(samples: samples)
        let numFrames = melFeatures.count / config.nMels
        guard numFrames > 0 else { return "" }

        // 2. Run Conformer encoder
        let melArray = try MLMultiArray(
            shape: [1, config.nMels as NSNumber, numFrames as NSNumber],
            dataType: .float16
        )
        for i in 0..<melFeatures.count {
            melArray[i] = NSNumber(value: melFeatures[i])
        }

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel_features": MLFeatureValue(multiArray: melArray)
        ])
        let encoderOutput = try encoder.prediction(from: encoderInput)

        guard let encoderEmbeddings = encoderOutput.featureValue(
            for: "encoder_output"
        )?.multiArrayValue else {
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
        return decodeTokens(outputTokens)
    }

    // MARK: - Mel Spectrogram (Accelerate/vDSP)

    /// Compute log-mel spectrogram features from raw audio samples using vDSP.
    ///
    /// Uses a pre-exported mel filterbank matrix from the model's AutoProcessor
    /// to guarantee exact match with the training preprocessing. The FFT is
    /// computed natively using Accelerate for maximum performance on Apple Silicon.
    private func computeMelSpectrogram(samples: [Float]) -> [Float] {
        guard let filterbank = melFilterbank,
              let window = hannWindow,
              let config else {
            return []
        }

        let nFFT = config.nFFT
        let hopLength = config.hopLength
        let nMels = config.nMels
        let freqBins = nFFT / 2 + 1

        // 1. Reflect-pad samples to center frames (matches librosa/torch default)
        let padLength = nFFT / 2
        var padded = [Float](repeating: 0, count: padLength + samples.count + padLength)
        // Reflect padding: mirror the edges
        for i in 0..<padLength {
            padded[padLength - 1 - i] = samples[min(i + 1, samples.count - 1)]
        }
        for i in 0..<samples.count {
            padded[padLength + i] = samples[i]
        }
        for i in 0..<padLength {
            padded[padLength + samples.count + i] = samples[max(samples.count - 2 - i, 0)]
        }

        // 2. Compute number of frames
        let numFrames = max(1, (padded.count - nFFT) / hopLength + 1)

        // 3. Setup vDSP FFT
        let log2n = vDSP_Length(log2(Float(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return []
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // 4. Compute power spectrum for each frame
        // Store as (freqBins, numFrames) so filterbank @ magnitudes = (nMels, numFrames)
        var magnitudes = [Float](repeating: 0, count: freqBins * numFrames)

        let halfN = nFFT / 2
        var windowedFrame = [Float](repeating: 0, count: nFFT)
        var realPart = [Float](repeating: 0, count: halfN)
        var imagPart = [Float](repeating: 0, count: halfN)

        for frame in 0..<numFrames {
            let start = frame * hopLength

            // Apply Hann window
            vDSP_vmul(
                Array(padded[start..<start + nFFT]), 1,
                window, 1,
                &windowedFrame, 1,
                vDSP_Length(nFFT)
            )

            // Pack into split complex for in-place FFT
            windowedFrame.withUnsafeBufferPointer { buf in
                buf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                    var split = DSPSplitComplex(realp: &realPart, imagp: &imagPart)
                    vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(halfN))
                }
            }

            // Forward FFT
            var split = DSPSplitComplex(realp: &realPart, imagp: &imagPart)
            vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))

            // Unpack power spectrum.
            // vDSP_fft_zrip packs: DC in realp[0], Nyquist in imagp[0].
            // Scale factor: vDSP FFT output is 2x, so divide by 2 (or adjust power by /4).
            let scale: Float = 1.0 / Float(nFFT)

            // DC bin (index 0): power = (realp[0] * scale)^2
            let dc = realPart[0] * scale
            magnitudes[0 * numFrames + frame] = dc * dc

            // Bins 1..halfN-1
            for k in 1..<halfN {
                let re = realPart[k] * scale
                let im = imagPart[k] * scale
                magnitudes[k * numFrames + frame] = re * re + im * im
            }

            // Nyquist bin (index halfN): power = (imagp[0] * scale)^2
            let nyq = imagPart[0] * scale
            magnitudes[halfN * numFrames + frame] = nyq * nyq
        }

        // 5. Apply mel filterbank: result = filterbank @ magnitudes
        // filterbank: (nMels, freqBins) row-major
        // magnitudes: (freqBins, numFrames) column-major
        // result:     (nMels, numFrames)
        var melSpec = [Float](repeating: 0, count: nMels * numFrames)
        vDSP_mmul(
            filterbank, 1,
            magnitudes, 1,
            &melSpec, 1,
            vDSP_Length(nMels),
            vDSP_Length(numFrames),
            vDSP_Length(freqBins)
        )

        // 6. Log scale: log10(max(mel, 1e-10))
        // Default to log10 (most common for speech models). The preprocessor_config.json
        // can specify "ln" if the model uses natural log instead.
        var floorVal: Float = 1e-10
        vDSP_vthr(melSpec, 1, &floorVal, &melSpec, 1, vDSP_Length(melSpec.count))
        var count = Int32(melSpec.count)
        vvlog10f(&melSpec, melSpec, &count)

        return melSpec
    }

    // MARK: - Tokenizer

    /// Decode token IDs to text using swift-transformers AutoTokenizer.
    private func decodeTokens(_ tokens: [Int32]) -> String {
        guard let tokenizer else {
            return tokens.map { "[\($0)]" }.joined()
        }
        return tokenizer.decode(tokens: tokens.map { Int($0) })
    }

    // MARK: - Model Management

    /// Directory where CoreML model files are cached.
    static func modelDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
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
            && fm.fileExists(atPath: dir.appendingPathComponent("mel_filterbank.bin").path)
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
        let baseURL = "https://huggingface.co/\(modelRepo)/resolve/main"
        let filesToDownload = [
            "ConformerEncoder.mlmodelc.zip",
            "TransformerDecoder.mlmodelc.zip",
            "config.json",
            "mel_filterbank.bin",
            "hann_window.bin",
            "tokenizer/vocab.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
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

    // MARK: - Helpers

    private func loadConfig(from dir: URL) throws -> CohereTranscribeConfig {
        let configURL = dir.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        return try JSONDecoder().decode(CohereTranscribeConfig.self, from: data)
    }

    /// Load a raw Float32 binary file into a Swift array.
    private func loadFloatBinary(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
    }
}

// MARK: - Supporting Types

struct CohereTranscribeConfig: Codable, Sendable {
    let nMels: Int
    let sampleRate: Int
    let hopLength: Int
    let nFFT: Int
    let windowLength: Int
    let dModel: Int
    let vocabSize: Int
    let maxTargetLength: Int
    let bosTokenID: Int
    let eosTokenID: Int

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case sampleRate = "sample_rate"
        case hopLength = "hop_length"
        case nFFT = "n_fft"
        case windowLength = "window_length"
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
