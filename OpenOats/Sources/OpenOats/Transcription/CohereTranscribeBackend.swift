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
/// preprocessing. The filterbank is extracted from the HuggingFace
/// AutoProcessor during conversion, guaranteeing numerical parity.
///
/// @unchecked Sendable: all mutable state is written once in prepare()
/// behind a lock, then only read during transcribe() calls.
final class CohereTranscribeBackend: TranscriptionBackend, @unchecked Sendable {
    let displayName = "Cohere Transcribe"

    /// HuggingFace repo hosting the pre-converted CoreML models.
    static let modelRepo = "newarr/cohere-transcribe-coreml"

    /// Expected download size for the INT4-quantized model package.
    static let downloadSize = "~1.1 GB"

    // All mutable state is guarded by prepareLock.
    // Written once in prepare(), then read-only during transcribe().
    private let prepareLock = NSLock()
    private var encoder: MLModel?
    private var decoder: MLModel?
    private var config: CohereTranscribeConfig?
    private var tokenizer: (any Tokenizer)?
    private var melFilterbank: [Float]?  // (nMels, freqBins) row-major
    private var hannWindow: [Float]?
    private var isPrepared = false

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
        // Guard against concurrent prepare() calls.
        let alreadyPrepared = prepareLock.withLock { isPrepared }
        if alreadyPrepared { return }

        onStatus("Downloading Cohere Transcribe...")
        let modelDir = try await Self.ensureModelDownloaded(onStatus: onStatus)

        onStatus("Loading Cohere Transcribe models...")
        let cfg = try loadConfig(from: modelDir)

        // Load pre-computed mel filterbank and window
        let filterbank = try loadFloatBinary(
            from: modelDir.appendingPathComponent("mel_filterbank.bin")
        )
        let expectedFilterbankSize = cfg.nMels * (cfg.nFFT / 2 + 1)
        guard filterbank.count == expectedFilterbankSize else {
            throw CohereTranscribeError.filterBankShapeMismatch(
                expected: expectedFilterbankSize, got: filterbank.count
            )
        }

        let window = try loadFloatBinary(
            from: modelDir.appendingPathComponent("hann_window.bin")
        )
        guard window.count == cfg.nFFT else {
            throw CohereTranscribeError.windowSizeMismatch(
                expected: cfg.nFFT, got: window.count
            )
        }

        // Load tokenizer via swift-transformers
        let tokenizerDir = modelDir.appendingPathComponent("tokenizer")
        let tok = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        // Configure compute units: Neural Engine + GPU + CPU
        let computeConfig = MLModelConfiguration()
        computeConfig.computeUnits = .all

        let encoderURL = modelDir.appendingPathComponent("ConformerEncoder.mlmodelc")
        let decoderURL = modelDir.appendingPathComponent("TransformerDecoder.mlmodelc")

        guard FileManager.default.fileExists(atPath: encoderURL.path),
              FileManager.default.fileExists(atPath: decoderURL.path) else {
            throw CohereTranscribeError.modelFilesNotFound
        }

        let enc = try MLModel(contentsOf: encoderURL, configuration: computeConfig)
        let dec = try MLModel(contentsOf: decoderURL, configuration: computeConfig)

        // Atomically publish all state
        prepareLock.withLock {
            self.config = cfg
            self.melFilterbank = filterbank
            self.hannWindow = window
            self.tokenizer = tok
            self.encoder = enc
            self.decoder = dec
            self.isPrepared = true
        }

        log.info("Cohere Transcribe loaded (filterbank: \(filterbank.count) floats, window: \(window.count))")
    }

    func transcribe(_ samples: [Float], locale: Locale) async throws -> String {
        // Snapshot state under lock
        let (encoder, decoder, config, tokenizer) = prepareLock.withLock {
            (self.encoder, self.decoder, self.config, self.tokenizer)
        }
        guard let encoder, let decoder, let config, let tokenizer else {
            throw TranscriptionBackendError.notPrepared
        }

        guard !samples.isEmpty else { return "" }

        // 1. Compute mel spectrogram
        let melResult = try computeMelSpectrogram(samples: samples, config: config)
        guard melResult.numFrames > 0 else { return "" }

        // 2. Run Conformer encoder
        let melArray = try MLMultiArray(
            shape: [1, NSNumber(value: config.nMels), NSNumber(value: melResult.numFrames)],
            dataType: .float16
        )
        // Copy mel features into MLMultiArray with correct 3D strides.
        // melResult.data is row-major (nMels, numFrames): mel index varies slowest.
        let melStrides = melArray.strides.map { $0.intValue }
        for m in 0..<config.nMels {
            for f in 0..<melResult.numFrames {
                let linearIdx = melStrides[0] * 0 + melStrides[1] * m + melStrides[2] * f
                melArray[linearIdx] = NSNumber(value: melResult.data[m * melResult.numFrames + f])
            }
        }

        let encoderOutput = try encoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "mel_features": MLFeatureValue(multiArray: melArray)
            ])
        )

        guard let encoderEmbeddings = encoderOutput.featureValue(
            for: "encoder_output"
        )?.multiArrayValue else {
            throw CohereTranscribeError.encoderOutputMissing
        }

        // 3. Autoregressive greedy decoding
        var tokens: [Int32] = [Int32(config.bosTokenID)]

        for _ in 0..<config.maxTargetLength {
            let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
            tokenArray[0] = NSNumber(value: tokens[tokens.count - 1])

            let decoderOutput = try decoder.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "input_ids": MLFeatureValue(multiArray: tokenArray),
                    "encoder_output": MLFeatureValue(multiArray: encoderEmbeddings),
                ])
            )

            guard let logits = decoderOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw CohereTranscribeError.decoderOutputMissing
            }

            // Greedy argmax over vocabulary dimension
            let vocabSize = logits.shape.last!.intValue
            guard vocabSize > 0 else { break }

            let logitsPtr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            var maxIdx: vDSP_Length = 0
            var maxVal: Float = -.infinity
            // Use vDSP to find argmax efficiently
            for v in 0..<vocabSize {
                let val = Float(logitsPtr[v])
                if val > maxVal {
                    maxVal = val
                    maxIdx = vDSP_Length(v)
                }
            }

            let nextToken = Int(maxIdx)
            if nextToken == config.eosTokenID { break }
            tokens.append(Int32(nextToken))
        }

        // 4. Decode tokens to text (skip BOS)
        let outputTokens = Array(tokens.dropFirst())
        guard !outputTokens.isEmpty else { return "" }
        return tokenizer.decode(tokens: outputTokens.map { Int($0) })
    }

    // MARK: - Mel Spectrogram (Accelerate/vDSP)

    /// Result of mel spectrogram computation.
    private struct MelResult {
        let data: [Float]   // (nMels, numFrames) row-major
        let numFrames: Int
    }

    /// Compute log-mel spectrogram features from raw 16kHz mono audio using vDSP.
    ///
    /// Uses a pre-exported mel filterbank matrix for exact match with training.
    /// The FFT is computed via Accelerate's vDSP_fft_zrip (real-to-complex).
    private func computeMelSpectrogram(
        samples: [Float],
        config: CohereTranscribeConfig
    ) throws -> MelResult {
        guard let filterbank = prepareLock.withLock({ self.melFilterbank }),
              let window = prepareLock.withLock({ self.hannWindow }) else {
            throw CohereTranscribeError.modelFilesNotFound
        }

        let nFFT = config.nFFT
        let hopLength = config.hopLength
        let nMels = config.nMels
        let halfN = nFFT / 2
        let freqBins = halfN + 1

        // 1. Reflect-pad samples (matches numpy reflect mode)
        let padLength = halfN
        let paddedCount = padLength + samples.count + padLength
        var padded = [Float](repeating: 0, count: paddedCount)

        // Left reflect: padded[padLength-1-i] = samples[i+1] (clamped)
        for i in 0..<padLength {
            let srcIdx = min(i + 1, samples.count - 1)
            padded[padLength - 1 - i] = samples[max(srcIdx, 0)]
        }
        // Center: copy original samples
        padded.withUnsafeMutableBufferPointer { dst in
            samples.withUnsafeBufferPointer { src in
                dst.baseAddress!.advanced(by: padLength)
                    .update(from: src.baseAddress!, count: samples.count)
            }
        }
        // Right reflect: padded[padLength+count+i] = samples[count-2-i] (clamped)
        for i in 0..<padLength {
            let srcIdx = max(samples.count - 2 - i, 0)
            padded[padLength + samples.count + i] = samples[srcIdx]
        }

        // 2. Number of STFT frames
        let numFrames = max(1, (paddedCount - nFFT) / hopLength + 1)

        // 3. FFT setup
        let log2n = vDSP_Length(log2(Float(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            throw CohereTranscribeError.fftSetupFailed
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // 4. Pre-allocate reusable buffers (outside the loop)
        var windowedFrame = [Float](repeating: 0, count: nFFT)
        var realPart = [Float](repeating: 0, count: halfN)
        var imagPart = [Float](repeating: 0, count: halfN)
        // Power spectrum per frame, stored as (freqBins, numFrames) for matrix multiply
        var powerSpec = [Float](repeating: 0, count: freqBins * numFrames)

        // 5. For each frame: window → FFT → power spectrum
        padded.withUnsafeBufferPointer { paddedBuf in
            for frame in 0..<numFrames {
                let start = frame * hopLength

                // Apply Hann window: windowedFrame = padded[start..] * window
                vDSP_vmul(
                    paddedBuf.baseAddress! + start, 1,
                    window, 1,
                    &windowedFrame, 1,
                    vDSP_Length(nFFT)
                )

                // Convert interleaved real data to split complex for in-place FFT
                windowedFrame.withUnsafeBufferPointer { frameBuf in
                    frameBuf.baseAddress!.withMemoryRebound(
                        to: DSPComplex.self, capacity: halfN
                    ) { complexPtr in
                        var split = DSPSplitComplex(realp: &realPart, imagp: &imagPart)
                        vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(halfN))
                    }
                }

                // Forward real-to-complex FFT
                var split = DSPSplitComplex(realp: &realPart, imagp: &imagPart)
                vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))

                // Extract power spectrum from packed output.
                // vDSP_fft_zrip packs DC component in realp[0], Nyquist in imagp[0].
                // Output is scaled by 2 relative to DFT definition.
                let fftScale: Float = 1.0 / Float(2 * nFFT)

                // DC bin (k=0): only real component
                let dc = realPart[0] * fftScale
                powerSpec[0 * numFrames + frame] = dc * dc

                // Bins k=1..halfN-1: complex magnitude squared
                for k in 1..<halfN {
                    let re = realPart[k] * fftScale
                    let im = imagPart[k] * fftScale
                    powerSpec[k * numFrames + frame] = re * re + im * im
                }

                // Nyquist bin (k=halfN): only real component (packed in imagp[0])
                let nyq = imagPart[0] * fftScale
                powerSpec[halfN * numFrames + frame] = nyq * nyq
            }
        }

        // 6. Apply mel filterbank via matrix multiply
        // filterbank: (nMels, freqBins)  *  powerSpec: (freqBins, numFrames)
        //           → melSpec: (nMels, numFrames)
        var melSpec = [Float](repeating: 0, count: nMels * numFrames)
        vDSP_mmul(
            filterbank, 1,
            powerSpec, 1,
            &melSpec, 1,
            vDSP_Length(nMels),
            vDSP_Length(numFrames),
            vDSP_Length(freqBins)
        )

        // 7. Log scale: log10(max(mel, 1e-10))
        var floor: Float = 1e-10
        vDSP_vthr(melSpec, 1, &floor, &melSpec, 1, vDSP_Length(melSpec.count))
        var count = Int32(melSpec.count)
        vvlog10f(&melSpec, melSpec, &count)

        return MelResult(data: melSpec, numFrames: numFrames)
    }

    // MARK: - Model Management

    /// Directory where CoreML model files are cached.
    static func modelDirectory() -> URL {
        guard let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first else {
            // Fallback to temp directory (should never happen on macOS)
            return URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("OpenOats/models/cohere-transcribe")
        }
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

        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

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

            try FileManager.default.createDirectory(
                at: localURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            let (data, response) = try await URLSession.shared.data(from: remoteURL)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw CohereTranscribeError.downloadFailed(file)
            }
            guard httpResponse.statusCode == 200 else {
                throw CohereTranscribeError.downloadHTTPError(
                    file: file, status: httpResponse.statusCode
                )
            }
            guard !data.isEmpty else {
                throw CohereTranscribeError.downloadEmpty(file)
            }
            try data.write(to: localURL)

            if file.hasSuffix(".zip") {
                onStatus("Extracting \(file)...")
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
                process.arguments = ["-o", localURL.path, "-d", dir.path]
                try process.run()
                process.waitUntilExit()
                guard process.terminationStatus == 0 else {
                    throw CohereTranscribeError.extractionFailed(file)
                }
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
    func loadFloatBinary(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count % MemoryLayout<Float>.size == 0 else {
            throw CohereTranscribeError.invalidBinaryFile(url.lastPathComponent)
        }
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
    case downloadHTTPError(file: String, status: Int)
    case downloadEmpty(String)
    case extractionFailed(String)
    case configMissing
    case filterBankShapeMismatch(expected: Int, got: Int)
    case windowSizeMismatch(expected: Int, got: Int)
    case fftSetupFailed
    case invalidBinaryFile(String)

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
        case .downloadHTTPError(let file, let status):
            "HTTP \(status) downloading \(file)"
        case .downloadEmpty(let file):
            "Downloaded empty file: \(file)"
        case .extractionFailed(let file):
            "Failed to extract: \(file)"
        case .configMissing:
            "Model config.json not found."
        case .filterBankShapeMismatch(let expected, let got):
            "Mel filterbank size mismatch: expected \(expected) floats, got \(got)"
        case .windowSizeMismatch(let expected, let got):
            "Hann window size mismatch: expected \(expected), got \(got)"
        case .fftSetupFailed:
            "Failed to create vDSP FFT setup."
        case .invalidBinaryFile(let name):
            "Invalid binary file: \(name) (size not aligned to Float32)"
        }
    }
}
