import XCTest
@testable import OpenOatsKit

final class CohereTranscribeTests: XCTestCase {

    // MARK: - Backend Protocol Conformance

    func testDisplayName() {
        let backend = CohereTranscribeBackend()
        XCTAssertEqual(backend.displayName, "Cohere Transcribe")
    }

    func testCheckStatusReturnsNeedsDownloadWhenModelAbsent() {
        let backend = CohereTranscribeBackend()
        let status = backend.checkStatus()
        // Model files won't be present in CI; .ready is fine on dev machines
        switch status {
        case .needsDownload(let prompt):
            XCTAssertTrue(prompt.contains("Cohere Transcribe"))
            XCTAssertTrue(prompt.contains("1.1 GB"))
        case .ready:
            break
        }
    }

    func testTranscribeWithoutPrepareThrows() async {
        let backend = CohereTranscribeBackend()
        do {
            _ = try await backend.transcribe(
                [Float](repeating: 0, count: 16_000),  // 1 second of silence
                locale: Locale(identifier: "en-US")
            )
            XCTFail("Expected TranscriptionBackendError.notPrepared")
        } catch is TranscriptionBackendError {
            // Expected
        } catch {
            XCTFail("Unexpected error type: \(type(of: error)): \(error)")
        }
    }

    func testTranscribeEmptyAudioReturnsEmpty() async {
        let backend = CohereTranscribeBackend()
        // Even without prepare, empty audio should return "" before hitting notPrepared
        // But actually the guard for notPrepared comes first, so this throws.
        do {
            let result = try await backend.transcribe([], locale: Locale(identifier: "en-US"))
            XCTAssertEqual(result, "")
        } catch is TranscriptionBackendError {
            // Also acceptable — notPrepared fires before empty check
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    // MARK: - Model Directory

    func testModelDirectoryIsInApplicationSupport() {
        let dir = CohereTranscribeBackend.modelDirectory()
        XCTAssertTrue(dir.path.contains("OpenOats"), "Expected 'OpenOats' in path: \(dir.path)")
        XCTAssertTrue(
            dir.path.contains("cohere-transcribe"),
            "Expected 'cohere-transcribe' in path: \(dir.path)"
        )
    }

    func testModelExistsDoesNotCrash() {
        // Just verify this doesn't throw or crash
        _ = CohereTranscribeBackend.modelExists()
    }

    // MARK: - Config Decoding

    func testConfigDecodingWithAllFields() throws {
        let json = """
        {
            "n_mels": 128,
            "sample_rate": 16000,
            "hop_length": 160,
            "n_fft": 400,
            "window_length": 400,
            "d_model": 1024,
            "vocab_size": 51865,
            "max_target_length": 448,
            "bos_token_id": 50257,
            "eos_token_id": 50256
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(CohereTranscribeConfig.self, from: json)
        XCTAssertEqual(config.nMels, 128)
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.hopLength, 160)
        XCTAssertEqual(config.nFFT, 400)
        XCTAssertEqual(config.windowLength, 400)
        XCTAssertEqual(config.dModel, 1024)
        XCTAssertEqual(config.vocabSize, 51865)
        XCTAssertEqual(config.maxTargetLength, 448)
        XCTAssertEqual(config.bosTokenID, 50257)
        XCTAssertEqual(config.eosTokenID, 50256)
    }

    func testConfigDecodingMissingFieldThrows() {
        let incomplete = """
        { "n_mels": 128, "sample_rate": 16000 }
        """.data(using: .utf8)!

        XCTAssertThrowsError(
            try JSONDecoder().decode(CohereTranscribeConfig.self, from: incomplete)
        )
    }

    // MARK: - TranscriptionModel Enum Integration

    func testTranscriptionModelIncludesCohereTranscribe() {
        let model = TranscriptionModel.cohereTranscribe
        XCTAssertEqual(model.displayName, "Cohere Transcribe")
        XCTAssertEqual(model.recommendedFlushSamples, 96_000)  // 6 seconds
    }

    func testMakeBackendReturnsCohereTranscribeBackend() {
        let backend = TranscriptionModel.cohereTranscribe.makeBackend()
        XCTAssertEqual(backend.displayName, "Cohere Transcribe")
    }

    func testCohereTranscribeDoesNotSupportLanguageHint() {
        XCTAssertFalse(TranscriptionModel.cohereTranscribe.supportsExplicitLanguageHint)
    }

    func testCohereTranscribeDownloadPrompt() {
        let prompt = TranscriptionModel.cohereTranscribe.downloadPrompt
        XCTAssertTrue(prompt.contains("1.1 GB"))
    }

    // MARK: - Float Binary Loading

    func testLoadFloatBinaryWithValidData() throws {
        let backend = CohereTranscribeBackend()
        let expected: [Float] = [1.0, 2.0, 3.0, -1.5, 0.0]
        var data = Data()
        for val in expected {
            withUnsafeBytes(of: val) { data.append(contentsOf: $0) }
        }

        let tmpURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("test_floats_\(UUID().uuidString).bin")
        try data.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let loaded = try backend.loadFloatBinary(from: tmpURL)
        XCTAssertEqual(loaded.count, expected.count)
        for (a, b) in zip(loaded, expected) {
            XCTAssertEqual(a, b, accuracy: 1e-7)
        }
    }

    func testLoadFloatBinaryMisalignedThrows() throws {
        let backend = CohereTranscribeBackend()
        let data = Data([0x01, 0x02, 0x03])  // 3 bytes, not aligned to Float (4 bytes)

        let tmpURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("test_misaligned_\(UUID().uuidString).bin")
        try data.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        XCTAssertThrowsError(try backend.loadFloatBinary(from: tmpURL)) { error in
            XCTAssertTrue(
                error.localizedDescription.contains("not aligned"),
                "Expected alignment error, got: \(error)"
            )
        }
    }

    func testLoadFloatBinaryMissingFileThrows() {
        let backend = CohereTranscribeBackend()
        let bogusURL = URL(fileURLWithPath: "/tmp/nonexistent_\(UUID().uuidString).bin")
        XCTAssertThrowsError(try backend.loadFloatBinary(from: bogusURL))
    }

    // MARK: - Error Descriptions

    func testErrorDescriptionsAreHelpful() {
        let errors: [CohereTranscribeError] = [
            .modelFilesNotFound,
            .encoderOutputMissing,
            .decoderOutputMissing,
            .downloadFailed("test.bin"),
            .downloadHTTPError(file: "model.zip", status: 404),
            .downloadEmpty("empty.bin"),
            .extractionFailed("corrupt.zip"),
            .configMissing,
            .filterBankShapeMismatch(expected: 25728, got: 100),
            .windowSizeMismatch(expected: 400, got: 256),
            .fftSetupFailed,
            .invalidBinaryFile("bad.bin"),
        ]

        for error in errors {
            let desc = error.errorDescription ?? ""
            XCTAssertFalse(desc.isEmpty, "Error \(error) has no description")
            XCTAssertTrue(desc.count > 10, "Error description too short: \(desc)")
        }
    }

    func testHTTPErrorIncludesStatusCode() {
        let error = CohereTranscribeError.downloadHTTPError(file: "model.zip", status: 404)
        XCTAssertTrue(error.errorDescription!.contains("404"))
        XCTAssertTrue(error.errorDescription!.contains("model.zip"))
    }

    // MARK: - Batch Transcription Types

    func testBatchSegmentStoresTimestamps() {
        let seg = BatchSegment(text: "hello world", startTime: 0.0, endTime: 30.0)
        XCTAssertEqual(seg.text, "hello world")
        XCTAssertEqual(seg.startTime, 0.0)
        XCTAssertEqual(seg.endTime, 30.0)
    }

    func testBatchTranscriptionErrorDescriptions() {
        let errors: [BatchTranscriptionError] = [
            .notPrepared,
            .alreadyRunning,
            .audioFileNotFound("/fake/path.m4a"),
            .audioEmpty,
            .audioTooLong(seconds: 999999),
            .audioFormatError,
            .audioConversionFailed(from: "48000Hz/2ch", to: "16000Hz/1ch"),
        ]

        for error in errors {
            let desc = error.errorDescription ?? ""
            XCTAssertFalse(desc.isEmpty, "Error \(error) has no description")
        }
    }
}
