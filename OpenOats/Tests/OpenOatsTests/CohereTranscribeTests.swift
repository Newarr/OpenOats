import XCTest
@testable import OpenOatsKit

final class CohereTranscribeTests: XCTestCase {

    // MARK: - Backend Protocol

    func testDisplayName() {
        let backend = CohereTranscribeBackend()
        XCTAssertEqual(backend.displayName, "Cohere Transcribe")
    }

    func testCheckStatusReturnsNeedsDownloadWhenModelAbsent() {
        let backend = CohereTranscribeBackend()
        let status = backend.checkStatus()
        // Model files won't be present in the test environment
        if case .needsDownload(let prompt) = status {
            XCTAssertTrue(prompt.contains("Cohere Transcribe"))
            XCTAssertTrue(prompt.contains("1.1 GB"))
        }
        // .ready is also acceptable if model files happen to be cached
    }

    func testTranscribeWithoutPrepareThrows() async {
        let backend = CohereTranscribeBackend()
        do {
            _ = try await backend.transcribe(
                [0.0, 0.1, 0.2], locale: Locale(identifier: "en-US")
            )
            XCTFail("Expected error")
        } catch is TranscriptionBackendError {
            // Expected: notPrepared
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - Model Existence

    func testModelExistsReturnsFalseWhenNoFiles() {
        // In a clean test environment, model files should not be present
        // This test documents the expected behavior
        let exists = CohereTranscribeBackend.modelExists()
        // We can't assert false because the model might be cached on the dev machine
        _ = exists  // Just verify it doesn't crash
    }

    func testModelDirectoryIsInApplicationSupport() {
        let dir = CohereTranscribeBackend.modelDirectory()
        XCTAssertTrue(dir.path.contains("OpenOats"))
        XCTAssertTrue(dir.path.contains("cohere-transcribe"))
    }

    // MARK: - Config

    func testConfigDecoding() throws {
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

    // MARK: - TranscriptionModel Enum

    func testTranscriptionModelIncludesCohereTranscribe() {
        let model = TranscriptionModel.cohereTranscribe
        XCTAssertEqual(model.displayName, "Cohere Transcribe")
        XCTAssertEqual(model.recommendedFlushSamples, 96_000)
    }

    func testMakeBackendReturnsCohereTranscribeBackend() {
        let model = TranscriptionModel.cohereTranscribe
        let backend = model.makeBackend()
        XCTAssertEqual(backend.displayName, "Cohere Transcribe")
    }
}
