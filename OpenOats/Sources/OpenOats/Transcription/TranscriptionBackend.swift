import Foundation

/// Status of a transcription backend's readiness.
enum BackendStatus: Equatable, Sendable {
    case ready
    case needsDownload(prompt: String)
    case preparing(status: String)
    case failed(message: String)
}

/// Unified interface for all transcription backends (local and cloud).
/// Each backend handles its own model lifecycle and transcription logic.
protocol TranscriptionBackend: Sendable {
    /// Human-readable name for UI display.
    var displayName: String { get }

    /// Check whether this backend is ready to transcribe.
    func checkStatus() -> BackendStatus

    /// Prepare the backend for use (download models, validate API keys, etc.).
    /// Called once before the first transcription. May be long-running.
    func prepare(onStatus: @Sendable (String) -> Void) async throws

    /// Transcribe a segment of Float32 audio samples at 16kHz mono.
    /// Returns the transcribed text, or empty string if no speech detected.
    func transcribe(_ samples: [Float], locale: Locale) async throws -> String
}

enum TranscriptionBackendError: Error {
    case notPrepared
}
