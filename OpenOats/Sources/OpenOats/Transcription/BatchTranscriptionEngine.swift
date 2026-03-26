@preconcurrency import AVFoundation
import Foundation
import os

/// Runs a high-accuracy batch re-transcription of a recorded session using
/// Cohere Transcribe (CoreML). Designed to run asynchronously after a meeting
/// ends, replacing the live transcript with a more accurate version.
///
/// The engine processes saved audio files (one per channel) through the
/// Cohere Transcribe CoreML model and produces refined text that is
/// backfilled into the session's JSONL and Markdown files.
actor BatchTranscriptionEngine {
    private let log = Logger(subsystem: "com.openoats", category: "BatchTranscription")
    private var backend: CohereTranscribeBackend?
    private var isRunning = false

    /// Prepare the Cohere Transcribe backend (download + load CoreML models).
    /// Call once before any transcription. Safe to call multiple times.
    func prepare(onStatus: @Sendable (String) -> Void) async throws {
        if backend != nil { return }
        let b = CohereTranscribeBackend()
        try await b.prepare(onStatus: onStatus)
        self.backend = b
    }

    /// Re-transcribe a session's audio file and return timestamped segments.
    ///
    /// - Parameters:
    ///   - audioURL: Path to the session's M4A audio recording.
    ///   - locale: The session's language locale.
    /// - Returns: Array of transcribed text segments with approximate timestamps.
    func transcribe(
        audioURL: URL,
        locale: Locale
    ) async throws -> [BatchSegment] {
        guard let backend else {
            throw BatchTranscriptionError.notPrepared
        }
        guard !isRunning else {
            throw BatchTranscriptionError.alreadyRunning
        }

        isRunning = true
        defer { isRunning = false }

        log.info("Starting batch transcription of \(audioURL.lastPathComponent)")

        // Load audio file and convert to 16kHz mono Float32 samples
        let samples = try loadAudio(from: audioURL)
        log.info("Loaded \(samples.count) samples (\(String(format: "%.1f", Double(samples.count) / 16000))s)")

        // Process in chunks matching the model's audio bucket sizes.
        // For batch mode we use larger chunks (30s) since latency doesn't matter.
        let chunkDuration = 30  // seconds
        let chunkSamples = chunkDuration * 16_000
        var segments: [BatchSegment] = []

        var offset = 0
        var chunkIndex = 0
        while offset < samples.count {
            let end = min(offset + chunkSamples, samples.count)
            let chunk = Array(samples[offset..<end])

            let startTime = Double(offset) / 16_000
            let endTime = Double(end) / 16_000

            let text = try await backend.transcribe(chunk, locale: locale)
            if !text.isEmpty {
                segments.append(BatchSegment(
                    text: text,
                    startTime: startTime,
                    endTime: endTime
                ))
            }

            chunkIndex += 1
            log.info("Chunk \(chunkIndex): \(String(format: "%.0f", startTime))s-\(String(format: "%.0f", endTime))s → \(text.prefix(80))...")
            offset = end
        }

        log.info("Batch transcription complete: \(segments.count) segments")
        return segments
    }

    /// Load audio from an M4A file and convert to 16kHz mono Float32 samples.
    private func loadAudio(from url: URL) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw BatchTranscriptionError.audioFileNotFound(url.path)
        }

        let file = try AVAudioFile(forReading: url)
        let srcFormat = file.processingFormat

        // Target: 16kHz mono
        let targetRate: Double = 16_000
        guard let targetFormat = AVAudioFormat(
            standardFormatWithSampleRate: targetRate, channels: 1
        ) else {
            throw BatchTranscriptionError.audioFormatError
        }

        let frameCount = AVAudioFrameCount(file.length)
        guard let readBuf = AVAudioPCMBuffer(
            pcmFormat: srcFormat, frameCapacity: frameCount
        ) else {
            throw BatchTranscriptionError.audioFormatError
        }
        try file.read(into: readBuf)

        // If already at target format, extract directly
        if srcFormat.sampleRate == targetRate && srcFormat.channelCount == 1 {
            return extractSamples(from: readBuf)
        }

        // Resample via AVAudioConverter
        guard let converter = AVAudioConverter(from: srcFormat, to: targetFormat) else {
            throw BatchTranscriptionError.audioFormatError
        }

        let ratio = targetRate / srcFormat.sampleRate
        let outFrames = AVAudioFrameCount(Double(frameCount) * ratio) + 1
        guard let outBuf = AVAudioPCMBuffer(
            pcmFormat: targetFormat, frameCapacity: outFrames
        ) else {
            throw BatchTranscriptionError.audioFormatError
        }

        var consumed = false
        var convError: NSError?
        converter.convert(to: outBuf, error: &convError) { _, status in
            if consumed { status.pointee = .endOfStream; return nil }
            consumed = true
            status.pointee = .haveData
            return readBuf
        }

        if let convError { throw convError }
        return extractSamples(from: outBuf)
    }

    private func extractSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
        let count = Int(buffer.frameLength)
        guard count > 0, let data = buffer.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: data, count: count))
    }
}

// MARK: - Supporting Types

struct BatchSegment: Sendable {
    let text: String
    let startTime: Double  // seconds from audio start
    let endTime: Double
}

enum BatchTranscriptionError: LocalizedError {
    case notPrepared
    case alreadyRunning
    case audioFileNotFound(String)
    case audioFormatError

    var errorDescription: String? {
        switch self {
        case .notPrepared:
            "Batch transcription engine not prepared. Call prepare() first."
        case .alreadyRunning:
            "A batch transcription is already in progress."
        case .audioFileNotFound(let path):
            "Audio file not found: \(path)"
        case .audioFormatError:
            "Failed to convert audio to required format (16kHz mono)."
        }
    }
}
