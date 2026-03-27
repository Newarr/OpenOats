@preconcurrency import AVFoundation
import Foundation
import os

/// Runs a high-accuracy batch re-transcription of a recorded session using
/// Cohere Transcribe (CoreML). Designed to run asynchronously after a meeting
/// ends, replacing the live transcript with a more accurate version.
///
/// The engine processes saved audio files through the Cohere Transcribe CoreML
/// model in chunks and produces refined text segments with timestamps.
actor BatchTranscriptionEngine {
    private let log = Logger(subsystem: "com.openoats", category: "BatchTranscription")
    private var backend: CohereTranscribeBackend?
    private var isRunning = false

    /// Prepare the Cohere Transcribe backend (download + load CoreML models).
    /// Safe to call multiple times; only the first call does work.
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
    ///   - onProgress: Called with (completedChunks, totalChunks) for UI updates.
    /// - Returns: Array of transcribed text segments with approximate timestamps.
    func transcribe(
        audioURL: URL,
        locale: Locale,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
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

        let samples = try loadAudio(from: audioURL)
        guard !samples.isEmpty else {
            log.warning("Audio file contained no samples: \(audioURL.lastPathComponent)")
            return []
        }

        let duration = Double(samples.count) / 16_000
        log.info("Loaded \(samples.count) samples (\(String(format: "%.1f", duration))s)")

        // Process in 30-second chunks. Batch mode — latency doesn't matter.
        let chunkDuration = 30
        let chunkSamples = chunkDuration * 16_000
        let totalChunks = max(1, (samples.count + chunkSamples - 1) / chunkSamples)
        var segments: [BatchSegment] = []

        for chunkIdx in 0..<totalChunks {
            let offset = chunkIdx * chunkSamples
            let end = min(offset + chunkSamples, samples.count)

            let startTime = Double(offset) / 16_000
            let endTime = Double(end) / 16_000

            // Pass a slice view to avoid per-chunk array allocation
            let text = try await samples.withUnsafeBufferPointer { buf in
                let chunkSlice = Array(UnsafeBufferPointer(
                    start: buf.baseAddress! + offset,
                    count: end - offset
                ))
                return try await backend.transcribe(chunkSlice, locale: locale)
            }

            if !text.isEmpty {
                segments.append(BatchSegment(
                    text: text,
                    startTime: startTime,
                    endTime: endTime
                ))
            }

            onProgress?(chunkIdx + 1, totalChunks)
            log.info("Chunk \(chunkIdx + 1)/\(totalChunks): \(String(format: "%.0f", startTime))s-\(String(format: "%.0f", endTime))s")
        }

        log.info("Batch transcription complete: \(segments.count) segments from \(totalChunks) chunks")
        return segments
    }

    // MARK: - Audio Loading

    /// Load audio from an M4A file and convert to 16kHz mono Float32 samples.
    private func loadAudio(from url: URL) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw BatchTranscriptionError.audioFileNotFound(url.path)
        }

        let file = try AVAudioFile(forReading: url)
        let srcFormat = file.processingFormat
        let totalFrames = file.length

        // Guard against overflow: AVAudioFrameCount is UInt32 (~74 hours at 16kHz)
        guard totalFrames > 0 else {
            throw BatchTranscriptionError.audioEmpty
        }
        guard totalFrames <= Int64(UInt32.max) else {
            throw BatchTranscriptionError.audioTooLong(
                seconds: Double(totalFrames) / srcFormat.sampleRate
            )
        }

        let frameCount = AVAudioFrameCount(totalFrames)
        guard let readBuf = AVAudioPCMBuffer(
            pcmFormat: srcFormat, frameCapacity: frameCount
        ) else {
            throw BatchTranscriptionError.audioFormatError
        }
        try file.read(into: readBuf)

        // Target: 16kHz mono
        let targetRate: Double = 16_000
        guard let targetFormat = AVAudioFormat(
            standardFormatWithSampleRate: targetRate, channels: 1
        ) else {
            throw BatchTranscriptionError.audioFormatError
        }

        // If already at target format, extract directly
        if srcFormat.sampleRate == targetRate && srcFormat.channelCount == 1 {
            return extractSamples(from: readBuf)
        }

        // Resample via AVAudioConverter
        guard let converter = AVAudioConverter(from: srcFormat, to: targetFormat) else {
            throw BatchTranscriptionError.audioConversionFailed(
                from: "\(srcFormat.sampleRate)Hz/\(srcFormat.channelCount)ch",
                to: "16000Hz/1ch"
            )
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

        if let convError {
            throw BatchTranscriptionError.audioConversionFailed(
                from: "\(srcFormat.sampleRate)Hz", to: convError.localizedDescription
            )
        }

        let result = extractSamples(from: outBuf)
        guard !result.isEmpty else {
            throw BatchTranscriptionError.audioConversionFailed(
                from: "\(srcFormat.sampleRate)Hz/\(srcFormat.channelCount)ch",
                to: "conversion produced 0 samples"
            )
        }
        return result
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
    case audioEmpty
    case audioTooLong(seconds: Double)
    case audioFormatError
    case audioConversionFailed(from: String, to: String)

    var errorDescription: String? {
        switch self {
        case .notPrepared:
            "Batch transcription engine not prepared. Call prepare() first."
        case .alreadyRunning:
            "A batch transcription is already in progress."
        case .audioFileNotFound(let path):
            "Audio file not found: \(path)"
        case .audioEmpty:
            "Audio file is empty."
        case .audioTooLong(let seconds):
            "Audio file too long for batch processing: \(String(format: "%.0f", seconds))s"
        case .audioFormatError:
            "Failed to create audio format or buffer."
        case .audioConversionFailed(let from, let to):
            "Audio conversion failed: \(from) → \(to)"
        }
    }
}
