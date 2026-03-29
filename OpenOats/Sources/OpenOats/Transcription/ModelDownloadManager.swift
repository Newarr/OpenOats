import AVFoundation
import FluidAudio
import Foundation
import os

/// Enriched download progress info computed from fraction changes over time.
struct DownloadProgressDetail: Sendable {
    let fraction: Double
    /// Formatted string like "142 MB / 800 MB"
    let sizeText: String?
    /// Formatted string like "3.5 MB/s"
    let speedText: String?
    /// Formatted string like "2m 15s remaining"
    let etaText: String?
}

/// Loaded backend instances for mic and system audio transcription.
struct LoadedBackends {
    let mic: any TranscriptionBackend
    let system: any TranscriptionBackend
}

/// Manages transcription model download, loading, and progress tracking.
@MainActor
final class ModelDownloadManager {
    private let settings: AppSettings

    /// Combined progress for UI binding.
    private(set) var needsDownload: Bool = false
    private(set) var downloadProgress: Double?
    private(set) var downloadDetail: DownloadProgressDetail?
    private(set) var isLoading: Bool = false

    /// Callbacks for status updates.
    var onStatusUpdate: (@Sendable (String) -> Void)?
    var onProgressUpdate: (@Sendable (Double) -> Void)?

    // Progress tracking state (not observed)
    private var downloadStartTime: Date?
    private var downloadTotalBytes: Int64?

    init(settings: AppSettings) {
        self.settings = settings
    }

    /// Check if the specified model needs to be downloaded.
    /// - Returns: true if a download is needed
    @discardableResult
    func checkAvailability(for model: TranscriptionModel) -> Bool {
        needsDownload = Self.modelNeedsDownload(model)
        return needsDownload
    }

    /// Load the specified model and return configured backends for mic and system audio.
    /// - Parameters:
    ///   - model: The transcription model to load
    ///   - customVocabulary: Custom vocabulary string to pass to the backend
    /// - Returns: Loaded backends for mic and system audio
    func loadModel(
        _ model: TranscriptionModel,
        customVocabulary: String
    ) async throws -> LoadedBackends {
        guard !isLoading else {
            throw TranscriptionBackendError.notPrepared
        }

        isLoading = true
        defer {
            isLoading = false
            // Clear progress state on both success and failure
            downloadProgress = nil
            downloadDetail = nil
            downloadStartTime = nil
            downloadTotalBytes = nil
        }

        let isDownloading = needsDownload

        if isDownloading {
            downloadProgress = 0
            downloadStartTime = Date()
            downloadTotalBytes = model.estimatedDownloadBytes
            downloadDetail = DownloadProgressDetail(
                fraction: 0,
                sizeText: nil,
                speedText: nil,
                etaText: nil
            )
        }

        Log.transcription.info("loading transcription model \(model.rawValue, privacy: .public)")

        // Load mic backend
        let mic = model.makeBackend(customVocabulary: customVocabulary)
        try await mic.prepare(
            onStatus: { [weak self] status in
                Task { @MainActor in
                    self?.onStatusUpdate?(status)
                }
            },
            onProgress: { [weak self] fraction in
                Task { @MainActor in
                    self?.downloadProgress = fraction
                    self?.updateDownloadDetail(fraction: fraction)
                    self?.onProgressUpdate?(fraction)
                }
            }
        )

        // Parakeet needs a separate backend for system audio (mutable decoder state).
        // Qwen3 is actor-based and thread-safe, so reuse the same instance.
        let system: any TranscriptionBackend
        if model == .qwen3ASR06B {
            system = mic
        } else {
            let sys = model.makeBackend(customVocabulary: customVocabulary)
            try await sys.prepare { _ in }
            system = sys
        }

        // Clear download progress on success
        needsDownload = false

        Log.transcription.info("transcription model loaded")

        return LoadedBackends(mic: mic, system: system)
    }

    /// Clear the model cache for the specified model.
    func clearCache(for model: TranscriptionModel) {
        model.makeBackend().clearModelCache()
        Log.transcription.info("cleared model cache for \(model.rawValue, privacy: .public)")
    }

    // MARK: - Private Helpers

    private static func modelNeedsDownload(_ model: TranscriptionModel) -> Bool {
        let backend = model.makeBackend()
        if case .needsDownload = backend.checkStatus() {
            return true
        }
        return false
    }

    private func updateDownloadDetail(fraction: Double) {
        guard let startTime = downloadStartTime else {
            downloadDetail = DownloadProgressDetail(
                fraction: fraction,
                sizeText: nil,
                speedText: nil,
                etaText: nil
            )
            return
        }

        let elapsed = Date().timeIntervalSince(startTime)
        let totalBytes = downloadTotalBytes

        // Size text: "142 MB / 800 MB" (only when total is known)
        var sizeText: String?
        if let totalBytes {
            let downloaded = Int64(fraction * Double(totalBytes))
            sizeText = "\(Self.formatBytes(downloaded)) / \(Self.formatBytes(totalBytes))"
        }

        // Speed and ETA need enough elapsed time to be meaningful
        var speedText: String?
        var etaText: String?
        if elapsed > 1, fraction > 0.01 {
            // Speed from fraction progress rate + known total
            if let totalBytes {
                let bytesDownloaded = fraction * Double(totalBytes)
                let bytesPerSecond = bytesDownloaded / elapsed
                speedText = "\(Self.formatBytes(Int64(bytesPerSecond)))/s"

                let remaining = Double(totalBytes) - bytesDownloaded
                if bytesPerSecond > 0 {
                    let secondsLeft = remaining / bytesPerSecond
                    etaText = Self.formatDuration(secondsLeft)
                }
            } else {
                // No total bytes known — estimate ETA from fraction rate alone
                let fractionPerSecond = fraction / elapsed
                if fractionPerSecond > 0 {
                    let remainingFraction = 1.0 - fraction
                    let secondsLeft = remainingFraction / fractionPerSecond
                    etaText = Self.formatDuration(secondsLeft)
                }
            }
        }

        downloadDetail = DownloadProgressDetail(
            fraction: fraction,
            sizeText: sizeText,
            speedText: speedText,
            etaText: etaText
        )
    }

    private static func formatBytes(_ bytes: Int64) -> String {
        if bytes >= 1_000_000_000 {
            return String(format: "%.1f GB", Double(bytes) / 1_000_000_000)
        } else {
            return String(format: "%.0f MB", Double(bytes) / 1_000_000)
        }
    }

    private static func formatDuration(_ seconds: Double) -> String {
        let s = Int(seconds)
        if s < 60 { return "\(s)s remaining" }
        let m = s / 60
        let rem = s % 60
        return rem > 0 ? "\(m)m \(rem)s remaining" : "\(m)m remaining"
    }
}
