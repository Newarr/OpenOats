import AVFoundation
import FluidAudio
import Foundation
import os

/// Errors that can occur during stream coordination with user-facing messages.
enum TranscriptionStreamError: LocalizedError {
    case microphonePermissionDenied
    case microphoneUnavailable
    case systemAudioCaptureFailed(Error)
    case transcriberCreationFailed(Speaker)

    var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Microphone access was denied. Enable it in System Settings > Privacy & Security > Microphone."
        case .microphoneUnavailable:
            return "The selected microphone is no longer available."
        case .systemAudioCaptureFailed:
            return "System audio capture failed to start."
        case .transcriberCreationFailed(let speaker):
            return "Failed to create the \(speaker == .you ? "microphone" : "system audio") transcriber. Try restarting."
        }
    }
}

/// Coordinates audio stream capture, transcribers, and optional recording.
/// Manages the lifecycle of mic and system audio transcription tasks.
@MainActor
final class TranscriptionStreamCoordinator {
    private let micCapture: MicCapture
    private let systemCapture: SystemAudioCapture

    init(micCapture: MicCapture = MicCapture(), systemCapture: SystemAudioCapture = SystemAudioCapture()) {
        self.micCapture = micCapture
        self.systemCapture = systemCapture
    }

    /// Audio recorder for tapping streams (set externally when recording is enabled).
    weak var audioRecorder: AudioRecorder?

    /// Combined audio level (mic + system) for the UI meter.
    nonisolated var audioLevel: Float {
        max(micCapture.audioLevel, systemCapture.audioLevel)
    }

    /// Mute/unmute the microphone. When muted, mic audio is not transcribed
    /// and the audio level reads as 0. System audio continues normally.
    nonisolated var isMicMuted: Bool {
        get { micCapture.isMuted }
        set { micCapture.isMuted = newValue }
    }

    /// Mic transcription task.
    private var micTask: Task<Void, Never>?

    /// System audio transcription task.
    private var sysTask: Task<Void, Never>?

    /// Health check task for mic audio.
    private var micHealthTask: Task<Void, Never>?

    /// Diarization audio feed task.
    private var diarizationTask: Task<Void, Never>?

    /// Track if mic is currently running for health checks.
    private var isMicRunning = false

    /// Start the mic audio stream and transcription.
    /// - Returns: The transcription task on success, or an error on failure
    @discardableResult
    func startMicStream(
        locale: Locale,
        deviceID: AudioDeviceID,
        backend: any TranscriptionBackend,
        vadManager: VadManager,
        transcriptStore: TranscriptStore,
        flushInterval: Int,
        useAEC: Bool
    ) -> Result<Task<Void, Never>, TranscriptionStreamError> {
        // Store state for potential restarts
        isMicRunning = true

        var micStream = micCapture.bufferStream(deviceID: deviceID, echoCancellation: useAEC)

        // Check for immediate mic capture failure
        if let micError = micCapture.captureError {
            Log.transcription.error("Mic capture setup error: \(micError)")
            isMicRunning = false
            return .failure(.microphoneUnavailable)
        }

        // Add recording tap if recorder is set
        if let recorder = audioRecorder {
            micStream = Self.tappedStream(micStream) { buffer in
                recorder.writeMicBuffer(buffer)
            }
        }

        guard let micTranscriber = makeTranscriber(
            backend: backend,
            locale: locale,
            vadManager: vadManager,
            speaker: .you,
            transcriptStore: transcriptStore,
            flushInterval: flushInterval
        ) else {
            Log.transcription.error("Failed to create mic transcriber")
            isMicRunning = false
            return .failure(.transcriberCreationFailed(.you))
        }

        micTask = Task { [weak self] in
            await micTranscriber.run(stream: micStream)
            await self?.markMicStopped()
        }

        // Health check: if mic produces no audio within 5 seconds, log the issue
        micHealthTask?.cancel()
        micHealthTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(5))
            guard let self, self.isMicRunning else { return }
            if !self.micCapture.hasCapturedFrames && self.micCapture.captureError == nil {
                Log.transcription.error("no mic audio after 5s")
            }
        }

        return .success(micTask!)
    }

    /// Mark mic as stopped - MainActor-isolated to prevent concurrency violations
    private func markMicStopped() {
        isMicRunning = false
    }

    /// Stop the mic audio stream and transcription.
    func stopMicStream() {
        micHealthTask?.cancel()
        micHealthTask = nil
        micCapture.finishStream()
        micTask?.cancel()
        micTask = nil
        micCapture.stop()
        isMicRunning = false
    }

    /// Finalize mic stream, waiting for transcriber to drain.
    func finalizeMicStream() async {
        micHealthTask?.cancel()
        micHealthTask = nil
        micCapture.finishStream()
        await micTask?.value
        micCapture.stop()
        micTask = nil
        isMicRunning = false
    }

    /// Start the system audio stream and transcription.
    /// - Returns: The transcription task on success, or an error on failure
    @discardableResult
    func startSystemStream(
        locale: Locale,
        backend: any TranscriptionBackend,
        vadManager: VadManager,
        diarizationManager: DiarizationManager?,
        transcriptStore: TranscriptStore,
        flushInterval: Int
    ) async -> Result<Task<Void, Never>, TranscriptionStreamError> {
        Log.transcription.info("starting system audio capture")

        let sysStreams: SystemAudioCapture.CaptureStreams
        do {
            sysStreams = try await systemCapture.bufferStream()
            Log.transcription.info("system audio capture started")
        } catch {
            Log.transcription.error("Failed to start system audio: \(error.localizedDescription, privacy: .public)")
            return .failure(.systemAudioCaptureFailed(error))
        }

        var sysStream = sysStreams.systemAudio

        // Track cumulative audio time for diarizer speaker attribution
        let sysAudioTime = SyncDouble()

        // Tee system audio to diarization manager if enabled
        if let dm = diarizationManager {
            let diarFlushSize = 16000
            let originalSysStream = sysStream
            let (diarTapped, diarContinuation) = AsyncStream<AVAudioPCMBuffer>.makeStream(bufferingPolicy: .bufferingNewest(1))

            diarizationTask?.cancel()
            diarizationTask = Task { [weak self, dm] in
                var diarBuf: [Float] = []
                for await buffer in originalSysStream {
                    // Check for cancellation
                    guard !Task.isCancelled else { break }
                    diarContinuation.yield(buffer)
                    guard let channelData = buffer.floatChannelData else { continue }
                    let frameCount = Int(buffer.frameLength)
                    sysAudioTime.add(Double(frameCount) / buffer.format.sampleRate)
                    diarBuf.append(contentsOf: UnsafeBufferPointer(start: channelData[0], count: frameCount))
                    if diarBuf.count >= diarFlushSize {
                        let batch = diarBuf
                        diarBuf.removeAll(keepingCapacity: true)
                        try? await dm.feedAudio(batch)
                    }
                }
                // Flush tail
                if !Task.isCancelled, !diarBuf.isEmpty {
                    try? await dm.feedAudio(diarBuf)
                }
                diarContinuation.finish()
                self?.diarizationTask = nil
            }
            sysStream = diarTapped
        }

        // Add recording tap if recorder is set
        if let recorder = audioRecorder {
            sysStream = Self.tappedStream(sysStream) { buffer in
                recorder.writeSysBuffer(buffer)
            }
        }

        guard let sysTranscriber = makeTranscriber(
            backend: backend,
            locale: locale,
            vadManager: vadManager,
            speaker: .them,
            transcriptStore: transcriptStore,
            flushInterval: flushInterval,
            diarizationManager: diarizationManager,
            audioTime: diarizationManager != nil ? sysAudioTime : nil
        ) else {
            Log.transcription.error("Failed to create system audio transcriber")
            return .failure(.transcriberCreationFailed(.them))
        }

        sysTask = Task {
            await sysTranscriber.run(stream: sysStream)
        }

        return .success(sysTask!)
    }

    /// Stop the system audio stream and transcription.
    func stopSystemStream() {
        diarizationTask?.cancel()
        systemCapture.finishStream()
        sysTask?.cancel()
        sysTask = nil
        Task {
            await diarizationTask?.value
            await systemCapture.stop()
            diarizationTask = nil
        }
    }

    /// Finalize system stream, waiting for transcriber to drain.
    func finalizeSystemStream() async {
        diarizationTask?.cancel()
        systemCapture.finishStream()
        await sysTask?.value
        await diarizationTask?.value
        await systemCapture.stop()
        sysTask = nil
        diarizationTask = nil
    }

    /// Stop all audio streams and transcription.
    func stopAll() {
        stopMicStream()
        stopSystemStream()
    }

    /// Finalize all streams and wait for tasks to complete.
    func finalize() async {
        micHealthTask?.cancel()
        micHealthTask = nil

        diarizationTask?.cancel()

        micCapture.finishStream()
        systemCapture.finishStream()

        await micTask?.value
        await sysTask?.value
        await diarizationTask?.value

        micCapture.stop()
        await systemCapture.stop()

        micTask = nil
        sysTask = nil
        diarizationTask = nil
        isMicRunning = false
    }

    // MARK: - Private Helpers

    private func makeTranscriber(
        backend: any TranscriptionBackend,
        locale: Locale,
        vadManager: VadManager,
        speaker: Speaker,
        transcriptStore: TranscriptStore,
        flushInterval: Int,
        diarizationManager: DiarizationManager? = nil,
        audioTime: SyncDouble? = nil
    ) -> StreamingTranscriber? {
        let store = transcriptStore

        let onPartial: @Sendable (String) -> Void
        let onFinal: @Sendable (String) -> Void

        if speaker == .you {
            onPartial = { text in
                Task { @MainActor in store.volatileYouText = text }
            }
            onFinal = { text in
                Task { @MainActor in
                    store.volatileYouText = ""
                    store.append(Utterance(text: text, speaker: .you))
                }
            }
        } else {
            let dm = diarizationManager
            let time = audioTime
            onPartial = { text in
                Task { @MainActor in store.volatileThemText = text }
            }
            onFinal = { text in
                Task { @MainActor in
                    store.volatileThemText = ""
                    let finalSpeaker: Speaker
                    if let dm, let t = time {
                        let endTime = t.value
                        let startTime = max(0, endTime - 5.0)
                        finalSpeaker = await dm.dominantSpeaker(from: startTime, to: endTime)
                    } else {
                        finalSpeaker = .them
                    }
                    store.append(Utterance(text: text, speaker: finalSpeaker))
                }
            }
        }

        return StreamingTranscriber(
            backend: backend,
            locale: locale,
            vadManager: vadManager,
            speaker: speaker,
            flushInterval: flushInterval,
            onPartial: onPartial,
            onFinal: onFinal
        )
    }

    /// Wrap an audio stream to forward each buffer to a synchronous tap before yielding it downstream.
    private nonisolated static func tappedStream(
        _ stream: AsyncStream<AVAudioPCMBuffer>,
        tap: @escaping @Sendable (AVAudioPCMBuffer) -> Void
    ) -> AsyncStream<AVAudioPCMBuffer> {
        let (output, continuation) = AsyncStream<AVAudioPCMBuffer>.makeStream(bufferingPolicy: .bufferingNewest(1))
        Task {
            for await buffer in stream {
                tap(buffer)
                continuation.yield(buffer)
            }
            continuation.finish()
        }
        return output
    }
}
