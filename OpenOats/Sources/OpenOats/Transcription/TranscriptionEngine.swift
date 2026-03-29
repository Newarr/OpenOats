import AVFoundation
import CoreAudio
import FluidAudio
import Observation
import os

enum TranscriptionEngineError: LocalizedError {
    case transcriberNotInitialized

    var errorDescription: String? {
        switch self {
        case .transcriberNotInitialized:
            "Transcription engine is not initialized. Please check your audio settings."
        }
    }
}

/// Orchestrates dual StreamingTranscriber instances for mic (you) and system audio (them).
@Observable
@MainActor
final class TranscriptionEngine {
    enum Mode {
        case live
        case scripted([Utterance])
    }

    // These properties are read from SwiftUI body during view evaluation.
    // SwiftUI's ViewBodyAccessor doesn't carry MainActor executor context
    // in Swift 6.2, so @MainActor-isolated @Observable properties trigger
    // a failing runtime check in SerialExecutor.isMainExecutor.getter
    // (EXC_BAD_ACCESS / KERN_PROTECTION_FAILURE).
    //
    // We use @ObservationIgnored nonisolated(unsafe) backing storage with
    // manual observation tracking to bypass the MainActor check while
    // keeping SwiftUI reactivity. Mutations only happen on MainActor.
    @ObservationIgnored nonisolated(unsafe) private var _isRunning = false
    var isRunning: Bool {
        get { access(keyPath: \.isRunning); return _isRunning }
        set { withMutation(keyPath: \.isRunning) { _isRunning = newValue } }
    }

    @ObservationIgnored nonisolated(unsafe) private var _assetStatus: String = "Ready"
    var assetStatus: String {
        get { access(keyPath: \.assetStatus); return _assetStatus }
        set { withMutation(keyPath: \.assetStatus) { _assetStatus = newValue } }
    }

    @ObservationIgnored nonisolated(unsafe) private var _lastError: String?
    var lastError: String? {
        get { access(keyPath: \.lastError); return _lastError }
        set { withMutation(keyPath: \.lastError) { _lastError = newValue } }
    }

    @ObservationIgnored nonisolated(unsafe) private var _needsModelDownload = false
    var needsModelDownload: Bool {
        get { access(keyPath: \.needsModelDownload); return _needsModelDownload }
        set { withMutation(keyPath: \.needsModelDownload) { _needsModelDownload = newValue } }
    }

    @ObservationIgnored nonisolated(unsafe) private var _downloadConfirmed = false
    var downloadConfirmed: Bool {
        get { access(keyPath: \.downloadConfirmed); return _downloadConfirmed }
        set { withMutation(keyPath: \.downloadConfirmed) { _downloadConfirmed = newValue } }
    }

    @ObservationIgnored nonisolated(unsafe) private var _downloadProgress: Double?
    /// Fraction complete (0…1) during model download, nil when not downloading.
    var downloadProgress: Double? {
        get { access(keyPath: \.downloadProgress); return _downloadProgress }
        set { withMutation(keyPath: \.downloadProgress) { _downloadProgress = newValue } }
    }

    @ObservationIgnored nonisolated(unsafe) private var _downloadDetail: DownloadProgressDetail?
    var downloadDetail: DownloadProgressDetail? {
        get { access(keyPath: \.downloadDetail); return _downloadDetail }
        set { withMutation(keyPath: \.downloadDetail) { _downloadDetail = newValue } }
    }

    private let transcriptStore: TranscriptStore
    private let settings: AppSettings
    private let mode: Mode

    private let modelDownloadManager: ModelDownloadManager
    private let deviceRoutingManager: DeviceRoutingManager
    private let streamCoordinator: TranscriptionStreamCoordinator

    /// Combined audio level (mic + system) for the UI meter.
    /// nonisolated is safe here — both audioLevel properties are thread-safe (NSLock).
    nonisolated var audioLevel: Float {
        switch mode {
        case .live:
            streamCoordinator.audioLevel
        case .scripted:
            _isRunning ? 0.35 : 0
        }
    }

    /// Mute/unmute the microphone. When muted, mic audio is not transcribed
    /// and the audio level reads as 0. System audio continues normally.
    nonisolated var isMicMuted: Bool {
        get { streamCoordinator.isMicMuted }
        set { streamCoordinator.isMicMuted = newValue }
    }

    private var micTask: Task<Void, Never>?
    private var sysTask: Task<Void, Never>?
    /// Keeps the mic stream alive for the audio level meter when transcription isn't running.
    private var micKeepAliveTask: Task<Void, Never>?

    /// Separate backend instances for mic and system audio.
    /// Parakeet keeps mutable decoder state per manager, so mic and system audio
    /// need separate instances even when they share the same loaded model files.
    /// For Qwen3 (actor-based, thread-safe), both point to the same backend instance.
    private var micBackend: (any TranscriptionBackend)?
    private var systemBackend: (any TranscriptionBackend)?
    private var vadManager: VadManager?

    /// Audio recorder for tapping streams (set by ContentView when recording is enabled).
    var audioRecorder: AudioRecorder? {
        didSet {
            streamCoordinator.audioRecorder = audioRecorder
        }
    }

    /// Speaker diarization manager for system audio (nil when diarization is disabled).
    private var diarizationManager: DiarizationManager?

    init(transcriptStore: TranscriptStore, settings: AppSettings, mode: Mode = .live) {
        self.transcriptStore = transcriptStore
        self.settings = settings
        self.mode = mode

        self.modelDownloadManager = ModelDownloadManager(settings: settings)
        self.deviceRoutingManager = DeviceRoutingManager()
        self.streamCoordinator = TranscriptionStreamCoordinator()

        // Wire up callbacks
        self.deviceRoutingManager.onMicRestartRequested = { [weak self] deviceID in
            await self?.performMicRestart(deviceID: deviceID)
        }
        self.deviceRoutingManager.onSystemRestartRequested = { [weak self] in
            await self?.performSystemAudioRestart()
        }

        switch mode {
        case .live:
            self.needsModelDownload = modelDownloadManager.checkAvailability(for: settings.transcriptionModel)
        case .scripted:
            self.needsModelDownload = false
        }
    }

    func refreshModelAvailability() {
        switch mode {
        case .live:
            needsModelDownload = modelDownloadManager.checkAvailability(for: settings.transcriptionModel)
        case .scripted:
            needsModelDownload = false
        }
    }

    func start(
        locale: Locale,
        inputDeviceID: AudioDeviceID = 0,
        transcriptionModel: TranscriptionModel
    ) async {
        Log.transcription.info("start() called, isRunning=\(self.isRunning, privacy: .public)")
        guard !isRunning else { return }
        lastError = nil
        refreshModelAvailability()

        if case .scripted(let scriptedUtterances) = mode {
            downloadConfirmed = false
            assetStatus = "Transcribing (UI Test)"
            isRunning = true
            for utterance in scriptedUtterances {
                transcriptStore.append(utterance)
            }
            return
        }

        if let localeMismatchMessage = localeMismatchMessage(
            for: locale,
            transcriptionModel: transcriptionModel
        ) {
            lastError = localeMismatchMessage
            assetStatus = "Ready"
            return
        }

        // Block start if models need downloading and user hasn't confirmed
        if needsModelDownload && !downloadConfirmed {
            return
        }

        guard await ensureMicrophonePermission() else { return }

        isRunning = true

        // 1. Load transcription models via manager
        assetStatus = "Loading transcription model..."
        let backends: LoadedBackends
        do {
            backends = try await modelDownloadManager.loadModel(
                settings.transcriptionModel,
                customVocabulary: settings.transcriptionCustomVocabulary
            )
            self.micBackend = backends.mic
            self.systemBackend = backends.system
        } catch {
            lastError = "Failed to load models: \(error.localizedDescription)"
            assetStatus = "Ready"
            isRunning = false
            modelDownloadManager.clearCache(for: settings.transcriptionModel)
            return
        }

        // Sync observable properties from manager
        self.needsModelDownload = modelDownloadManager.needsDownload
        self.downloadProgress = modelDownloadManager.downloadProgress
        self.downloadDetail = modelDownloadManager.downloadDetail

        // Load VAD model
        assetStatus = "Loading VAD model..."
        Log.transcription.info("Loading VAD model")
        do {
            let vad = try await VadManager()
            self.vadManager = vad
        } catch {
            lastError = "Failed to load VAD model: \(error.localizedDescription)"
            assetStatus = "Ready"
            isRunning = false
            return
        }

        // Optionally load speaker diarization model
        if settings.enableDiarization {
            assetStatus = "Loading diarization model..."
            Log.transcription.info("Loading LS-EEND diarization model")
            let dm = DiarizationManager()
            let variant = LSEENDVariant(rawValue: settings.diarizationVariant.rawValue) ?? .dihard3
            do {
                try await dm.load(variant: variant)
                self.diarizationManager = dm
                Log.transcription.info("Diarization model loaded")
            } catch {
                Log.transcription.error("Failed to load diarization model: \(error, privacy: .public)")
                // Non-fatal: continue without diarization
                self.diarizationManager = nil
            }
        } else {
            self.diarizationManager = nil
        }

        assetStatus = "Models ready"
        Log.transcription.info("Transcription model loaded")

        guard let vadManager else { return }

        // 2. Resolve mic device and start listening for device changes
        guard let targetMicID = deviceRoutingManager.resolvedMicDeviceID(for: inputDeviceID) else {
            let msg = deviceRoutingManager.unavailableMicMessage(for: inputDeviceID)
            Log.transcription.error("Mic unavailable: \(msg, privacy: .public)")
            lastError = msg
            assetStatus = "Ready"
            isRunning = false
            return
        }

        deviceRoutingManager.startListening(isUsingDefaultDevice: inputDeviceID == 0)
        deviceRoutingManager.updateCurrentDeviceID(targetMicID, isUserSelection: true)

        // AEC (voice processing) conflicts with system audio capture on macOS
        let useAEC = false
        if settings.enableEchoCancellation {
            Log.transcription.info("AEC disabled - conflicts with system audio capture")
        }

        // 3. Start mic stream via coordinator
        Log.transcription.info("Starting mic capture, targetMicID=\(targetMicID, privacy: .public), aec=\(useAEC, privacy: .public)")
        let micResult = streamCoordinator.startMicStream(
            locale: locale,
            deviceID: targetMicID,
            backend: micBackend!,
            vadManager: vadManager,
            transcriptStore: transcriptStore,
            flushInterval: settings.transcriptionModel.flushIntervalSamples,
            useAEC: useAEC
        )
        switch micResult {
        case .success(let task):
            self.micTask = task
        case .failure(let error):
            lastError = error.localizedDescription
            isRunning = false
            return
        }

        // 4. Start system audio stream via coordinator
        let sysResult = await streamCoordinator.startSystemStream(
            locale: locale,
            backend: systemBackend!,
            vadManager: vadManager,
            diarizationManager: diarizationManager,
            transcriptStore: transcriptStore,
            flushInterval: settings.transcriptionModel.flushIntervalSamples
        )
        switch sysResult {
        case .success(let task):
            self.sysTask = task
        case .failure(let error):
            lastError = error.localizedDescription
        }

        assetStatus = "Transcribing (\(micBackend?.displayName ?? transcriptionModel.displayName))"
        Log.transcription.info("All transcription tasks started")

        // Store state for restarts
        deviceRoutingManager.storeRestartState(
            locale: locale,
            vadManager: vadManager,
            micBackend: micBackend!,
            systemBackend: systemBackend!,
            flushInterval: settings.transcriptionModel.flushIntervalSamples,
            transcriptStore: transcriptStore
        )
    }

    /// Restart only the mic capture with a new device, keeping system audio and models intact.
    /// Pass the raw setting value (0 = system default, or a specific AudioDeviceID).
    func restartMic(inputDeviceID: AudioDeviceID) {
        if case .scripted = mode { return }
        guard isRunning else { return }
        deviceRoutingManager.requestMicRestart(deviceID: inputDeviceID)
    }

    private func ensureMicrophonePermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .audio)
            if !granted {
                lastError = "Microphone access denied. Enable it in System Settings > Privacy & Security > Microphone."
                assetStatus = "Ready"
            }
            return granted
        case .denied, .restricted:
            lastError = "Microphone access is disabled. Enable it in System Settings > Privacy & Security > Microphone."
            assetStatus = "Ready"
            return false
        @unknown default:
            lastError = "Unable to verify microphone permission."
            assetStatus = "Ready"
            return false
        }
    }

    func finalize() async {
        Log.transcription.info("finalize() called")

        if case .scripted = mode {
            isRunning = false
            assetStatus = "Ready"
            transcriptStore.volatileYouText = ""
            transcriptStore.volatileThemText = ""
            return
        }

        isRunning = false
        assetStatus = "Finalizing..."

        // Stop listening for device changes
        deviceRoutingManager.stopListening()

        // Finalize streams via coordinator
        await streamCoordinator.finalize()

        // Finalize and release diarization manager
        if let dm = diarizationManager {
            await dm.finalize()
        }
        diarizationManager = nil

        micBackend = nil
        systemBackend = nil
        micTask = nil
        sysTask = nil
        transcriptStore.volatileYouText = ""
        transcriptStore.volatileThemText = ""

        assetStatus = "Ready"
        Log.transcription.info("finalize() completed")
    }

    func stop() {
        Log.transcription.info("stop() called")

        if case .scripted = mode {
            isRunning = false
            assetStatus = "Ready"
            transcriptStore.volatileYouText = ""
            transcriptStore.volatileThemText = ""
            return
        }

        isRunning = false
        assetStatus = "Ready"

        micKeepAliveTask?.cancel()
        micKeepAliveTask = nil

        // Stop device listeners
        deviceRoutingManager.stopListening()

        // Stop streams via coordinator
        streamCoordinator.stopAll()

        micTask = nil
        sysTask = nil
        micBackend = nil
        systemBackend = nil
        diarizationManager = nil
        transcriptStore.volatileYouText = ""
        transcriptStore.volatileThemText = ""

        Log.transcription.info("stop() completed")
    }

    // MARK: - Restart Implementation (Called by DeviceRoutingManager)

    private func performMicRestart(deviceID: AudioDeviceID) async {
        guard isRunning, let vadManager else { return }

        guard let targetMicID = deviceRoutingManager.resolvedMicDeviceID(for: deviceID) else {
            Log.transcription.error("Mic swap failed: device unavailable")
            return
        }

        guard targetMicID != deviceRoutingManager.currentMicID else {
            Log.transcription.debug("Mic swap skipped, same device \(targetMicID, privacy: .public)")
            return
        }

        Log.transcription.info("Switching mic to \(targetMicID, privacy: .public)")

        // Stop current mic stream
        streamCoordinator.stopMicStream()
        micTask = nil

        if Task.isCancelled || !isRunning {
            return
        }

        // Start new mic stream
        guard let micBackend else { return }
        let micResult = streamCoordinator.startMicStream(
            locale: deviceRoutingManager.currentLocale ?? settings.locale,
            deviceID: targetMicID,
            backend: micBackend,
            vadManager: vadManager,
            transcriptStore: transcriptStore,
            flushInterval: deviceRoutingManager.currentFlushInterval ?? settings.transcriptionModel.flushIntervalSamples,
            useAEC: false
        )
        switch micResult {
        case .success(let task):
            self.micTask = task
            deviceRoutingManager.updateCurrentDeviceID(targetMicID, isUserSelection: false)
            lastError = nil
            Log.transcription.info("Mic restarted on device \(targetMicID, privacy: .public)")
        case .failure(let error):
            lastError = error.localizedDescription
        }
    }

    private func performSystemAudioRestart() async {
        guard isRunning, let vadManager else { return }

        Log.transcription.info("Restarting system audio stream")

        // Stop current system stream
        await streamCoordinator.finalizeSystemStream()
        sysTask = nil

        if Task.isCancelled || !isRunning {
            return
        }

        // Start new system stream
        guard let systemBackend else { return }
        let sysResult = await streamCoordinator.startSystemStream(
            locale: deviceRoutingManager.currentLocale ?? settings.locale,
            backend: systemBackend,
            vadManager: vadManager,
            diarizationManager: diarizationManager,
            transcriptStore: transcriptStore,
            flushInterval: deviceRoutingManager.currentFlushInterval ?? settings.transcriptionModel.flushIntervalSamples
        )
        switch sysResult {
        case .success(let task):
            self.sysTask = task
            Log.transcription.info("System audio stream restarted")
        case .failure(let error):
            lastError = error.localizedDescription
        }
    }

    // MARK: - Utility Methods

    private func localeMismatchMessage(
        for locale: Locale,
        transcriptionModel: TranscriptionModel
    ) -> String? {
        guard transcriptionModel == .parakeetV2,
              let languageCode = normalizedLanguageCode(for: locale),
              languageCode != "en"
        else {
            return nil
        }

        let localeIdentifier = locale.identifier.replacingOccurrences(of: "_", with: "-")
        return "Parakeet TDT v2 is English-only. Switch to Parakeet TDT v3 or Qwen3 ASR for \(localeIdentifier)."
    }

    private func normalizedLanguageCode(for locale: Locale) -> String? {
        let identifier = locale.identifier.replacingOccurrences(of: "_", with: "-")
        return identifier.split(separator: "-").first.map { String($0).lowercased() }
    }

    private func clearSystemAudioErrorIfPresent() {
        guard let lastError else { return }
        if lastError.localizedCaseInsensitiveContains("system audio") ||
            lastError.localizedCaseInsensitiveContains("audio output device") {
            self.lastError = nil
        }
    }
}
