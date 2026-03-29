import CoreAudio
import FluidAudio
import Foundation
import os

/// Manages CoreAudio device listeners and coordinates audio stream restarts
/// when the default input or output device changes.
@MainActor
final class DeviceRoutingManager {
    /// Shared queue for all CoreAudio property listener registrations.
    /// Must use the same queue instance for Add and Remove calls.
    private static let listenerQueue = DispatchQueue(label: "com.openoats.device-routing", qos: .utility)

    /// Called when a mic restart is requested with the target device ID.
    var onMicRestartRequested: (@Sendable (AudioDeviceID) async -> Void)?

    /// Called when a system audio restart is requested.
    var onSystemRestartRequested: (@Sendable () async -> Void)?

    /// Tracks whether user selected "System Default" (0) or a specific device.
    private var userSelectedDeviceID: AudioDeviceID = 0

    /// Tracks the resolved mic device ID currently in use.
    private var currentMicDeviceID: AudioDeviceID = 0

    /// Queued mic restart device ID (nil if no pending restart).
    private var pendingMicDeviceID: AudioDeviceID?

    /// Queued system audio restart flag.
    private var pendingSystemAudioRestart = false

    /// Active mic restart task.
    private var micRestartTask: Task<Void, Never>?

    /// Active system restart task.
    private var sysRestartTask: Task<Void, Never>?

    /// Listens for default input device changes at the OS level.
    private var defaultDeviceListenerBlock: AudioObjectPropertyListenerBlock?

    /// Listens for default output device changes at the OS level.
    private var defaultOutputDeviceListenerBlock: AudioObjectPropertyListenerBlock?

    /// Whether device listeners are currently installed.
    private var isListening = false

    /// Stored state for restarts (populated by TranscriptionEngine)
    private(set) var currentLocale: Locale?
    private(set) var currentVadManager: VadManager?
    private(set) var currentMicBackend: (any TranscriptionBackend)?
    private(set) var currentSystemBackend: (any TranscriptionBackend)?
    private(set) var currentFlushInterval: Int?
    private(set) weak var currentTranscriptStore: TranscriptStore?

    /// Store restart state for later use during device changes.
    func storeRestartState(
        locale: Locale,
        vadManager: VadManager,
        micBackend: any TranscriptionBackend,
        systemBackend: any TranscriptionBackend,
        flushInterval: Int,
        transcriptStore: TranscriptStore
    ) {
        currentLocale = locale
        currentVadManager = vadManager
        currentMicBackend = micBackend
        currentSystemBackend = systemBackend
        currentFlushInterval = flushInterval
        currentTranscriptStore = transcriptStore
    }

    /// Start listening for device changes.
    /// - Parameter isUsingDefaultDevice: True if the user has selected "System Default" (deviceID = 0)
    func startListening(isUsingDefaultDevice: Bool) {
        guard !isListening else { return }
        isListening = true

        // Store the default device selection state
        if isUsingDefaultDevice {
            userSelectedDeviceID = 0
        }

        installDefaultDeviceListener()
        installDefaultOutputDeviceListener()
    }

    /// Stop listening for device changes and cancel any pending restarts.
    func stopListening() {
        isListening = false

        removeDefaultDeviceListener()
        removeDefaultOutputDeviceListener()

        micRestartTask?.cancel()
        sysRestartTask?.cancel()
        micRestartTask = nil
        sysRestartTask = nil
        pendingMicDeviceID = nil
        pendingSystemAudioRestart = false

        // Clear restart state to free heavy objects (VadManager, backends)
        currentLocale = nil
        currentVadManager = nil
        currentMicBackend = nil
        currentSystemBackend = nil
        currentFlushInterval = nil
        currentTranscriptStore = nil
    }

    // No deinit: Swift 6 strict concurrency prevents accessing non-Sendable
    // AudioObjectPropertyListenerBlock from nonisolated deinit. The original
    // deinit used [weak self] which was always nil anyway.
    // Contract: callers MUST call stopListening() before releasing this object.

    /// Request a mic restart with a new device.
    /// Pass 0 to use the system default device.
    func requestMicRestart(deviceID: AudioDeviceID) {
        guard isListening else { return }

        pendingMicDeviceID = deviceID

        if micRestartTask != nil {
            Log.transcription.debug("queued mic restart for device \(deviceID, privacy: .public)")
            return
        }

        micRestartTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer { self.micRestartTask = nil }

            while !Task.isCancelled, let requestedDeviceID = self.pendingMicDeviceID {
                self.pendingMicDeviceID = nil
                await self.performMicRestart(deviceID: requestedDeviceID)
            }
        }
    }

    /// Request a system audio restart (e.g., when output device changes).
    func requestSystemAudioRestart() {
        guard isListening else { return }

        pendingSystemAudioRestart = true

        if sysRestartTask != nil {
            Log.transcription.debug("queued system audio restart")
            return
        }

        sysRestartTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer { self.sysRestartTask = nil }

            while !Task.isCancelled, self.pendingSystemAudioRestart {
                self.pendingSystemAudioRestart = false
                await self.performSystemAudioRestart()
            }
        }
    }

    /// Update the tracked device IDs after a successful restart.
    func updateCurrentDeviceID(_ deviceID: AudioDeviceID, isUserSelection: Bool) {
        if isUserSelection {
            userSelectedDeviceID = deviceID
        }
        currentMicDeviceID = deviceID
    }

    /// Get the current mic device ID.
    var currentMicID: AudioDeviceID { currentMicDeviceID }

    /// Get the user selected device ID (0 = system default).
    var selectedDeviceID: AudioDeviceID { userSelectedDeviceID }

    // MARK: - Default Device Listeners

    private func installDefaultDeviceListener() {
        guard defaultDeviceListenerBlock == nil else { return }

        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            guard let self else { return }
            Task { @MainActor in
                // Only auto-restart if user is using system default
                guard self.userSelectedDeviceID == 0 else { return }
                self.requestMicRestart(deviceID: 0)
            }
        }
        defaultDeviceListenerBlock = block

        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            DeviceRoutingManager.listenerQueue,
            block
        )
    }

    private func removeDefaultDeviceListener() {
        guard let block = defaultDeviceListenerBlock else { return }
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectRemovePropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            DeviceRoutingManager.listenerQueue,
            block
        )
        defaultDeviceListenerBlock = nil
    }

    private func installDefaultOutputDeviceListener() {
        guard defaultOutputDeviceListenerBlock == nil else { return }

        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            guard let self else { return }
            Task { @MainActor in
                self.requestSystemAudioRestart()
            }
        }
        defaultOutputDeviceListenerBlock = block

        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            DeviceRoutingManager.listenerQueue,
            block
        )
    }

    private func removeDefaultOutputDeviceListener() {
        guard let block = defaultOutputDeviceListenerBlock else { return }
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectRemovePropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            DeviceRoutingManager.listenerQueue,
            block
        )
        defaultOutputDeviceListenerBlock = nil
    }

    // MARK: - Restart Implementation

    private func performMicRestart(deviceID: AudioDeviceID) async {
        guard !Task.isCancelled else { return }

        userSelectedDeviceID = deviceID

        guard let targetMicID = resolvedMicDeviceID(for: deviceID) else {
            Log.transcription.error("mic swap failed: device unavailable")
            return
        }

        guard targetMicID != currentMicDeviceID else {
            Log.transcription.debug("mic swap skipped, same device \(targetMicID, privacy: .public)")
            return
        }

        Log.transcription.info("switching mic from \(self.currentMicDeviceID, privacy: .public) to \(targetMicID, privacy: .public)")

        // Notify the engine to perform the actual restart
        // Note: State is updated only after successful restart via updateCurrentDeviceID
        await onMicRestartRequested?(targetMicID)
    }

    private func performSystemAudioRestart() async {
        guard !Task.isCancelled else { return }

        Log.transcription.info("restarting system audio stream")

        // Notify the engine to perform the actual restart
        await onSystemRestartRequested?()
    }

    // MARK: - Device Resolution

    /// Generate an error message for when a mic device is unavailable.
    func unavailableMicMessage(for inputDeviceID: AudioDeviceID) -> String {
        if inputDeviceID > 0 {
            return "The selected microphone is no longer available."
        }

        return "No default microphone is currently available."
    }

    /// Check if a specific device ID is available.
    func isDeviceAvailable(_ deviceID: AudioDeviceID) -> Bool {
        if deviceID == 0 {
            return MicCapture.defaultInputDeviceID() != nil
        }
        let available = Set(MicCapture.availableInputDevices().map(\.id))
        return available.contains(deviceID)
    }
}

// MARK: - Public Extension for Engine Access

extension DeviceRoutingManager {
    /// Resolve the target mic device ID, handling "System Default" (0) selection.
    func resolvedMicDeviceID(for inputDeviceID: AudioDeviceID) -> AudioDeviceID? {
        if inputDeviceID > 0 {
            let availableDeviceIDs = Set(MicCapture.availableInputDevices().map(\.id))
            return availableDeviceIDs.contains(inputDeviceID) ? inputDeviceID : nil
        }
        return MicCapture.defaultInputDeviceID()
    }
}
