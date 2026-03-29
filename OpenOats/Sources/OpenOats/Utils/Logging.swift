import Foundation
import os

/// Centralized logger factory. One subsystem, per-component categories.
///
/// Usage: `Log.mic.debug("buffer received")`
///
/// Filter in Terminal:
///   log stream --predicate 'subsystem == "com.openoats.app"' --level debug
///   log stream --predicate 'subsystem == "com.openoats.app" AND category == "MicCapture"'
enum Log {
    static let mic = Logger(subsystem: subsystem, category: "MicCapture")
    static let recorder = Logger(subsystem: subsystem, category: "AudioRecorder")
    static let transcription = Logger(subsystem: subsystem, category: "TranscriptionEngine")
    static let streaming = Logger(subsystem: subsystem, category: "StreamingTranscriber")
    static let transcript = Logger(subsystem: subsystem, category: "TranscriptStore")
    static let echo = Logger(subsystem: subsystem, category: "AcousticEchoFilter")
    static let whisperkit = Logger(subsystem: subsystem, category: "WhisperKitManager")

    private static let subsystem = Bundle.main.bundleIdentifier ?? "com.openoats.app"
}
