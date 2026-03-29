import Foundation
import os

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
