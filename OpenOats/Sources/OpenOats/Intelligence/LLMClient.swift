import Foundation

/// A chat message for LLM completions, decoupled from any specific provider.
struct ChatMessage: Codable, Sendable {
    let role: String
    let content: String
}

/// Protocol for non-streaming LLM completions, enabling test injection.
protocol LLMCompleting: Sendable {
    func complete(
        apiKey: String?,
        model: String,
        messages: [ChatMessage],
        maxTokens: Int,
        baseURL: URL?,
        temperature: Double?
    ) async throws -> String
}
