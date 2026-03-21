import XCTest
@testable import OpenOatsKit

// MARK: - Mock LLM Client

private actor MockLLMClient: LLMCompleting {
    var responses: [String] = []
    var callCount = 0
    var lastMessages: [OpenRouterClient.Message] = []
    var lastModel: String?
    var lastApiKey: String?
    var lastBaseURL: URL?
    var shouldThrow = false
    var delay: Duration?

    func complete(
        apiKey: String?,
        model: String,
        messages: [OpenRouterClient.Message],
        maxTokens: Int,
        baseURL: URL?
    ) async throws -> String {
        if let delay {
            try? await Task.sleep(for: delay)
        }
        lastMessages = messages
        lastModel = model
        lastApiKey = apiKey
        lastBaseURL = baseURL
        let index = callCount
        callCount += 1
        if shouldThrow {
            throw NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "mock error"])
        }
        guard index < responses.count else { return responses.last ?? "" }
        return responses[index]
    }
}

// MARK: - Helpers

@MainActor
private func makeSettings() -> AppSettings {
    let defaults = UserDefaults(suiteName: "TranscriptRefinementEngineTests-\(UUID().uuidString)")!
    defaults.removePersistentDomain(forName: defaults.suiteName!)
    let storage = AppSettingsStorage(
        defaults: defaults,
        secretStore: .ephemeral,
        defaultNotesDirectory: FileManager.default.temporaryDirectory,
        runMigrations: false
    )
    return AppSettings(storage: storage)
}

private func makeUtterance(text: String, speaker: Speaker = .them) -> Utterance {
    Utterance(text: text, speaker: speaker)
}

// MARK: - Tests

final class TranscriptRefinementEngineTests: XCTestCase {

    // MARK: - buildSystemPrompt

    func testBuildSystemPromptWithoutVocabulary() async {
        let prompt = TranscriptRefinementEngine.buildSystemPrompt(customVocabulary: "")
        XCTAssertTrue(prompt.contains("Clean up this speech transcript"))
        XCTAssertTrue(prompt.contains("filler words"))
        XCTAssertFalse(prompt.contains("Known proper nouns"))
    }

    func testBuildSystemPromptWithVocabulary() async {
        let prompt = TranscriptRefinementEngine.buildSystemPrompt(customVocabulary: "Acme Corp\nJohn Doe: CEO")
        XCTAssertTrue(prompt.contains("Known proper nouns"))
        XCTAssertTrue(prompt.contains("Acme Corp"))
        XCTAssertTrue(prompt.contains("John Doe"))
    }

    func testBuildSystemPromptTrimsBlankVocabulary() async {
        let prompt = TranscriptRefinementEngine.buildSystemPrompt(customVocabulary: "   \n  \n  ")
        XCTAssertFalse(prompt.contains("Known proper nouns"))
    }

    // MARK: - Short Utterance Skipping

    @MainActor
    func testShortUtteranceIsSkipped() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let short = makeUtterance(text: "Yes okay")
        store.append(short)

        await engine.refine(short)
        // Give the MainActor task time to execute
        try await Task.sleep(for: .milliseconds(50))

        XCTAssertEqual(store.utterances.first?.refinementStatus, .skipped)
        let callCount = await mockClient.callCount
        XCTAssertEqual(callCount, 0, "LLM should not be called for short utterances")
    }

    @MainActor
    func testShortQuestionIsNotSkipped() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["What?"])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let question = makeUtterance(text: "What?")
        store.append(question)

        await engine.refine(question)
        await engine.drain(timeout: .seconds(2))
        try await Task.sleep(for: .milliseconds(50))

        let callCount = await mockClient.callCount
        XCTAssertEqual(callCount, 1, "Questions should be refined even if short")
    }

    @MainActor
    func testExactlyFiveWordsIsNotSkipped() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["This has five words exactly."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "This has five words exactly")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))
        try await Task.sleep(for: .milliseconds(50))

        let callCount = await mockClient.callCount
        XCTAssertEqual(callCount, 1)
    }

    // MARK: - Successful Refinement

    @MainActor
    func testSuccessfulRefinementUpdatesStore() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["This is the cleaned up version of the text."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "Uh this is like the um cleaned up version of the text")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))
        try await Task.sleep(for: .milliseconds(100))

        XCTAssertEqual(store.utterances.first?.refinedText, "This is the cleaned up version of the text.")
        XCTAssertEqual(store.utterances.first?.refinementStatus, .completed)
    }

    // MARK: - Failure Handling

    @MainActor
    func testLLMErrorMarksAsFailed() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setShouldThrow(true)

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "This is a sentence that should fail during refinement")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))
        try await Task.sleep(for: .milliseconds(100))

        XCTAssertEqual(store.utterances.first?.refinementStatus, .failed)
        XCTAssertNil(store.utterances.first?.refinedText)
    }

    @MainActor
    func testEmptyResponseMarksAsFailed() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["   \n  "])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "This should fail because the LLM returns empty whitespace")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))
        try await Task.sleep(for: .milliseconds(100))

        XCTAssertEqual(store.utterances.first?.refinementStatus, .failed)
    }

    // MARK: - Provider Routing

    @MainActor
    func testOpenRouterProviderUsesCorrectModel() async throws {
        let settings = makeSettings()
        settings.llmProvider = .openRouter
        settings.openRouterApiKey = "test-key-123"
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["Refined text from the model output."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "Some long enough text to pass the word count filter here")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))

        let model = await mockClient.lastModel
        let apiKey = await mockClient.lastApiKey
        let baseURL = await mockClient.lastBaseURL
        XCTAssertEqual(model, "openai/gpt-4o-mini")
        XCTAssertEqual(apiKey, "test-key-123")
        XCTAssertNil(baseURL, "OpenRouter should use default base URL")
    }

    @MainActor
    func testOllamaProviderUsesCustomBaseURL() async throws {
        let settings = makeSettings()
        settings.llmProvider = .ollama
        settings.ollamaBaseURL = "http://localhost:11434"
        settings.ollamaLLMModel = "llama3"
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["Refined via ollama model output."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "A sentence that is long enough to not be skipped by filter")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))

        let model = await mockClient.lastModel
        let apiKey = await mockClient.lastApiKey
        let baseURL = await mockClient.lastBaseURL
        XCTAssertEqual(model, "llama3")
        XCTAssertNil(apiKey)
        XCTAssertEqual(baseURL?.absoluteString, "http://localhost:11434/v1/chat/completions")
    }

    @MainActor
    func testMLXProviderUsesCustomBaseURL() async throws {
        let settings = makeSettings()
        settings.llmProvider = .mlx
        settings.mlxBaseURL = "http://localhost:8080"
        settings.mlxModel = "my-mlx-model"
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["Refined via MLX model output here."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "A sentence that is long enough to not be skipped here")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))

        let model = await mockClient.lastModel
        let baseURL = await mockClient.lastBaseURL
        XCTAssertEqual(model, "my-mlx-model")
        XCTAssertEqual(baseURL?.absoluteString, "http://localhost:8080/v1/chat/completions")
    }

    // MARK: - Concurrency Bounds

    @MainActor
    func testConcurrencyIsBoundedToThree() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        // Slow responses to keep tasks in-flight
        await mockClient.setDelay(.milliseconds(200))
        await mockClient.setResponses((0..<6).map { "Refined utterance number \($0) from LLM" })

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        // Queue 6 utterances
        for i in 0..<6 {
            let u = makeUtterance(text: "This is utterance number \(i) with enough words to pass")
            store.append(u)
            await engine.refine(u)
        }

        // After a short wait, at most 3 should have started
        try await Task.sleep(for: .milliseconds(50))
        let earlyCount = await mockClient.callCount
        XCTAssertLessThanOrEqual(earlyCount, 3, "At most 3 concurrent tasks should start")

        // Wait for all to finish
        await engine.drain(timeout: .seconds(5))
        try await Task.sleep(for: .milliseconds(100))

        let finalCount = await mockClient.callCount
        XCTAssertEqual(finalCount, 6, "All 6 utterances should eventually be refined")
    }

    // MARK: - Drain

    @MainActor
    func testDrainReturnsImmediatelyWhenIdle() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        // Should return immediately with no work
        await engine.drain(timeout: .seconds(1))
        // If we get here without hanging, the test passes
    }

    // MARK: - System Prompt Includes User Text

    @MainActor
    func testRefinementSendsSystemAndUserMessages() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["Cleaned up text from the LLM response."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "Uh so like we need to discuss the um project timeline")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))

        let messages = await mockClient.lastMessages
        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[0].role, "system")
        XCTAssertTrue(messages[0].content.contains("Clean up this speech transcript"))
        XCTAssertEqual(messages[1].role, "user")
        XCTAssertEqual(messages[1].content, "Uh so like we need to discuss the um project timeline")
    }

    // MARK: - Custom Vocabulary in Prompt

    @MainActor
    func testCustomVocabularyIsIncludedInSystemPrompt() async throws {
        let settings = makeSettings()
        settings.transcriptionCustomVocabulary = "OpenOats\nAcme Corp: company"
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["Cleaned text with OpenOats mentioned in it properly."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "We were talking about open oats and the acme corp deal")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))

        let messages = await mockClient.lastMessages
        let systemPrompt = messages.first?.content ?? ""
        XCTAssertTrue(systemPrompt.contains("OpenOats"))
        XCTAssertTrue(systemPrompt.contains("Acme Corp"))
    }

    // MARK: - Empty OpenRouter API Key

    @MainActor
    func testEmptyOpenRouterKeyPassesNilApiKey() async throws {
        let settings = makeSettings()
        settings.llmProvider = .openRouter
        // Key defaults to "" from ephemeral secret store
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["Refined text output from the LLM model."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "A sentence with enough words to not be filtered out")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))

        let apiKey = await mockClient.lastApiKey
        XCTAssertNil(apiKey, "Empty API key string should become nil")
    }
}

// MARK: - MockLLMClient setters (actor-isolated)

private extension MockLLMClient {
    func setResponses(_ r: [String]) { responses = r }
    func setShouldThrow(_ v: Bool) { shouldThrow = v }
    func setDelay(_ d: Duration) { delay = d }
}
