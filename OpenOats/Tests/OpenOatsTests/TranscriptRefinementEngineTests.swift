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
    var lastTemperature: Double?
    var shouldThrow = false
    var delay: Duration?

    func complete(
        apiKey: String?,
        model: String,
        messages: [OpenRouterClient.Message],
        maxTokens: Int,
        baseURL: URL?,
        temperature: Double?
    ) async throws -> String {
        if let delay {
            try? await Task.sleep(for: delay)
        }
        lastMessages = messages
        lastModel = model
        lastApiKey = apiKey
        lastBaseURL = baseURL
        lastTemperature = temperature
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
        let prompt = TranscriptRefinementEngine.buildSystemPrompt(languages: "English", customVocabulary: "")
        XCTAssertTrue(prompt.contains("professional transcript editor"))
        XCTAssertTrue(prompt.contains("speech disfluencies"))
        XCTAssertTrue(prompt.contains("The speakers use English."))
        XCTAssertFalse(prompt.contains("Known proper nouns"))
    }

    func testBuildSystemPromptWithVocabulary() async {
        let prompt = TranscriptRefinementEngine.buildSystemPrompt(languages: "English", customVocabulary: "Acme Corp\nJohn Doe: CEO")
        XCTAssertTrue(prompt.contains("Known proper nouns"))
        XCTAssertTrue(prompt.contains("Acme Corp"))
        XCTAssertTrue(prompt.contains("John Doe"))
    }

    func testBuildSystemPromptTrimsBlankVocabulary() async {
        let prompt = TranscriptRefinementEngine.buildSystemPrompt(languages: "English", customVocabulary: "   \n  \n  ")
        XCTAssertFalse(prompt.contains("Known proper nouns"))
    }

    func testBuildSystemPromptMultipleLanguages() async {
        let prompt = TranscriptRefinementEngine.buildSystemPrompt(languages: "English, Polish, German", customVocabulary: "")
        XCTAssertTrue(prompt.contains("English, Polish, German"))
        XCTAssertTrue(prompt.contains("switch between them"))
    }

    func testBuildSystemPromptSingleLanguage() async {
        let prompt = TranscriptRefinementEngine.buildSystemPrompt(languages: "Japanese", customVocabulary: "")
        XCTAssertTrue(prompt.contains("The speakers use Japanese."))
        XCTAssertFalse(prompt.contains("switch between"))
    }

    func testBuildSystemPromptEmptyLanguagesFallsBackToEnglish() async {
        let prompt = TranscriptRefinementEngine.buildSystemPrompt(languages: "", customVocabulary: "")
        XCTAssertTrue(prompt.contains("The speakers use English."))
    }

    func testBuildSystemPromptPreservationRules() async {
        let prompt = TranscriptRefinementEngine.buildSystemPrompt(languages: "English", customVocabulary: "")
        XCTAssertTrue(prompt.contains("Do NOT add, remove, or change any substantive content"))
        XCTAssertTrue(prompt.contains("Do NOT paraphrase or summarize"))
        XCTAssertTrue(prompt.contains("unclear or ambiguous, keep it as-is"))
        XCTAssertTrue(prompt.contains("Return the original text unchanged if no cleanup is needed"))
    }

    func testBuildSystemPromptBackchannelRule() async {
        let prompt = TranscriptRefinementEngine.buildSystemPrompt(languages: "English", customVocabulary: "")
        XCTAssertTrue(prompt.contains("uh-huh"))
        XCTAssertTrue(prompt.contains("backchannels"))
    }

    // MARK: - buildContextBlock

    func testBuildContextBlockEmpty() async {
        let result = TranscriptRefinementEngine.buildContextBlock([])
        XCTAssertNil(result)
    }

    func testBuildContextBlockFormatsUtterances() async {
        let context = [
            makeUtterance(text: "How is the project going?", speaker: .you),
            makeUtterance(text: "It's going well, we finished the sprint.", speaker: .them),
        ]
        let result = TranscriptRefinementEngine.buildContextBlock(context)!
        XCTAssertTrue(result.contains("Previous context (do not modify):"))
        XCTAssertTrue(result.contains("Speaker A: How is the project going?"))
        XCTAssertTrue(result.contains("Speaker B: It's going well"))
    }

    // MARK: - needsCleanup

    func testNeedsCleanupDetectsFillerWords() {
        XCTAssertTrue(TranscriptRefinementEngine.needsCleanup("Uh so we need to discuss this"))
        XCTAssertTrue(TranscriptRefinementEngine.needsCleanup("I think like we should do it"))
        XCTAssertTrue(TranscriptRefinementEngine.needsCleanup("Um let me think about that"))
        XCTAssertTrue(TranscriptRefinementEngine.needsCleanup("You know what I mean right"))
    }

    func testNeedsCleanupDetectsMissingPunctuation() {
        XCTAssertTrue(TranscriptRefinementEngine.needsCleanup("This sentence has no ending punctuation"))
        XCTAssertFalse(TranscriptRefinementEngine.needsCleanup("This sentence is properly punctuated."))
        XCTAssertFalse(TranscriptRefinementEngine.needsCleanup("Is this a question?"))
        XCTAssertFalse(TranscriptRefinementEngine.needsCleanup("This is exciting!"))
    }

    func testNeedsCleanupReturnsFalseForCleanText() {
        XCTAssertFalse(TranscriptRefinementEngine.needsCleanup("The quarterly results exceeded expectations."))
        XCTAssertFalse(TranscriptRefinementEngine.needsCleanup("We should schedule a follow-up meeting."))
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

    // MARK: - Detect-then-correct Skipping

    @MainActor
    func testCleanUtteranceIsSkipped() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let clean = makeUtterance(text: "The quarterly results exceeded our expectations by a wide margin.")
        store.append(clean)

        await engine.refine(clean)
        try await Task.sleep(for: .milliseconds(50))

        XCTAssertEqual(store.utterances.first?.refinementStatus, .skipped)
        let callCount = await mockClient.callCount
        XCTAssertEqual(callCount, 0, "Clean text should not be sent to the LLM")
    }

    @MainActor
    func testDirtyUtteranceIsNotSkipped() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["We need to discuss the project timeline."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let dirty = makeUtterance(text: "Uh so like we need to discuss the um project timeline")
        store.append(dirty)

        await engine.refine(dirty)
        await engine.drain(timeout: .seconds(2))
        try await Task.sleep(for: .milliseconds(50))

        let callCount = await mockClient.callCount
        XCTAssertEqual(callCount, 1, "Dirty text should be sent to the LLM")
    }

    @MainActor
    func testQuestionBypassesNeedsCleanupCheck() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["What do you think about this approach?"])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        // This looks clean but is a question — should still go through
        let question = makeUtterance(text: "What do you think about this particular approach?")
        store.append(question)

        await engine.refine(question)
        await engine.drain(timeout: .seconds(2))
        try await Task.sleep(for: .milliseconds(50))

        let callCount = await mockClient.callCount
        XCTAssertEqual(callCount, 1, "Questions should bypass the needsCleanup check")
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

    // MARK: - Temperature

    @MainActor
    func testRefinementUsesTemperatureZero() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["Cleaned output from the model."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "Uh so we need to um discuss this particular topic here")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))

        let temperature = await mockClient.lastTemperature
        XCTAssertEqual(temperature, 0, "Refinement should use temperature 0 for deterministic output")
    }

    // MARK: - Failure Handling

    @MainActor
    func testLLMErrorMarksAsFailed() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setShouldThrow(true)

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "Um this is a sentence that should like fail during refinement")
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

        let utterance = makeUtterance(text: "Uh this should like fail because the LLM returns empty whitespace")
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

        let utterance = makeUtterance(text: "Uh some long enough text to like pass the word count filter here")
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

        let utterance = makeUtterance(text: "Um a sentence that is like long enough to not be skipped")
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

        let utterance = makeUtterance(text: "Er a sentence that is like long enough to not be skipped")
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
        await mockClient.setDelay(.milliseconds(200))
        await mockClient.setResponses((0..<6).map { "Refined utterance number \($0) from LLM" })

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        for i in 0..<6 {
            let u = makeUtterance(text: "Um this is like utterance number \(i) with enough words")
            store.append(u)
            await engine.refine(u)
        }

        try await Task.sleep(for: .milliseconds(50))
        let earlyCount = await mockClient.callCount
        XCTAssertLessThanOrEqual(earlyCount, 3, "At most 3 concurrent tasks should start")

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

        await engine.drain(timeout: .seconds(1))
        // If we get here without hanging, the test passes
    }

    // MARK: - User Message Structure

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
        XCTAssertTrue(messages[0].content.contains("professional transcript editor"))
        XCTAssertEqual(messages[1].role, "user")
        XCTAssertTrue(messages[1].content.contains("Clean this utterance:"))
        XCTAssertTrue(messages[1].content.contains("Uh so like we need to discuss the um project timeline"))
    }

    @MainActor
    func testRefinementWithContextIncludesPrecedingUtterances() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["We need to discuss the project timeline."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let context = [
            makeUtterance(text: "Let's talk about the project.", speaker: .you),
            makeUtterance(text: "Sure, what aspect?", speaker: .them),
        ]
        let utterance = makeUtterance(text: "Uh so like we need to discuss the um project timeline")
        store.append(context[0])
        store.append(context[1])
        store.append(utterance)

        await engine.refine(utterance, context: context)
        await engine.drain(timeout: .seconds(2))

        let messages = await mockClient.lastMessages
        let userContent = messages[1].content
        XCTAssertTrue(userContent.contains("Previous context (do not modify):"))
        XCTAssertTrue(userContent.contains("Speaker A: Let's talk about the project."))
        XCTAssertTrue(userContent.contains("Speaker B: Sure, what aspect?"))
        XCTAssertTrue(userContent.contains("Clean this utterance:"))
    }

    @MainActor
    func testRefinementWithoutContextOmitsContextBlock() async throws {
        let settings = makeSettings()
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["Cleaned text from the LLM response output."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "Um a sentence with like enough words to not be filtered")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))

        let messages = await mockClient.lastMessages
        let userContent = messages[1].content
        XCTAssertFalse(userContent.contains("Previous context"))
        XCTAssertTrue(userContent.hasPrefix("Clean this utterance:"))
    }

    // MARK: - Language Settings

    @MainActor
    func testRefinementLanguagesAppearInSystemPrompt() async throws {
        let settings = makeSettings()
        settings.refinementLanguages = "Spanish, French"
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["Texto limpio de la salida del modelo LLM."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "Uh bueno entonces como que necesitamos discutir el proyecto")
        store.append(utterance)

        await engine.refine(utterance)
        await engine.drain(timeout: .seconds(2))

        let messages = await mockClient.lastMessages
        let systemPrompt = messages[0].content
        XCTAssertTrue(systemPrompt.contains("Spanish, French"))
        XCTAssertTrue(systemPrompt.contains("switch between them"))
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

        let utterance = makeUtterance(text: "So like we were talking about open oats and the acme corp deal")
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
        let store = TranscriptStore()
        let mockClient = MockLLMClient()
        await mockClient.setResponses(["Refined text output from the LLM model."])

        let engine = TranscriptRefinementEngine(settings: settings, transcriptStore: store, client: mockClient)

        let utterance = makeUtterance(text: "Uh a sentence with like enough words to not be filtered out")
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
