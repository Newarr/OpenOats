import Foundation

/// Refines utterances by cleaning up filler words and fixing punctuation via LLM.
/// Runs as a background actor with bounded concurrency.
actor TranscriptRefinementEngine {
    private let client: any LLMCompleting
    private let settings: AppSettings
    private let transcriptStore: TranscriptStore

    private let maxConcurrent = 3
    private var inFlightCount = 0
    private var pendingQueue: [(utterance: Utterance, context: [Utterance])] = []
    private var activeTasks: [UUID: Task<Void, Never>] = []

    /// Hardcoded cheap model for refinement (keeps cost low).
    private let refinementModel = "openai/gpt-4o-mini"
    private let minimumWordCount = 5
    private let contextWindowSize = 3

    static func buildSystemPrompt(languages: String, customVocabulary: String) -> String {
        let langList = languages
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        let languageClause: String
        if langList.count > 1 {
            languageClause = """
                The speakers use \(langList.joined(separator: ", ")) and may switch \
                between them freely, especially for technical or business terminology.
                """
        } else if let single = langList.first, !single.isEmpty {
            languageClause = "The speakers use \(single)."
        } else {
            languageClause = "The speakers use English."
        }

        var prompt = """
            You are a professional transcript editor. Clean up the following speech \
            transcript segment while strictly preserving the speaker's original meaning, \
            language, and intent.

            \(languageClause)

            Guidelines:
            - Remove speech disfluencies: filler words, false starts, and unnecessary repetitions.
            - Fix punctuation and capitalization.
            - Do NOT add, remove, or change any substantive content.
            - Do NOT paraphrase or summarize — keep the speaker's own words and phrasing.
            - Preserve technical terms, proper nouns, and numbers exactly as spoken.
            - If the speaker switches languages, preserve their language choice per phrase.
            - If a passage is unclear or ambiguous, keep it as-is rather than guessing.
            - Return the original text unchanged if no cleanup is needed.
            - Output only the cleaned text, nothing else.
            """

        let vocab = customVocabulary.trimmingCharacters(in: .whitespacesAndNewlines)
        if !vocab.isEmpty {
            let terms = vocab.split(separator: "\n")
                .compactMap { line -> String? in
                    let t = line.split(separator: ":").first.map { String($0).trimmingCharacters(in: .whitespaces) }
                        ?? String(line).trimmingCharacters(in: .whitespaces)
                    return t.isEmpty ? nil : t
                }
                .joined(separator: ", ")
            if !terms.isEmpty {
                prompt += "\n- Known proper nouns (use exact spelling): \(terms)"
            }
        }

        return prompt
    }

    /// Format preceding utterances as read-only context for the LLM.
    static func buildContextBlock(_ context: [Utterance]) -> String? {
        guard !context.isEmpty else { return nil }
        let lines = context.map { u in
            let speaker = u.speaker == .you ? "Speaker A" : "Speaker B"
            return "\(speaker): \(u.displayText)"
        }
        return "Previous context (do not modify):\n" + lines.joined(separator: "\n")
    }

    init(settings: AppSettings, transcriptStore: TranscriptStore, client: any LLMCompleting = OpenRouterClient()) {
        self.settings = settings
        self.transcriptStore = transcriptStore
        self.client = client
    }

    /// Queue an utterance for refinement, capturing preceding context.
    func refine(_ utterance: Utterance, context: [Utterance] = []) {
        // Skip short utterances unless they look like a question
        let words = utterance.text.split(separator: " ")
        if words.count < minimumWordCount && !utterance.text.contains("?") {
            Task { @MainActor in
                transcriptStore.updateRefinedText(id: utterance.id, refinedText: nil, status: .skipped)
            }
            return
        }

        pendingQueue.append((utterance: utterance, context: context))
        drainQueue()
    }

    /// Await all pending and in-flight refinements, with a timeout.
    func drain(timeout: Duration = .seconds(5)) async {
        guard inFlightCount > 0 || !pendingQueue.isEmpty else { return }

        let tasks = activeTasks.values.map { $0 }
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                for task in tasks {
                    await task.value
                }
            }
            group.addTask {
                try? await Task.sleep(for: timeout)
            }
            // Return as soon as either completes
            await group.next()
            group.cancelAll()
        }
    }

    // MARK: - Private

    private func drainQueue() {
        while inFlightCount < maxConcurrent, let item = pendingQueue.first {
            pendingQueue.removeFirst()
            inFlightCount += 1

            // Mark as pending on main actor
            let store = transcriptStore
            let id = item.utterance.id
            Task { @MainActor in
                store.updateRefinedText(id: id, refinedText: nil, status: .pending)
            }

            let task = Task { [weak self] in
                guard let self else { return }
                await self.performRefinement(item.utterance, context: item.context)
                await self.taskCompleted(id: id)
            }
            activeTasks[id] = task
        }
    }

    private func taskCompleted(id: UUID) {
        activeTasks.removeValue(forKey: id)
        inFlightCount -= 1
        drainQueue()
    }

    private func performRefinement(_ utterance: Utterance, context: [Utterance]) async {
        let apiKey: String?
        let baseURL: URL?
        let model: String

        // Read settings on MainActor
        let provider = await MainActor.run { settings.llmProvider }
        let openRouterKey = await MainActor.run { settings.openRouterApiKey }
        let ollamaURL = await MainActor.run { settings.ollamaBaseURL }
        let ollamaModel = await MainActor.run { settings.ollamaLLMModel }
        let mlxURL = await MainActor.run { settings.mlxBaseURL }
        let mlxModelName = await MainActor.run { settings.mlxModel }
        let customVocab = await MainActor.run { settings.transcriptionCustomVocabulary }
        let languages = await MainActor.run { settings.refinementLanguages }
        let systemPromptText = Self.buildSystemPrompt(languages: languages, customVocabulary: customVocab)

        switch provider {
        case .openRouter:
            apiKey = openRouterKey.isEmpty ? nil : openRouterKey
            baseURL = nil
            model = refinementModel
        case .ollama:
            apiKey = nil
            let base = ollamaURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            guard let url = URL(string: base + "/v1/chat/completions") else {
                await markFailed(utterance.id)
                return
            }
            baseURL = url
            model = ollamaModel
        case .mlx:
            apiKey = nil
            let base = mlxURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            guard let url = URL(string: base + "/v1/chat/completions") else {
                await markFailed(utterance.id)
                return
            }
            baseURL = url
            model = mlxModelName
        }

        var userContent = ""
        if let contextBlock = Self.buildContextBlock(context) {
            userContent += contextBlock + "\n\n"
        }
        userContent += "Clean this utterance:\n" + utterance.text

        let messages: [OpenRouterClient.Message] = [
            .init(role: "system", content: systemPromptText),
            .init(role: "user", content: userContent)
        ]

        do {
            let refined = try await client.complete(
                apiKey: apiKey,
                model: model,
                messages: messages,
                maxTokens: 512,
                baseURL: baseURL
            )

            let trimmed = refined.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                await markFailed(utterance.id)
                return
            }

            let store = transcriptStore
            Task { @MainActor in
                store.updateRefinedText(id: utterance.id, refinedText: trimmed, status: .completed)
            }
        } catch {
            await markFailed(utterance.id)
        }
    }

    private func markFailed(_ id: UUID) async {
        let store = transcriptStore
        Task { @MainActor in
            store.updateRefinedText(id: id, refinedText: nil, status: .failed)
        }
    }
}
