import Foundation

/// Refines utterances by cleaning up filler words and fixing punctuation via LLM.
/// Runs as a background actor with bounded concurrency.
actor TranscriptRefinementEngine {
    private let client: any LLMCompleting
    private let settings: AppSettings
    private let transcriptStore: TranscriptStore

    private let maxConcurrent = 3
    private let maxPendingQueueSize = 50
    private var inFlightCount = 0
    private var pendingQueue: [(utterance: Utterance, context: [Utterance])] = []
    private var activeTasks: [UUID: Task<Void, Never>] = [:]

    /// Hardcoded cheap model for refinement (keeps cost low).
    private let refinementModel = "openai/gpt-4o-mini"
    private let minimumWordCount = 5

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
            - Keep verbal backchannels (e.g. "uh-huh", "yeah", "mm-hmm") when they serve as \
            a direct response or acknowledgment.
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

    /// Quick heuristic: does the text contain likely disfluencies worth cleaning?
    static func needsCleanup(_ text: String) -> Bool {
        let lower = text.lowercased()
        // Common disfluency markers across languages
        let markers = [
            // Filler sounds
            " uh ", " um ", " ah ", " er ",
            // English fillers
            " like ", " you know ", " i mean ", " basically ", " actually ",
            " literally ", " right ", " so ", " well ",
            // Repetition patterns (word repeated)
            "...", " -- ",
            // Punctuation issues (run-on without punctuation)
        ]
        for marker in markers where lower.contains(marker) {
            return true
        }
        // Check for text starting/ending with fillers
        let prefixes = ["uh ", "um ", "ah ", "er ", "so ", "well ", "like "]
        for prefix in prefixes where lower.hasPrefix(prefix) {
            return true
        }
        // Check for missing sentence-final punctuation (likely needs cleanup)
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty && !trimmed.hasSuffix(".") && !trimmed.hasSuffix("!") &&
           !trimmed.hasSuffix("?") && !trimmed.hasSuffix(",") && !trimmed.hasSuffix("\"") {
            return true
        }
        return false
    }

    init(settings: AppSettings, transcriptStore: TranscriptStore, client: any LLMCompleting = OpenRouterClient()) {
        self.settings = settings
        self.transcriptStore = transcriptStore
        self.client = client
    }

    /// Queue an utterance for refinement, capturing preceding context.
    func refine(_ utterance: Utterance, context: [Utterance] = []) async {
        // Skip short utterances unless they look like a question
        let words = utterance.text.split(separator: " ")
        let isQuestion = utterance.text.contains("?")
        if words.count < minimumWordCount && !isQuestion {
            await updateStore(id: utterance.id, refinedText: nil, status: .skipped)
            return
        }

        // Detect-then-correct: skip utterances that look clean already.
        // Always send questions through (even if they look clean, short questions
        // passed the word-count gate specifically because they're questions).
        if !isQuestion && !Self.needsCleanup(utterance.text) {
            await updateStore(id: utterance.id, refinedText: nil, status: .skipped)
            return
        }

        // Cap queue depth to prevent unbounded memory growth during prolonged LLM unavailability.
        if pendingQueue.count >= maxPendingQueueSize {
            let evicted = pendingQueue.removeFirst()
            await updateStore(id: evicted.utterance.id, refinedText: nil, status: .skipped)
        }

        pendingQueue.append((utterance: utterance, context: context))
        dispatchPending()
    }

    /// Await all pending and in-flight refinements, with a timeout.
    func drain(timeout: Duration = .seconds(5)) async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                // Re-check after each round in case didCompleteTask spawned new work.
                while await self.hasPendingWork {
                    let tasks = await self.snapshotActiveTasks()
                    for task in tasks {
                        await task.value
                    }
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

    private var hasPendingWork: Bool {
        inFlightCount > 0 || !pendingQueue.isEmpty
    }

    private func snapshotActiveTasks() -> [Task<Void, Never>] {
        Array(activeTasks.values)
    }

    // MARK: - Private

    /// Centralized store update to avoid scattered MainActor hops.
    private func updateStore(id: UUID, refinedText: String?, status: RefinementStatus) async {
        let store = transcriptStore
        await MainActor.run {
            store.updateRefinedText(id: id, refinedText: refinedText, status: status)
        }
    }

    private func dispatchPending() {
        while inFlightCount < maxConcurrent, let item = pendingQueue.first {
            pendingQueue.removeFirst()
            inFlightCount += 1

            let id = item.utterance.id

            // Mark as pending — awaited to ensure status lands before drain() returns.
            let store = transcriptStore
            let pendingTask = Task { @MainActor in
                store.updateRefinedText(id: id, refinedText: nil, status: .pending)
            }

            let task = Task {
                await pendingTask.value
                await self.performRefinement(item.utterance, context: item.context)
                await self.didCompleteTask(id: id)
            }
            activeTasks[id] = task
        }
    }

    private func didCompleteTask(id: UUID) {
        activeTasks.removeValue(forKey: id)
        inFlightCount -= 1
        dispatchPending()
    }

    private func performRefinement(_ utterance: Utterance, context: [Utterance]) async {
        let apiKey: String?
        let baseURL: URL?
        let model: String

        // Read all settings atomically in a single MainActor hop
        let config = await MainActor.run {
            (
                provider: settings.llmProvider,
                openRouterKey: settings.openRouterApiKey,
                ollamaURL: settings.ollamaBaseURL,
                ollamaModel: settings.ollamaLLMModel,
                mlxURL: settings.mlxBaseURL,
                mlxModelName: settings.mlxModel,
                customVocab: settings.transcriptionCustomVocabulary,
                languages: settings.refinementLanguages
            )
        }
        let systemPromptText = Self.buildSystemPrompt(languages: config.languages, customVocabulary: config.customVocab)

        switch config.provider {
        case .openRouter:
            apiKey = config.openRouterKey.isEmpty ? nil : config.openRouterKey
            baseURL = nil
            model = refinementModel
        case .ollama:
            apiKey = nil
            let base = config.ollamaURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            guard let url = URL(string: base + "/v1/chat/completions") else {
                diagLog("[REFINE] invalid Ollama URL: \(config.ollamaURL)")
                await updateStore(id: utterance.id, refinedText: nil, status: .failed)
                return
            }
            baseURL = url
            model = config.ollamaModel
        case .mlx:
            apiKey = nil
            let base = config.mlxURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            guard let url = URL(string: base + "/v1/chat/completions") else {
                diagLog("[REFINE] invalid MLX URL: \(config.mlxURL)")
                await updateStore(id: utterance.id, refinedText: nil, status: .failed)
                return
            }
            baseURL = url
            model = config.mlxModelName
        }

        var userContent = ""
        if let contextBlock = Self.buildContextBlock(context) {
            userContent += contextBlock + "\n\n"
        }
        userContent += "Clean this utterance:\n" + utterance.text

        let messages: [ChatMessage] = [
            .init(role: "system", content: systemPromptText),
            .init(role: "user", content: userContent)
        ]

        do {
            let refined = try await client.complete(
                apiKey: apiKey,
                model: model,
                messages: messages,
                maxTokens: 512,
                baseURL: baseURL,
                temperature: 0
            )

            let trimmed = refined.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                diagLog("[REFINE] empty response for utterance \(utterance.id)")
                await updateStore(id: utterance.id, refinedText: nil, status: .failed)
                return
            }

            await updateStore(id: utterance.id, refinedText: trimmed, status: .completed)
        } catch {
            diagLog("[REFINE] error for utterance \(utterance.id): \(error.localizedDescription)")
            await updateStore(id: utterance.id, refinedText: nil, status: .failed)
        }
    }
}
