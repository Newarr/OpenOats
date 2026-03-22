import XCTest
@testable import OpenOatsKit

final class TranscriptCleanupEngineTests: XCTestCase {

    // MARK: - Helpers

    private func makeRecord(
        speaker: Speaker = .you,
        text: String = "Hello",
        timestamp: Date = Date(),
        refinedText: String? = nil
    ) -> SessionRecord {
        SessionRecord(
            speaker: speaker,
            text: text,
            timestamp: timestamp,
            refinedText: refinedText
        )
    }

    private func makeRecordsWithGap(
        count: Int,
        gapSeconds: TimeInterval
    ) -> [SessionRecord] {
        let start = Date(timeIntervalSince1970: 0)
        return (0..<count).map { i in
            makeRecord(
                text: "Line \(i)",
                timestamp: start.addingTimeInterval(Double(i) * gapSeconds)
            )
        }
    }

    // MARK: - chunkRecords Tests

    func testChunkRecordsEmpty() {
        let chunks = TranscriptCleanupEngine.chunkRecords([])
        XCTAssertTrue(chunks.isEmpty)
    }

    func testChunkRecordsSingleRecord() {
        let records = [makeRecord()]
        let chunks = TranscriptCleanupEngine.chunkRecords(records)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].count, 1)
    }

    func testChunkRecordsAllWithinOneWindow() {
        // 10 records, 10 seconds apart = 90 seconds total, under 150s threshold
        let records = makeRecordsWithGap(count: 10, gapSeconds: 10)
        let chunks = TranscriptCleanupEngine.chunkRecords(records)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].count, 10)
    }

    func testChunkRecordsSplitsAtTimeBoundary() {
        // 10 records, 60 seconds apart = 540 seconds total
        // Should split at ~150s boundary
        let records = makeRecordsWithGap(count: 10, gapSeconds: 60)
        let chunks = TranscriptCleanupEngine.chunkRecords(records)

        // With 60s gaps, first chunk gets records at 0, 60, 120 (3 records, next at 180 >= 150)
        XCTAssertGreaterThan(chunks.count, 1)

        // All records accounted for
        let totalRecords = chunks.reduce(0) { $0 + $1.count }
        XCTAssertEqual(totalRecords, 10)
    }

    func testChunkRecordsPreservesOrder() {
        let records = makeRecordsWithGap(count: 20, gapSeconds: 30)
        let chunks = TranscriptCleanupEngine.chunkRecords(records)

        let flattened = chunks.flatMap { $0 }
        XCTAssertEqual(flattened.count, 20)
        for (i, record) in flattened.enumerated() {
            XCTAssertEqual(record.text, "Line \(i)")
        }
    }

    // MARK: - parseResponse Tests

    func testParseResponseMatchingLineCount() {
        let records = [
            makeRecord(speaker: .you, text: "um hello world"),
            makeRecord(speaker: .them, text: "uh yeah hi"),
        ]

        let response = """
        [10:00:00] You: Hello world.
        [10:00:05] Them: Yeah, hi.
        """

        let result = TranscriptCleanupEngine.parseResponse(response, originalRecords: records)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.count, 2)
        XCTAssertEqual(result?[0].refinedText, "Hello world.")
        XCTAssertEqual(result?[1].refinedText, "Yeah, hi.")
    }

    func testParseResponseMismatchedLineCountReturnsNil() {
        let records = [
            makeRecord(text: "hello"),
            makeRecord(text: "world"),
        ]

        // Only one line in response (should be 2)
        let response = "[10:00:00] You: Hello world."
        let result = TranscriptCleanupEngine.parseResponse(response, originalRecords: records)
        XCTAssertNil(result)
    }

    func testParseResponseExtraLinesReturnsNil() {
        let records = [makeRecord(text: "hello")]

        let response = """
        [10:00:00] You: Hello.
        [10:00:05] Them: Extra line.
        """

        let result = TranscriptCleanupEngine.parseResponse(response, originalRecords: records)
        XCTAssertNil(result)
    }

    func testParseResponseMissingPrefixStillWorks() {
        let records = [makeRecord(text: "um hello")]

        // Response without the [HH:MM:SS] Speaker: prefix
        let response = "Hello."
        let result = TranscriptCleanupEngine.parseResponse(response, originalRecords: records)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?[0].refinedText, "Hello.")
    }

    func testParseResponseEmptyLineUsesOriginalText() {
        let records = [makeRecord(text: "hello")]

        // Response with empty content after prefix
        let response = "[10:00:00] You: "
        let result = TranscriptCleanupEngine.parseResponse(response, originalRecords: records)
        XCTAssertNotNil(result)
        // Empty cleaned text falls back to original
        XCTAssertEqual(result?[0].refinedText, "hello")
    }

    func testParseResponsePreservesOriginalFields() {
        let record = SessionRecord(
            speaker: .them,
            text: "um yeah",
            timestamp: Date(timeIntervalSince1970: 1000),
            suggestions: ["suggestion1"],
            kbHits: ["hit1"]
        )

        let response = "[10:00:00] Them: Yeah."
        let result = TranscriptCleanupEngine.parseResponse(response, originalRecords: [record])
        XCTAssertNotNil(result)
        XCTAssertEqual(result?[0].speaker, .them)
        XCTAssertEqual(result?[0].text, "um yeah") // original text preserved
        XCTAssertEqual(result?[0].refinedText, "Yeah.")
        XCTAssertEqual(result?[0].suggestions, ["suggestion1"])
        XCTAssertEqual(result?[0].kbHits, ["hit1"])
    }

    func testParseResponseEmptyResponseReturnsNil() {
        let records = [makeRecord(text: "hello")]
        let result = TranscriptCleanupEngine.parseResponse("", originalRecords: records)
        XCTAssertNil(result, "Empty response should return nil (0 lines != 1 record)")
    }

    // MARK: - Cancellation

    func testCancelReturnOriginals() async {
        let engine = await TranscriptCleanupEngine()
        // Cancel immediately - no cleanup in flight, should be a no-op
        await engine.cancel()
        let isCleaning = await engine.isCleaningUp
        XCTAssertFalse(isCleaning)
    }
}
