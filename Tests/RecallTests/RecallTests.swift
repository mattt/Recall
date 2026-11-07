import Foundation
import Testing

@testable import Recall

@Suite struct FusionTests {
  @Test func scoresAreWeightedReciprocalRanks() {
    let scores = Fusion.scores(
      [
        (weight: 1, ids: ["a", "b", "c"]),
        (weight: 2, ids: ["c", "a"]),
      ],
      k: 60
    )
    #expect(scores["a"] == 1.0 / 61 + 2.0 / 62)
    #expect(scores["b"] == 1.0 / 62)
    #expect(scores["c"] == 1.0 / 63 + 2.0 / 61)
  }

  @Test func zeroWeightContributesNothing() {
    let scores = Fusion.scores([(weight: 0, ids: ["a"]), (weight: 1, ids: ["b"])], k: 0)
    #expect(scores["a"] == 0)
    #expect(scores["b"] == 1)
  }

  @Test func ranksStartAtOneAndKeepFirstPosition() {
    #expect(Fusion.ranks(["a", "b", "a"]) == ["a": 1, "b": 2])
  }
}

@Suite struct RecallTests {
  static let vectors: [String: [Float]] = [
    "The cat sleeps on the sofa.": [1, 0, 0],
    "A kitten chased the yarn.": [0.8, 0.6, 0],
    "Quarterly taxes are due in April.": [0, 0, 1],
    "feline": [1, 0.1, 0],
    "cat": [0.9, 0.3, 0.1],
  ]

  func makeStore() async throws -> (MemoryStore, [String: Memory]) {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder(vectors: Self.vectors))
    var memories: [String: Memory] = [:]
    for (offset, text) in [
      "The cat sleeps on the sofa.",
      "A kitten chased the yarn.",
      "Quarterly taxes are due in April.",
    ].enumerated() {
      memories[text] = try await store.remember(
        text,
        createdAt: Date(timeIntervalSinceReferenceDate: Double(offset))
      )
    }
    return (store, memories)
  }

  @Test func resultsReportEachRankAndTheFusedScore() async throws {
    let (store, memories) = try await makeStore()
    let cat = try #require(memories["The cat sleeps on the sofa."])
    let kitten = try #require(memories["A kitten chased the yarn."])
    let taxes = try #require(memories["Quarterly taxes are due in April."])

    let results = try await store.recall("cat")
    #expect(results.map(\.id) == [cat.id, kitten.id, taxes.id])

    // "cat" matches one memory by keyword.
    // By vector, the order is cat, kitten, taxes.
    // By recency among those three, the order is taxes, kitten, cat.
    let first = results[0]
    #expect(first.keywordRank == 1)
    #expect(first.vectorRank == 1)
    #expect(first.recencyRank == 3)
    #expect(first.score == 1.0 / 61 + 1.0 / 61 + 0.5 / 63)

    let second = results[1]
    #expect(second.keywordRank == nil)
    #expect(second.vectorRank == 2)
    #expect(second.recencyRank == 2)
    #expect(second.score == 1.0 / 62 + 0.5 / 62)

    let third = results[2]
    #expect(third.vectorRank == 3)
    #expect(third.recencyRank == 1)
    #expect(third.score == 1.0 / 63 + 0.5 / 61)
    #expect(try #require(third.similarity) < 0.2)
  }

  @Test func weightsAndKChangeTheArithmetic() async throws {
    let (store, memories) = try await makeStore()
    let cat = try #require(memories["The cat sleeps on the sofa."])
    let configuration = RecallConfiguration(
      k: 10,
      keywordWeight: 3,
      vectorWeight: 0.5,
      recencyWeight: 0
    )
    let results = try await store.recall("cat", configuration: configuration)
    let first = try #require(results.first)
    #expect(first.id == cat.id)
    #expect(first.score == 3.0 / 11 + 0.5 / 11)
    #expect(results.allSatisfy { $0.recencyRank == nil })
  }

  @Test func zeroVectorWeightSkipsTheEmbedder() async throws {
    let embedder = TestEmbedder(vectors: Self.vectors)
    let store = try MemoryStore.inMemory(embedder: embedder)
    try await store.remember("The cat sleeps on the sofa.")
    let results = try await store.recall(
      "cat",
      configuration: RecallConfiguration(vectorWeight: 0)
    )
    #expect(results.count == 1)
    #expect(results[0].vectorRank == nil)
    #expect(embedder.calls.all == ["The cat sleeps on the sofa."])
  }

  @Test func recencyWeightCanReverseKeywordOrder() async throws {
    let store = try MemoryStore.inMemory()
    let older = try await store.remember(
      "tea tea tea",
      createdAt: Date(timeIntervalSinceReferenceDate: 0)
    )
    let newer = try await store.remember(
      "tea and a long sentence about other things",
      createdAt: Date(timeIntervalSinceReferenceDate: 100)
    )

    // Older: keyword rank 1, recency rank 2. Newer: keyword rank 2, recency rank 1.
    // Newer wins when 1/62 + w/61 > 1/61 + w/62, which holds when w > 1.
    let none = try await store.recall("tea", configuration: RecallConfiguration(recencyWeight: 0))
    #expect(none.map(\.id) == [older.id, newer.id])
    #expect(none[0].score == 1.0 / 61)

    let half = try await store.recall("tea", configuration: RecallConfiguration(recencyWeight: 0.5))
    #expect(half.map(\.id) == [older.id, newer.id])
    #expect(half[0].score == 1.0 / 61 + 0.5 / 62)
    #expect(half[1].score == 1.0 / 62 + 0.5 / 61)

    let double = try await store.recall("tea", configuration: RecallConfiguration(recencyWeight: 2))
    #expect(double.map(\.id) == [newer.id, older.id])
    #expect(double[0].recencyRank == 1)
  }

  @Test func recencyDoesNotAddUnmatchedMemories() async throws {
    let store = try MemoryStore.inMemory()
    let match = try await store.remember("tea", createdAt: Date(timeIntervalSinceReferenceDate: 0))
    try await store.remember("coffee", createdAt: Date(timeIntervalSinceReferenceDate: 100))
    let results = try await store.recall(
      "tea",
      configuration: RecallConfiguration(recencyWeight: 10)
    )
    #expect(results.map(\.id) == [match.id])
  }

  @Test func equalScoresPutNewerFirst() async throws {
    let store = try MemoryStore.inMemory()
    let older = try await store.remember("tea", createdAt: Date(timeIntervalSinceReferenceDate: 0))
    let newer = try await store.remember("tea", createdAt: Date(timeIntervalSinceReferenceDate: 1))
    let results = try await store.recall(
      "tea",
      configuration: RecallConfiguration(keywordWeight: 0, recencyWeight: 0)
    )
    #expect(results.isEmpty)
    let tied = try await store.recall(
      "tea",
      configuration: RecallConfiguration(recencyWeight: 0)
    )
    // Identical text has identical BM25 scores,
    // and the keyword query orders ties newest first.
    #expect(tied.map(\.id) == [newer.id, older.id])
  }

  @Test func rankingsDescribeTheReturnedText() async throws {
    let gate = Gate()
    let store = try MemoryStore.inMemory(embedder: GatedEmbedder(gates: ["cat": gate]))
    let memory = try await store.remember("The cat sleeps on the sofa.")
    let recall = Task { try await store.recall("cat") }
    await gate.arrival()

    // Change the matching text while the query's vector is being made.
    try await store.update(memory.id, text: "The dog barks at the door.")
    gate.open()
    let results = try await recall.value
    let result = try #require(results.first { $0.id == memory.id })
    #expect(result.memory.text == "The dog barks at the door.")
    #expect(result.keywordRank == nil)
  }

  @Test func limitTruncatesResults() async throws {
    let (store, _) = try await makeStore()
    #expect(try await store.recall("cat", limit: 2).count == 2)
    #expect(try await store.recall("cat", limit: 0).isEmpty)
  }

  @Test func queryWithoutTermsUsesOnlyVectors() async throws {
    var vectors = Self.vectors
    vectors["🐈"] = [1, 0, 0]
    let store = try MemoryStore.inMemory(embedder: TestEmbedder(vectors: vectors))
    let cat = try await store.remember("The cat sleeps on the sofa.")
    let results = try await store.recall("🐈")
    #expect(results.map(\.id) == [cat.id])
    #expect(results[0].keywordRank == nil)
    #expect(results[0].vectorRank == 1)
  }

  @Test func unusableQueryVectorFindsNothing() async throws {
    let (store, _) = try await makeStore()
    // The test embedder returns a zero vector for text without words.
    #expect(try await store.recall("\"*)").isEmpty)
  }

  @Test(
    arguments: [
      (RecallConfiguration(k: -1), .k),
      (RecallConfiguration(k: .infinity), .k),
      (RecallConfiguration(k: .nan), .k),
      (RecallConfiguration(keywordWeight: -1), .weight),
      (RecallConfiguration(vectorWeight: .nan), .weight),
      (RecallConfiguration(recencyWeight: .infinity), .weight),
      (RecallConfiguration(candidateLimit: 0), .candidateLimit),
      (RecallConfiguration(minimumSimilarity: .nan), .minimumSimilarity),
      (RecallConfiguration(minimumSimilarity: .infinity), .minimumSimilarity),
      (RecallConfiguration(minimumSimilarity: 1.5), .minimumSimilarity),
      (RecallConfiguration(minimumSimilarity: -1.01), .minimumSimilarity),
      (
        RecallConfiguration(
          k: 0, keywordWeight: .greatestFiniteMagnitude, vectorWeight: .greatestFiniteMagnitude),
        .scoreOverflow
      ),
      (
        RecallConfiguration(
          k: 1, keywordWeight: .greatestFiniteMagnitude, vectorWeight: .greatestFiniteMagnitude,
          recencyWeight: .greatestFiniteMagnitude),
        .scoreOverflow
      ),
    ] as [(RecallConfiguration, RecallConfiguration.Problem)])
  func rejectsInvalidConfiguration(
    _ configuration: RecallConfiguration,
    problem: RecallConfiguration.Problem
  ) async throws {
    let store = try MemoryStore.inMemory()
    await #expect(throws: MemoryStoreError.invalidConfiguration(problem)) {
      try await store.recall("tea", configuration: configuration)
    }
  }

  @Test(arguments: [
    RecallConfiguration(minimumSimilarity: -1),
    RecallConfiguration(minimumSimilarity: 1),
    RecallConfiguration(k: 0, keywordWeight: .greatestFiniteMagnitude, vectorWeight: 0),
    RecallConfiguration(
      k: 1, keywordWeight: .greatestFiniteMagnitude, vectorWeight: .greatestFiniteMagnitude,
      recencyWeight: 0),
  ])
  func acceptsBoundaryConfiguration(_ configuration: RecallConfiguration) async throws {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder())
    try await store.remember("tea")
    let results = try await store.recall("tea", configuration: configuration)
    #expect(results.allSatisfy { $0.score.isFinite })
  }
}
