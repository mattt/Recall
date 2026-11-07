import Foundation
import Testing

@testable import Recall

@Suite struct VectorTests {
  @Test func normalizesToUnitLength() throws {
    let vector = try #require(Vector.normalized([3, 4]))
    #expect(vector == [0.6, 0.8])
  }

  @Test(arguments: [[], [0, 0], [1, .nan], [.infinity, 1]] as [[Float]])
  func rejectsUnusableVectors(_ vector: [Float]) {
    #expect(Vector.normalized(vector) == nil)
  }

  @Test func dotProduct() {
    #expect(Vector.dot([1, 2, 3], [4, 5, 6]) == 32)
  }

  @Test func encodingRoundTrips() throws {
    let vector: [Float] = [0.25, -1, 3.5, .leastNonzeroMagnitude]
    let data = Vector.encode(vector)
    #expect(data.count == 16)
    #expect(Vector.decode(data, dimension: 4) == vector)
    #expect(Vector.decode(data, dimension: 3) == nil)
  }
}

@Suite struct VectorSearchTests {
  static let vectors: [String: [Float]] = [
    "cats": [1, 0, 0],
    "kittens": [0.9, 0.1, 0],
    "dogs": [0, 1, 0],
    "feline": [1, 0.05, 0],
    "taxes": [0, 0, 1],
  ]

  @Test func ranksByCosineSimilarity() async throws {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder(vectors: Self.vectors))
    let cats = try await store.remember("cats")
    let kittens = try await store.remember("kittens")
    let dogs = try await store.remember("dogs")
    let results = try await store.vectorSearch("feline", limit: 10)
    #expect(results.map(\.id) == [cats.id, kittens.id, dogs.id])
    #expect(abs(results[0].similarity - 0.99875) < 0.0001)
    #expect(try await store.vectorSearch("feline", limit: 1).map(\.id) == [cats.id])
  }

  @Test func minimumSimilarityFilters() async throws {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder(vectors: Self.vectors))
    let cats = try await store.remember("cats")
    _ = try await store.remember("taxes")
    let results = try await store.vectorSearch("feline", limit: 10, minimumSimilarity: 0.5)
    #expect(results.map(\.id) == [cats.id])
  }

  @Test func storeWithoutEmbedderHasNoVectorResults() async throws {
    let store = try MemoryStore.inMemory()
    _ = try await store.remember("cats")
    #expect(try await store.vectorSearch("cats", limit: 10).isEmpty)
  }

  @Test func failedEmbeddingSavesNothing() async throws {
    let store = try MemoryStore.inMemory(embedder: FailingEmbedder())
    await #expect(throws: FailingEmbedder.Failure.self) {
      try await store.remember("cats")
    }
    #expect(try await store.count() == 0)
  }

  @Test func rejectsZeroVector() async throws {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder(vectors: ["zero": [0, 0, 0]]))
    await #expect(throws: MemoryStoreError.invalidEmbedding(embedder: "test-v1")) {
      try await store.remember("zero")
    }
  }

  @Test func rejectsMixedDimensions() async throws {
    let embedder = TestEmbedder(vectors: ["short": [1, 0], "long": [1, 0, 0]])
    let store = try MemoryStore.inMemory(embedder: embedder)
    try await store.remember("short")
    await #expect(
      throws: MemoryStoreError.dimensionMismatch(embedder: "test-v1", expected: 2, actual: 3)
    ) {
      try await store.remember("long")
    }
    #expect(try await store.count() == 1)
  }

  @Test func rejectsQueryVectorOfAnotherDimension() async throws {
    let embedder = TestEmbedder(vectors: ["cats": [1, 0], "query": [1, 0, 0]])
    let store = try MemoryStore.inMemory(embedder: embedder)
    try await store.remember("cats")
    let error = MemoryStoreError.dimensionMismatch(embedder: "test-v1", expected: 2, actual: 3)
    await #expect(throws: error) {
      try await store.vectorSearch("query", limit: 10)
    }
    await #expect(throws: error) {
      try await store.recall("query")
    }
  }

  @Test func queryVectorOfAnyDimensionFindsNothingInEmptyStore() async throws {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder(vectors: ["query": [1, 0, 0]]))
    #expect(try await store.vectorSearch("query", limit: 10).isEmpty)
  }

  @Test func updatingTextReplacesVector() async throws {
    let embedder = TestEmbedder(vectors: Self.vectors)
    let store = try MemoryStore.inMemory(embedder: embedder)
    let memory = try await store.remember("dogs")
    try await store.update(memory.id, text: "cats")
    let results = try await store.vectorSearch("feline", limit: 1)
    #expect(results.first?.id == memory.id)
    #expect(try #require(results.first?.similarity) > 0.99)
  }

  @Test func updatingOnlyMetadataDoesNotEmbed() async throws {
    let embedder = TestEmbedder(vectors: Self.vectors)
    let store = try MemoryStore.inMemory(embedder: embedder)
    var memory = try await store.remember("dogs")
    memory.metadata = ["kind": "pet"]
    try await store.update(memory)
    #expect(embedder.calls.all == ["dogs"])
  }

  @Test func forgettingDeletesVectorsFromEveryEmbedder() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let store = try MemoryStore(url: url, embedder: TestEmbedder(vectors: Self.vectors))
    let other = try MemoryStore(url: url, embedder: TestEmbedder(identifier: "other"))
    let memory = try await store.remember("cats")
    let kept = try await store.remember("dogs")
    try await other.reembed()
    #expect(try await store.embeddingRowCount() == 4)

    try await store.forget(memory.id)
    // Look at the table itself; a search would also miss a vector without a memory.
    #expect(try await store.embeddingRowCount() == 2)
    #expect(try await store.embeddingRowCount(for: memory.id) == 0)
    #expect(try await store.embeddingRowCount(for: kept.id) == 2)
  }

  @Test func failedVectorSaveRollsBackTextUpdate() async throws {
    let embedder = TestEmbedder(vectors: ["cats": [1, 0], "dogs": [0, 1], "long": [1, 0, 0]])
    let store = try MemoryStore.inMemory(embedder: embedder)
    try await store.remember("cats")
    let memory = try await store.remember("dogs")

    // The update changes the text and deletes the old vector,
    // then fails to save a vector of the wrong dimension.
    await #expect(
      throws: MemoryStoreError.dimensionMismatch(embedder: "test-v1", expected: 2, actual: 3)
    ) {
      try await store.update(memory.id, text: "long")
    }
    #expect(try await store.memory(memory.id) == memory)
    #expect(try await store.keywordSearch("long", limit: 10).isEmpty)
    #expect(try await store.keywordSearch("dogs", limit: 10) == [memory.id])
    #expect(try await store.embeddingRowCount(for: memory.id) == 1)
    #expect(try await store.staleEmbeddingCount() == 0)
  }
}

extension MemoryStore {
  /// The number of rows in the embeddings table,
  /// for all memories or for one.
  func embeddingRowCount(for id: UUID? = nil) throws -> Int {
    let statement = try database.prepare(
      """
      SELECT count(*) FROM embeddings e
      WHERE ?1 IS NULL OR e.memory_rowid = (SELECT rowid FROM memories WHERE id = ?1)
      """
    )
    try statement.bind(.optionalText(id?.uuidString))
    try statement.step()
    return statement.int(at: 0)
  }
}

@Suite struct ReembeddingTests {
  @Test func newEmbedderReembedsEveryMemory() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let first = TestEmbedder(identifier: "first", vectors: ["cats": [1, 0], "dogs": [0, 1]])
    do {
      let store = try MemoryStore(url: url, embedder: first)
      try await store.remember("cats")
      try await store.remember("dogs")
    }

    // The second embedder maps the texts the other way round,
    // so a search returns the right memory only after re-embedding.
    let second = TestEmbedder(
      identifier: "second",
      vectors: ["cats": [0, 1, 0], "dogs": [1, 0, 0], "query": [1, 0, 0]]
    )
    let store = try MemoryStore(url: url, embedder: second)
    #expect(try await store.staleEmbeddingCount() == 2)
    let results = try await store.vectorSearch("query", limit: 1)
    #expect(try await store.memory(try #require(results.first?.id))?.text == "dogs")
    #expect(try await store.staleEmbeddingCount() == 0)
    #expect(Set(second.calls.all) == ["cats", "dogs", "query"])

    // Nothing is stale now, so no more texts are embedded.
    #expect(try await store.reembed() == 0)
    #expect(second.calls.count == 3)
  }

  @Test func sameEmbedderDoesNotReembed() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    do {
      let store = try MemoryStore(url: url, embedder: TestEmbedder())
      try await store.remember("cats")
    }
    let embedder = TestEmbedder()
    let store = try MemoryStore(url: url, embedder: embedder)
    #expect(try await store.reembed() == 0)
    #expect(embedder.calls.count == 0)
  }

  @Test func memoriesSavedWithoutEmbedderAreEmbeddedLater() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    try await MemoryStore(url: url).remember("cats")
    let store = try MemoryStore(url: url, embedder: TestEmbedder())
    #expect(try await store.reembed() == 1)
  }

  @Test func reembeddingSkipsTextChangedByAnotherConnection() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let other = try MemoryStore(url: url)
    let memory = try await other.remember("cats")

    let gate = Gate()
    let base = TestEmbedder(vectors: ["cats": [1, 0], "dogs": [0, 1], "query": [0, 1]])
    let store = try MemoryStore(url: url, embedder: GatedEmbedder(base, gates: ["cats": gate]))
    let reembedding = Task { try await store.reembed() }
    await gate.arrival()

    // The other connection changes the text while the embedder has the old text.
    try await other.update(memory.id, text: "dogs")
    gate.open()
    #expect(try await reembedding.value == 0)
    #expect(try await store.staleEmbeddingCount() == 1)

    // The next call embeds the new text.
    #expect(try await store.reembed() == 1)
    let results = try await store.vectorSearch("query", limit: 1)
    #expect(results.first?.id == memory.id)
    #expect(try #require(results.first?.similarity) > 0.99)
  }

  @Test func reembeddingSkipsMemoryForgottenByAnotherConnection() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let other = try MemoryStore(url: url)
    let memory = try await other.remember("cats")

    let gate = Gate()
    let store = try MemoryStore(url: url, embedder: GatedEmbedder(gates: ["cats": gate]))
    let reembedding = Task { try await store.reembed() }
    await gate.arrival()
    try await other.forget(memory.id)
    gate.open()
    #expect(try await reembedding.value == 0)
    #expect(try await store.count() == 0)
  }

  @Test func concurrentReembedCallsShareOneTask() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let plain = try MemoryStore(url: url)
    for index in 0..<20 {
      try await plain.remember("memory number \(index)")
    }
    // The first memory's embedding waits at the gate,
    // so every call starts while the shared work is still running.
    let gate = Gate()
    let embedder = GatedEmbedder(gates: ["memory number 0": gate])
    let store = try MemoryStore(url: url, embedder: embedder)
    let calls = (0..<4).map { _ in Task { try await store.reembed() } }
    await gate.arrival()
    try await waitUntil { await store.reembedWaiterCount == 4 }
    gate.open()

    var counts: [Int] = []
    for call in calls {
      counts.append(try await call.value)
    }
    #expect(counts == [20, 20, 20, 20])
    #expect(embedder.base.calls.count == 20)
  }

  @Test func partialReembeddingKeepsSavedVectors() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let plain = try MemoryStore(url: url)
    for text in ["one", "two", "three"] {
      try await plain.remember(text)
    }

    // Memories are embedded in the order they were saved,
    // so "one" is saved before "two" fails, and "three" isn't reached.
    let failing = SelectivelyFailingEmbedder(base: TestEmbedder(), failing: ["two"])
    let store = try MemoryStore(url: url, embedder: failing)
    await #expect(throws: FailingEmbedder.Failure.self) {
      try await store.reembed()
    }
    #expect(try await store.staleEmbeddingCount() == 2)
    #expect(failing.base.calls.all == ["one"])

    // A later call with a working embedder embeds only what is still stale.
    let embedder = TestEmbedder()
    let retry = try MemoryStore(url: url, embedder: embedder)
    #expect(try await retry.reembed() == 2)
    #expect(embedder.calls.all == ["two", "three"])
  }
}
