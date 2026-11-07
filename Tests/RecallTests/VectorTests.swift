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
    await #expect(throws: MemoryStoreError.invalidEmbedding(embedder: "test-v1")) {
      try await store.remember("long")
    }
    #expect(try await store.count() == 1)
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

  @Test func forgettingDeletesVector() async throws {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder(vectors: Self.vectors))
    let memory = try await store.remember("cats")
    try await store.forget(memory.id)
    #expect(try await store.vectorSearch("feline", limit: 10).isEmpty)
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

  @Test func concurrentReembedCallsShareOneTask() async throws {
    let embedder = TestEmbedder(identifier: "second")
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let plain = try MemoryStore(url: url)
    for index in 0..<20 {
      try await plain.remember("memory number \(index)")
    }
    let store = try MemoryStore(url: url, embedder: embedder)
    let counts = try await withThrowingTaskGroup(of: Int.self) { group in
      for _ in 0..<4 {
        group.addTask { try await store.reembed() }
      }
      return try await group.reduce(into: []) { $0.append($1) }
    }
    #expect(embedder.calls.count == 20)
    #expect(counts.contains(20))
  }
}
