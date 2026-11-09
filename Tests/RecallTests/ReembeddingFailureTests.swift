import Foundation
import Testing

@testable import Recall

/// An embedder that throws `CancellationError` for one text.
private struct CancelingEmbedder: EmbedderSource {
  let base = TestEmbedder()
  let canceling: String

  var identifier: String { base.identifier }

  func embed(_ text: String, for purpose: EmbeddingPurpose) async throws -> [Float] {
    if text == canceling { throw CancellationError() }
    return try await base.embed(text, for: purpose)
  }
}

@Suite struct ReembeddingFailureTests {
  private static let bad = "zebra stripes"

  /// Opens a file that a store with no embedder filled,
  /// with an embedder that can't embed ``bad``.
  private static func makeStore(
    in directory: TemporaryDirectory,
    texts: [String]
  ) async throws -> (store: MemoryStore, embedder: SelectivelyFailingEmbedder, ids: [String: UUID])
  {
    let url = directory.url.appending(path: "memories.sqlite")
    let plain = try MemoryStore(url: url)
    var ids: [String: UUID] = [:]
    for text in texts {
      ids[text] = try await plain.remember(text).id
    }
    let embedder = SelectivelyFailingEmbedder(base: TestEmbedder(), failing: [bad])
    return (try MemoryStore(url: url, embedder: embedder), embedder, ids)
  }

  @Test(arguments: [
    ["zebra stripes", "cats purr", "dogs bark"],
    ["cats purr", "zebra stripes", "dogs bark"],
    ["cats purr", "dogs bark", "zebra stripes"],
  ])
  func recallFindsOthersByVectorAndTheFailedMemoryByKeyword(texts: [String]) async throws {
    let directory = try TemporaryDirectory()
    let (store, embedder, ids) = try await Self.makeStore(in: directory, texts: texts)

    let vectorResults = try await store.recall("cats purr")
    let cats = try #require(vectorResults.first { $0.id == ids["cats purr"] })
    #expect(cats.vectorRank != nil)
    #expect(try await store.staleEmbeddingCount() == 1)
    #expect(Set(embedder.base.calls.all).isSuperset(of: ["cats purr", "dogs bark"]))

    let keywordResults = try await store.recall("zebra")
    let zebra = try #require(keywordResults.first { $0.id == ids[Self.bad] })
    #expect(zebra.keywordRank == 1)
    #expect(zebra.vectorRank == nil)
  }

  @Test func recallSkipsTextWithAZeroVector() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let plain = try MemoryStore(url: url)
    let good = try await plain.remember("cats")
    try await plain.remember("\u{200B}")

    let embedder = TestEmbedder(vectors: ["cats": [1, 0], "\u{200B}": [0, 0], "query": [1, 0]])
    let store = try MemoryStore(url: url, embedder: embedder)
    let results = try await store.recall("query")
    #expect(results.map(\.id) == [good.id])
  }

  @Test func onlyMemoryThatFailsIsRememberedAfterTheQueryWorks() async throws {
    let directory = try TemporaryDirectory()
    let (store, embedder, _) = try await Self.makeStore(in: directory, texts: [Self.bad])

    _ = try await store.recall("query")
    _ = try await store.recall("query")
    #expect(embedder.failures.count == 1)
  }

  @Test func directReembedReportsFailureAfterSavingTheRest() async throws {
    let directory = try TemporaryDirectory()
    let (store, embedder, ids) = try await Self.makeStore(
      in: directory,
      texts: ["cats purr", Self.bad, "dogs bark"]
    )

    let error = await #expect(throws: ReembeddingError.self) {
      try await store.reembed()
    }
    let failure = try #require(error)
    #expect(failure.count == 1)
    #expect(failure.failures.map(\.id) == [try #require(ids[Self.bad])])
    #expect(failure.failures.first?.error is FailingEmbedder.Failure)
    #expect(failure.embeddedCount == 2)
    #expect(failure.localizedDescription.contains("1 memory"))
    #expect(failure.localizedDescription.contains(try #require(ids[Self.bad]).uuidString))
    #expect(embedder.base.calls.all == ["cats purr", "dogs bark"])
    #expect(try await store.staleEmbeddingCount() == 1)
  }

  @Test func failedMemoryIsNotEmbeddedAgainOnTheNextSearch() async throws {
    let directory = try TemporaryDirectory()
    let (store, embedder, _) = try await Self.makeStore(
      in: directory,
      texts: [Self.bad, "cats purr"]
    )

    _ = try await store.recall("cats")
    #expect(embedder.failures.all == [Self.bad])
    _ = try await store.recall("cats")
    #expect(embedder.failures.all == [Self.bad])

    // The record is in the database, so another connection also skips the memory.
    let url = directory.url.appending(path: "memories.sqlite")
    let reopenedEmbedder = SelectivelyFailingEmbedder(base: TestEmbedder(), failing: [Self.bad])
    let reopened = try MemoryStore(url: url, embedder: reopenedEmbedder)
    _ = try await reopened.recall("cats")
    #expect(reopenedEmbedder.failures.count == 0)
  }

  @Test func directReembedRetriesAFailedMemory() async throws {
    let directory = try TemporaryDirectory()
    let (store, embedder, ids) = try await Self.makeStore(
      in: directory,
      texts: [Self.bad, "cats purr"]
    )
    _ = try await store.recall("cats")
    #expect(embedder.failures.count == 1)

    await #expect(throws: ReembeddingError.self) {
      try await store.reembed()
    }
    #expect(embedder.failures.count == 2)

    // A working embedder with the same identifier replaces the record with a vector.
    let url = directory.url.appending(path: "memories.sqlite")
    let working = TestEmbedder()
    let fixed = try MemoryStore(url: url, embedder: working)
    #expect(try await fixed.reembed() == 1)
    #expect(working.calls.all == [Self.bad])
    #expect(try await fixed.staleEmbeddingCount() == 0)
    #expect(try await fixed.embeddingRowCount(for: try #require(ids[Self.bad])) == 1)
  }

  @Test func editingTheTextRetriesAFailedMemory() async throws {
    let directory = try TemporaryDirectory()
    let (store, embedder, ids) = try await Self.makeStore(
      in: directory,
      texts: [Self.bad, "cats purr"]
    )
    _ = try await store.recall("cats")
    #expect(try await store.staleEmbeddingCount() == 1)
    let id = try #require(ids[Self.bad])

    // Another connection edits the text, so this store didn't embed the new text.
    let url = directory.url.appending(path: "memories.sqlite")
    try await MemoryStore(url: url).update(id, text: "horses gallop")
    #expect(try await store.embeddingRowCount(for: id) == 0)

    _ = try await store.recall("horses")
    #expect(embedder.base.calls.all.contains("horses gallop"))
    #expect(try await store.staleEmbeddingCount() == 0)
    #expect(try await store.vectorSearch("horses gallop", limit: 1).first?.id == id)
  }

  @Test func anotherEmbedderRetriesAFailedMemory() async throws {
    let directory = try TemporaryDirectory()
    let (store, _, _) = try await Self.makeStore(in: directory, texts: [Self.bad, "cats purr"])
    _ = try await store.recall("cats")

    let url = directory.url.appending(path: "memories.sqlite")
    let other = TestEmbedder(identifier: "other")
    let otherStore = try MemoryStore(url: url, embedder: other)
    _ = try await otherStore.recall("zebra")
    #expect(other.calls.all.contains(Self.bad))
    #expect(try await otherStore.staleEmbeddingCount() == 0)
  }

  @Test func recordedFailureDoesNotAffectVectorSearch() async throws {
    let directory = try TemporaryDirectory()
    let (store, _, ids) = try await Self.makeStore(in: directory, texts: [Self.bad, "cats purr"])
    _ = try await store.recall("cats")

    let results = try await store.vectorSearch("cats purr", limit: 10)
    #expect(results.map(\.id) == [try #require(ids["cats purr"])])
    // A saved vector isn't compared with the record, which has no dimension.
    try await store.remember("birds sing")
    #expect(try await store.staleEmbeddingCount() == 1)
  }

  @Test func outageStopsThePassAndRecordsNothing() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let plain = try MemoryStore(url: url)
    let texts = (0..<20).map { "memory number \($0)" }
    for text in texts {
      try await plain.remember(text)
    }

    let down = SelectivelyFailingEmbedder(base: TestEmbedder(), failing: Set(texts + ["query"]))
    let store = try MemoryStore(url: url, embedder: down)
    await #expect(throws: FailingEmbedder.Failure.self) {
      try await store.recall("query")
    }
    #expect(down.failures.count == 8)
    #expect(try await store.embeddingRowCount() == 0)

    // Nothing was recorded, so the memories are embedded when the service returns.
    let working = TestEmbedder()
    let restored = try MemoryStore(url: url, embedder: working)
    _ = try await restored.recall("query")
    #expect(try await restored.staleEmbeddingCount() == 0)
    #expect(Set(working.calls.all).isSuperset(of: Set(texts)))
  }

  @Test func outageThatEndsThePassRecordsNothingAndKeepsVectors() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let plain = try MemoryStore(url: url)
    try await plain.remember("cats purr")
    try await plain.remember("dogs bark")

    // The service answers for "cats purr", then goes down for the rest of the recall.
    let down = SelectivelyFailingEmbedder(base: TestEmbedder(), failing: ["dogs bark", "query"])
    let store = try MemoryStore(url: url, embedder: down)
    await #expect(throws: FailingEmbedder.Failure.self) {
      try await store.recall("query")
    }
    #expect(try await store.embeddingRowCount() == 1)

    // The memory that failed is tried again, because no later vector confirmed the failure.
    await #expect(throws: FailingEmbedder.Failure.self) {
      try await store.recall("query")
    }
    #expect(down.failures.all == ["dogs bark", "query", "dogs bark", "query"])
    #expect(try await store.embeddingRowCount() == 1)
  }

  @Test func queryFailureStillThrows() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    try await MemoryStore(url: url).remember("cats")
    let embedder = SelectivelyFailingEmbedder(base: TestEmbedder(), failing: ["query"])
    let store = try MemoryStore(url: url, embedder: embedder)
    await #expect(throws: FailingEmbedder.Failure.self) {
      try await store.recall("query")
    }
    #expect(try await store.staleEmbeddingCount() == 0)
  }

  @Test func cancellationFromTheEmbedderStopsTheWorkAndRecordsNothing() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let plain = try MemoryStore(url: url)
    for text in ["one", "two", "three"] {
      try await plain.remember(text)
    }
    let embedder = CancelingEmbedder(canceling: "two")
    let store = try MemoryStore(url: url, embedder: embedder)

    await #expect(throws: CancellationError.self) {
      try await store.recall("one")
    }
    await #expect(throws: CancellationError.self) {
      try await store.reembed()
    }
    #expect(try await store.staleEmbeddingCount() == 2)
    #expect(try await store.embeddingRowCount() == 1)
    #expect(!embedder.base.calls.all.contains("three"))
  }

  @Test func canceledRecallStopsWaitingForReembedding() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    try await MemoryStore(url: url).remember("cats")
    let gate = Gate()
    let store = try MemoryStore(url: url, embedder: GatedEmbedder(gates: ["cats": gate]))

    let recall = Task { try await store.recall("cats") }
    await gate.arrival()
    recall.cancel()
    await #expect(throws: CancellationError.self) { try await recall.value }

    gate.open()
    try await waitUntil { await !store.isReembedding }
    #expect(try await store.staleEmbeddingCount() == 0)
  }

  @Test func directReembedWaitsForARecallAndThenRetries() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let plain = try MemoryStore(url: url)
    try await plain.remember(Self.bad)
    try await plain.remember("cats purr")

    let gate = Gate()
    let failing = SelectivelyFailingEmbedder(base: TestEmbedder(), failing: [Self.bad])
    let gated = Embedder(identifier: failing.identifier) { text, purpose in
      if text == "dogs bark" { await gate.pass() }
      return try await failing.embed(text, for: purpose)
    }
    let store = try MemoryStore(url: url, embedder: gated)
    _ = try await store.recall("cats")
    #expect(failing.failures.count == 1)

    // A new stale memory makes the next recall's pass stop at the gate.
    try await plain.remember("dogs bark")
    let recall = Task { try await store.recall("cats") }
    await gate.arrival()
    let direct = Task { try await store.reembed() }
    try await waitUntil { await store.reembedWaiterCount == 2 }
    #expect(failing.failures.count == 1)
    gate.open()

    _ = try await recall.value
    await #expect(throws: ReembeddingError.self) {
      try await direct.value
    }
    #expect(failing.failures.count == 2)
  }
}
