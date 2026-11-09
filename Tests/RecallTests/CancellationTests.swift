import Foundation
import Testing

@testable import Recall

@Suite struct CancellationTests {
  @Test func canceledCallDoesNoWork() async throws {
    let embedder = TestEmbedder()
    let store = try MemoryStore.inMemory(embedder: embedder)
    let memory = try await store.remember("cats")
    let task = Task {
      withUnsafeCurrentTask { $0?.cancel() }
      _ = try? await store.remember("dogs")
      _ = try? await store.update(memory.id, text: "birds")
      _ = try? await store.recall("cats")
      _ = try? await store.reembed()
    }
    await task.value
    #expect(embedder.calls.all == ["cats"])
    #expect(try await store.list().map(\.text) == ["cats"])
  }

  @Test func rememberCanceledWhileEmbeddingSavesNothing() async throws {
    let gate = Gate()
    let store = try MemoryStore.inMemory(embedder: GatedEmbedder(gates: ["cats": gate]))
    let task = Task { try await store.remember("cats") }
    await gate.arrival()
    task.cancel()
    gate.open()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(try await store.count() == 0)
  }

  @Test func textUpdateCanceledWhileEmbeddingSavesNothing() async throws {
    let gate = Gate()
    let store = try MemoryStore.inMemory(embedder: GatedEmbedder(gates: ["dogs": gate]))
    let memory = try await store.remember("cats")
    let task = Task { try await store.update(memory.id, text: "dogs") }
    await gate.arrival()
    task.cancel()
    gate.open()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(try await store.memory(memory.id) == memory)
    #expect(try await store.staleEmbeddingCount() == 0)
  }

  @Test func memoryUpdateCanceledWhileEmbeddingSavesNothing() async throws {
    let gate = Gate()
    let store = try MemoryStore.inMemory(embedder: GatedEmbedder(gates: ["dogs": gate]))
    let memory = try await store.remember("cats")
    var changed = memory
    changed.text = "dogs"
    changed.metadata = ["kind": "pet"]
    let task = Task { [changed] in try await store.update(changed) }
    await gate.arrival()
    task.cancel()
    gate.open()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(try await store.memory(memory.id) == memory)
  }

  @Test func canceledWaiterStopsWaitingForSharedReembedding() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let plain = try MemoryStore(url: url)
    for text in ["one", "two", "three"] {
      try await plain.remember(text)
    }
    let gate = Gate()
    let embedder = GatedEmbedder(gates: ["one": gate])
    let store = try MemoryStore(url: url, embedder: embedder)

    let first = Task { try await store.reembed() }
    await gate.arrival()
    let second = Task { try await store.reembed() }
    try await waitUntil { await store.reembedWaiterCount == 2 }

    // The second call stops waiting while the shared work is still paused.
    second.cancel()
    await #expect(throws: CancellationError.self) { try await second.value }
    #expect(await store.reembedWaiterCount == 1)

    gate.open()
    #expect(try await first.value == 3)
    #expect(try await store.staleEmbeddingCount() == 0)
  }

  @Test func reembeddingContinuesWhenItsOnlyWaiterIsCanceled() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let plain = try MemoryStore(url: url)
    for text in ["one", "two", "three"] {
      try await plain.remember(text)
    }
    let gate = Gate()
    let embedder = GatedEmbedder(gates: ["one": gate])
    let store = try MemoryStore(url: url, embedder: embedder)

    let task = Task { try await store.reembed() }
    await gate.arrival()
    task.cancel()
    await #expect(throws: CancellationError.self) { try await task.value }

    // The work goes on without a waiter, and a later call doesn't start it again.
    gate.open()
    try await waitUntil { await !store.isReembedding }
    #expect(try await store.staleEmbeddingCount() == 0)
    #expect(try await store.reembed() == 0)
    #expect(embedder.base.calls.all.sorted() == ["one", "three", "two"])
  }
}

/// Waits until `condition` is true, for up to five seconds.
func waitUntil(_ condition: () async -> Bool) async throws {
  let deadline = ContinuousClock.now + .seconds(5)
  while !(await condition()) {
    guard ContinuousClock.now < deadline else {
      Issue.record("The condition didn't become true in time.")
      return
    }
    try await Task.sleep(for: .milliseconds(1))
  }
}
