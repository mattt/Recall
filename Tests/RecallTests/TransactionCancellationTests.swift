import Foundation
import Testing

@testable import Recall

#if canImport(SQLite3)
  import SQLite3
#else
  import CSQLite
#endif

/// Tests that cancel a write while SQLite runs its statements,
/// after the checks that happen before the transaction starts.
@Suite struct TransactionCancellationTests {
  static let vectors: [String: [Float]] = ["cats": [1, 0], "dogs": [0, 1]]

  @Test func rememberCanceledDuringWriteSavesNothing() async throws {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder(vectors: Self.vectors))
    // The vector is the last row that `remember` writes.
    try await store.cancelTask(on: "AFTER INSERT ON embeddings")
    let task = Task { try await store.remember("dogs") }
    await #expect(throws: CancellationError.self) { try await task.value }

    try await store.removeCancelTrigger()
    #expect(try await store.count() == 0)
    #expect(try await store.keywordSearch("dogs", limit: 10).isEmpty)
    #expect(try await store.embeddingRowCount() == 0)
    // The rollback leaves the connection ready for the next write.
    try await store.remember("dogs")
    #expect(try await store.count() == 1)
  }

  @Test func textUpdateCanceledDuringWriteSavesNothing() async throws {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder(vectors: Self.vectors))
    let memory = try await store.remember("cats")
    try await store.cancelTask(on: "AFTER INSERT ON embeddings")
    let task = Task { try await store.update(memory.id, text: "dogs") }
    await #expect(throws: CancellationError.self) { try await task.value }

    try await store.removeCancelTrigger()
    try await expectUnchanged(memory, in: store)
  }

  @Test func memoryUpdateCanceledDuringWriteSavesNothing() async throws {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder(vectors: Self.vectors))
    let memory = try await store.remember("cats")
    var changed = memory
    changed.text = "dogs"
    changed.source = "changed"
    changed.metadata = ["kind": "pet"]
    try await store.cancelTask(on: "AFTER INSERT ON embeddings")
    let task = Task { [changed] in try await store.update(changed) }
    await #expect(throws: CancellationError.self) { try await task.value }

    try await store.removeCancelTrigger()
    try await expectUnchanged(memory, in: store)
  }

  @Test func metadataUpdateCanceledDuringWriteSavesNothing() async throws {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder(vectors: Self.vectors))
    let memory = try await store.remember("cats")
    var changed = memory
    changed.metadata = ["kind": "pet"]
    // This update doesn't embed, so it writes only the memory row.
    try await store.cancelTask(on: "AFTER UPDATE ON memories")
    let task = Task { [changed] in try await store.update(changed) }
    await #expect(throws: CancellationError.self) { try await task.value }

    try await store.removeCancelTrigger()
    try await expectUnchanged(memory, in: store)
  }

  /// Checks that the memory row, its keyword index, and its vector have their earlier values.
  func expectUnchanged(_ memory: Memory, in store: MemoryStore) async throws {
    #expect(try await store.memory(memory.id) == memory)
    #expect(try await store.keywordSearch("dogs", limit: 10).isEmpty)
    #expect(try await store.keywordSearch("cats", limit: 10) == [memory.id])
    #expect(try await store.embeddingRowCount(for: memory.id) == 1)
    #expect(try await store.staleEmbeddingCount() == 0)
    let results = try await store.vectorSearch("cats", limit: 1)
    #expect(results.first?.id == memory.id)
    #expect(try #require(results.first?.similarity) > 0.99)
  }
}

/// A SQLite function that cancels the Swift task that is running the statement.
private func cancelCurrentTask(
  _ context: OpaquePointer?,
  _ count: Int32,
  _ values: UnsafeMutablePointer<OpaquePointer?>?
) {
  withUnsafeCurrentTask { $0?.cancel() }
  sqlite3_result_null(context)
}

extension MemoryStore {
  /// Adds a temporary trigger that cancels the current task when `event` happens,
  /// such as `"AFTER INSERT ON embeddings"`.
  func cancelTask(on event: String) throws {
    let code = sqlite3_create_function_v2(
      database.handle, "cancel_current_task", 0, SQLITE_UTF8, nil, cancelCurrentTask, nil, nil,
      nil)
    guard code == SQLITE_OK else {
      throw MemoryStoreError.sqlite(code: code, message: database.errorMessage)
    }
    try database.execute(
      "CREATE TEMP TRIGGER cancel_task \(event) BEGIN SELECT cancel_current_task(); END"
    )
  }

  func removeCancelTrigger() throws {
    try database.execute("DROP TRIGGER temp.cancel_task")
  }
}
