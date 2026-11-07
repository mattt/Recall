import Foundation
import Testing

@testable import Recall

@Suite struct ConcurrencyTests {
  @Test func manyTasksShareOneStore() async throws {
    let store = try MemoryStore.inMemory(embedder: TestEmbedder())
    let ids = try await withThrowingTaskGroup(of: UUID?.self) { group in
      for index in 0..<200 {
        group.addTask {
          switch index % 4 {
          case 0, 1:
            return try await store.remember("note \(index) about tea").id
          case 2:
            _ = try await store.recall("tea")
            return nil
          default:
            _ = try await store.list(limit: 10)
            return nil
          }
        }
      }
      return try await group.reduce(into: Set<UUID>()) { set, id in
        if let id { set.insert(id) }
      }
    }
    #expect(ids.count == 100)
    #expect(try await store.count() == 100)
    #expect(try await store.staleEmbeddingCount() == 0)

    // Forget half of the memories while other tasks recall.
    let doomed = Array(ids.prefix(50))
    try await withThrowingTaskGroup(of: Void.self) { group in
      for id in doomed {
        group.addTask { try await store.forget(id) }
        group.addTask { _ = try await store.recall("tea", limit: 100) }
      }
      try await group.waitForAll()
    }
    #expect(try await store.count() == 50)
    let remaining = Set(try await store.recall("tea", limit: 100).map(\.id))
    #expect(remaining == ids.subtracting(doomed))
  }

  @Test func twoStoresShareOneFile() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let first = try MemoryStore(url: url, embedder: TestEmbedder())
    let second = try MemoryStore(url: url, embedder: TestEmbedder())
    try await withThrowingTaskGroup(of: Void.self) { group in
      for index in 0..<50 {
        group.addTask { try await first.remember("first \(index)") }
        group.addTask { try await second.remember("second \(index)") }
        group.addTask { _ = try await second.recall("first") }
      }
      try await group.waitForAll()
    }
    #expect(try await first.count() == 100)
    #expect(try await second.count() == 100)
    // The first store sees every memory that the second store wrote.
    let results = try await first.recall("second", limit: 100)
    #expect(results.filter { $0.keywordRank != nil }.count == 50)
  }

  @Test func storesOpenedTogetherMigrateOnce() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let stores = try await withThrowingTaskGroup(of: MemoryStore.self) { group in
      for _ in 0..<8 {
        group.addTask { try MemoryStore(url: url) }
      }
      return try await group.reduce(into: []) { $0.append($1) }
    }
    #expect(stores.count == 8)
    #expect(try Database(url: url).userVersion == Schema.currentVersion)
  }
}
