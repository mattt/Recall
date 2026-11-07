import Foundation
import Testing

@testable import Recall

@Suite struct SchemaTests {
  @Test func newDatabaseHasCurrentVersion() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    _ = try MemoryStore(url: url)
    #expect(try Database(url: url).userVersion == Schema.currentVersion)
  }

  @Test func reopeningDoesNotRerunMigrations() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    _ = try await MemoryStore(url: url).remember("First")
    // A rerun of migration 1 would fail because the tables exist.
    let store = try MemoryStore(url: url)
    #expect(try await store.count() == 1)
  }

  @Test func migratesVersion1Database() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let id = UUID()
    do {
      let database = try Database(url: url)
      try Schema.migrate(database, to: 1)
      try database.prepare(
        "INSERT INTO memories (id, text, created_at, source) VALUES (?, ?, ?, ?)"
      )
      .bind(.text(id.uuidString), .text("Written by version 1."), .real(0), .text("old"))
      .step()
      #expect(try database.userVersion == 1)
    }

    let store = try MemoryStore(url: url, embedder: TestEmbedder())
    #expect(try Database(url: url).userVersion == Schema.currentVersion)
    let memory = try #require(try await store.memory(id))
    #expect(memory.text == "Written by version 1.")
    #expect(memory.source == "old")
    #expect(memory.createdAt == Date(timeIntervalSinceReferenceDate: 0))
    #expect(try await store.keywordSearch("version", limit: 10) == [id])
    #expect(try await store.reembed() == 1)
    #expect(try await store.vectorSearch("version 1", limit: 10).map(\.id) == [id])
  }

  @Test func migratesVersion2DatabaseAndKeepsVectors() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let id = UUID()
    do {
      let database = try Database(url: url)
      try database.execute("PRAGMA foreign_keys = ON")
      try Schema.migrate(database, to: 2)
      try database.prepare("INSERT INTO memories (id, text, created_at) VALUES (?, ?, ?)")
        .bind(.text(id.uuidString), .text("cats"), .real(0))
        .step()
      try database.prepare(
        """
        INSERT INTO embeddings (memory_rowid, embedder, dimension, vector)
        SELECT rowid, 'test-v1', 2, ? FROM memories
        """
      )
      .bind(.blob(Vector.encode([1, 0])))
      .step()
      #expect(try database.userVersion == 2)
    }

    let embedder = TestEmbedder(vectors: ["query": [1, 0], "dogs": [0, 1]])
    let store = try MemoryStore(url: url, embedder: embedder)
    #expect(try Database(url: url).userVersion == Schema.currentVersion)

    // The version 2 vector is still there, so nothing is embedded again.
    #expect(try await store.staleEmbeddingCount() == 0)
    let results = try await store.vectorSearch("query", limit: 10)
    #expect(results.map(\.id) == [id])
    #expect(embedder.calls.all == ["query"])

    // A second embedder adds its own vector beside the first.
    let other = try MemoryStore(url: url, embedder: TestEmbedder(identifier: "test-v2"))
    #expect(try await other.reembed() == 1)
    #expect(try await store.staleEmbeddingCount() == 0)

    // The migrated trigger still deletes every vector when the text changes.
    try await MemoryStore(url: url).update(id, text: "dogs")
    #expect(try await store.staleEmbeddingCount() == 1)
    #expect(try await other.staleEmbeddingCount() == 1)
  }

  @Test func refusesNewerSchema() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let newer = Schema.currentVersion + 1
    try Database(url: url).execute("PRAGMA user_version = \(newer)")
    #expect(throws: MemoryStoreError.unsupportedSchemaVersion(newer)) {
      try MemoryStore(url: url)
    }
  }
}
