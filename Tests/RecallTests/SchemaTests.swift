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
