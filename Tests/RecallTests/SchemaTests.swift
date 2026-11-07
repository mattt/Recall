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
