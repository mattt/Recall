import Foundation
import Testing

@testable import Recall

@Suite struct MemoryStoreTests {
  @Test func rememberAndReadBack() async throws {
    let store = try MemoryStore.inMemory()
    let date = Date(timeIntervalSince1970: 1_700_000_000.25)
    let saved = try await store.remember(
      "The user prefers tea to coffee.",
      source: "iMCP",
      metadata: ["topic": "drinks"],
      createdAt: date
    )
    let loaded = try #require(try await store.memory(saved.id))
    #expect(loaded == saved)
    #expect(loaded.createdAt == date)
    #expect(try await store.count() == 1)
  }

  @Test func rejectsEmptyText() async throws {
    let store = try MemoryStore.inMemory()
    await #expect(throws: MemoryStoreError.emptyText) {
      try await store.remember(" \n\t")
    }
  }

  @Test func updateKeepsIdentifierAndDate() async throws {
    let store = try MemoryStore.inMemory()
    let original = try await store.remember("Meeting on Tuesday.")
    var changed = original
    changed.text = "Meeting on Wednesday."
    changed.source = nil
    changed.metadata = ["moved": "yes"]
    let saved = try await store.update(changed)
    #expect(saved == changed)
    #expect(saved.createdAt == original.createdAt)
    #expect(try await store.keywordSearch("Tuesday", limit: 10).isEmpty)
    #expect(try await store.keywordSearch("Wednesday", limit: 10) == [original.id])
  }

  @Test func updateMissingMemoryThrows() async throws {
    let store = try MemoryStore.inMemory()
    let id = UUID()
    await #expect(throws: MemoryStoreError.notFound(id)) {
      try await store.update(id, text: "Anything")
    }
    await #expect(throws: MemoryStoreError.notFound(id)) {
      try await store.update(Memory(id: id, text: "Anything"))
    }
  }

  @Test func forgetRemovesMemoryAndIndex() async throws {
    let store = try MemoryStore.inMemory()
    let memory = try await store.remember("Parking spot is B12.")
    #expect(try await store.forget(memory.id))
    #expect(try await store.forget(memory.id) == false)
    #expect(try await store.memory(memory.id) == nil)
    #expect(try await store.keywordSearch("parking", limit: 10).isEmpty)
  }

  @Test func listIsNewestFirstWithPaging() async throws {
    let store = try MemoryStore.inMemory()
    var ids: [UUID] = []
    for index in 0..<5 {
      let memory = try await store.remember(
        "Memory \(index)",
        source: index.isMultiple(of: 2) ? "even" : "odd",
        createdAt: Date(timeIntervalSince1970: Double(index))
      )
      ids.append(memory.id)
    }
    #expect(try await store.list().map(\.id) == ids.reversed())
    #expect(try await store.list(limit: 2, offset: 1).map(\.id) == [ids[3], ids[2]])
    #expect(try await store.list(source: "even").map(\.id) == [ids[4], ids[2], ids[0]])
  }

  @Test func keywordSearchRanksByBM25() async throws {
    let store = try MemoryStore.inMemory()
    let strong = try await store.remember("Rust, Rust, and more Rust.")
    let weak = try await store.remember(
      "I tried Rust once and went back to Swift for the rest of the year.")
    _ = try await store.remember("Unrelated note about gardening.")
    #expect(try await store.keywordSearch("rust", limit: 10) == [strong.id, weak.id])
  }

  @Test func keywordSearchStemsAndFoldsDiacritics() async throws {
    let store = try MemoryStore.inMemory()
    let memory = try await store.remember("She was running to the café.")
    #expect(try await store.keywordSearch("runs", limit: 10) == [memory.id])
    #expect(try await store.keywordSearch("CAFE", limit: 10) == [memory.id])
  }

  @Test func reopensFileStore() async throws {
    let directory = try TemporaryDirectory()
    let url = directory.url.appending(path: "memories.sqlite")
    let id: UUID
    do {
      let store = try MemoryStore(url: url)
      id = try await store.remember("Saved to disk.").id
    }
    let reopened = try MemoryStore(url: url)
    #expect(try await reopened.memory(id)?.text == "Saved to disk.")
  }
}

/// A directory that is deleted when the value is deinitialized.
final class TemporaryDirectory: Sendable {
  let url: URL

  init() throws {
    url = FileManager.default.temporaryDirectory
      .appending(path: "RecallTests-\(UUID().uuidString)", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
  }

  deinit {
    try? FileManager.default.removeItem(at: url)
  }
}
