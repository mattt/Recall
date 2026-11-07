import Foundation
import Testing

@testable import Recall

@Suite struct KeywordQueryTests {
  @Test func quotesEachTerm() {
    #expect(KeywordQuery.match(for: "tea or coffee") == #""tea" OR "or" OR "coffee""#)
  }

  @Test func removesDuplicateTermsIgnoringCase() {
    #expect(KeywordQuery.match(for: "Tea tea TEA") == #""Tea""#)
  }

  @Test func doublesQuotationMarks() {
    #expect(KeywordQuery.quoted(#"say "hi""#) == #""say ""hi""""#)
  }

  @Test(arguments: ["", "   ", "\"", "***", "()", "-", "^", ":", "\"\"", "🙂"])
  func textWithoutTermsHasNoQuery(_ text: String) {
    #expect(KeywordQuery.match(for: text) == nil)
  }

  /// Most of these strings fail if they are passed to `MATCH` unchanged.
  /// The others use FTS5 column filters, prefix queries, or initial-token queries.
  static let hostileQueries = [
    "\"unbalanced",
    "tea AND",
    "OR",
    "NOT coffee",
    "NEAR(tea coffee",
    "text: tea",
    "missing:column",
    "tea*",
    "*tea",
    "^tea",
    "-tea",
    "+tea",
    "(tea",
    "tea)",
    "{text} : tea",
    "tea's \"coffee\" (hot)",
    "AND OR NOT NEAR",
    "'; DROP TABLE memories; --",
  ]

  @Test(arguments: hostileQueries)
  func hostileQueryDoesNotFail(_ query: String) async throws {
    let store = try MemoryStore.inMemory()
    let memory = try await store.remember("tea and coffee near the text column")
    let results = try await store.keywordSearch(query, limit: 10)
    #expect(results.allSatisfy { $0 == memory.id })
    #expect(try await store.count() == 1)
  }

  /// Shows that the raw strings fail, so that the test above checks the escaping.
  @Test(arguments: ["\"unbalanced", "tea AND", "NOT coffee", "*tea", "-tea", "tea's", "(tea"])
  func rawQueryFails(_ query: String) throws {
    let database = try Database(url: nil)
    try Schema.migrate(database)
    try database.execute("INSERT INTO memories (id, text, created_at) VALUES ('x', 'tea', 0)")
    let statement = try database.prepare(
      "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?"
    )
    try statement.bind(.text(query))
    #expect(throws: MemoryStoreError.self) { try statement.step() }
  }

  @Test func operatorWordsMatchAsPlainWords() async throws {
    let store = try MemoryStore.inMemory()
    let memory = try await store.remember("Do NOT forget the NEAR field.")
    #expect(try await store.keywordSearch("not", limit: 10) == [memory.id])
    #expect(try await store.keywordSearch("NEAR", limit: 10) == [memory.id])
  }
}
