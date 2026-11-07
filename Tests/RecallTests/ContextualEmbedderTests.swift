#if canImport(NaturalLanguage)
  import Foundation
  import NaturalLanguage
  import Testing

  @testable import Recall

  /// Returns whether the English model's assets are already on the device.
  private func englishAssetsAreAvailable() -> Bool {
    NLContextualEmbedding(language: .english)?.hasAvailableAssets ?? false
  }

  /// These tests use the English model only if its assets are already on the device,
  /// so that a test run never starts a download.
  @Suite(
    .enabled(if: englishAssetsAreAvailable(), "The English model's assets aren't on this device."))
  struct ContextualEmbedderTests {
    @Test func vectorsHaveTheModelDimension() async throws {
      let embedder = try ContextualEmbedder(language: .english)
      let vector = try await embedder.embed("The user prefers tea to coffee.")
      #expect(vector.count == embedder.dimension)
      #expect(vector.allSatisfy { $0.isFinite })
    }

    @Test func identifierNamesModelAndRevision() throws {
      let embedder = try ContextualEmbedder(language: .english)
      #expect(embedder.identifier.contains(embedder.modelIdentifier))
      #expect(embedder.identifier.contains("/\(embedder.revision)/"))
    }

    @Test func embeddingIsDeterministic() async throws {
      let embedder = try ContextualEmbedder(language: .english)
      let first = try await embedder.embed("Same text twice.")
      let second = try await embedder.embed("Same text twice.")
      #expect(first == second)
    }

    @Test func relatedTextIsMoreSimilar() async throws {
      let embedder = try ContextualEmbedder(language: .english)
      func unit(_ text: String) async throws -> [Float] {
        try #require(Vector.normalized(try await embedder.embed(text)))
      }
      let query = try await unit("What does the user like to drink?")
      let related = try await unit("The user prefers tea to coffee.")
      let unrelated = try await unit("The car needs new tires before winter.")
      #expect(Vector.dot(query, related) > Vector.dot(query, unrelated))
    }

    @Test func storeRecallsWithoutSharedWords() async throws {
      let store = try MemoryStore.inMemory(embedder: try ContextualEmbedder(language: .english))
      let drink = try await store.remember("The user prefers tea to coffee.")
      try await store.remember("The car needs new tires before winter.")
      let results = try await store.recall(
        "favorite beverage",
        configuration: RecallConfiguration(recencyWeight: 0)
      )
      #expect(results.first?.id == drink.id)
      #expect(results.first?.keywordRank == nil)
    }

    @Test func unknownModelIdentifierThrows() {
      #expect(throws: ContextualEmbedder.Error.modelUnavailable("not-a-model")) {
        try ContextualEmbedder(modelIdentifier: "not-a-model")
      }
    }
  }
#endif
