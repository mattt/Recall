#if canImport(NaturalLanguage)
  import Foundation
  import NaturalLanguage

  /// An embedder that uses a contextual embedding model from the NaturalLanguage framework.
  ///
  /// The model returns one vector for each subword token of the text.
  /// This embedder takes the mean of those vectors,
  /// so each text has one vector of ``dimension`` elements.
  /// The model reads at most ``maximumSequenceLength`` tokens
  /// and ignores the rest of a longer text.
  ///
  /// The system downloads model assets on demand.
  /// Before the first embedding,
  /// this embedder checks `hasAvailableAssets`,
  /// requests the assets if they aren't on the device,
  /// and loads the model.
  /// Call ``prepare()`` to do this work before you need a vector.
  ///
  /// ```swift
  /// let embedder = try ContextualEmbedder(language: .english)
  /// try await embedder.prepare()
  /// let store = try MemoryStore(url: url, embedder: embedder)
  /// ```
  @available(macOS 14, iOS 17, tvOS 17, watchOS 10, visionOS 1, *)
  public actor ContextualEmbedder: Embedder {
    /// An error that a contextual embedder throws.
    public enum Error: Swift.Error, Hashable, Sendable {
      /// The system has no contextual embedding model for the language, script, or identifier.
      case modelUnavailable(String)

      /// The model assets aren't on the device and the system couldn't download them.
      case assetsUnavailable(modelIdentifier: String)

      /// The model didn't return a vector for the text.
      case noVectors
    }

    /// Identifies the model and its revision,
    /// so that a store embeds its memories again when either changes.
    public nonisolated let identifier: String

    /// The identifier of the NaturalLanguage model.
    public nonisolated let modelIdentifier: String

    /// The revision of the model.
    public nonisolated let revision: Int

    /// The number of elements in each vector.
    public nonisolated let dimension: Int

    /// The maximum number of tokens that the model reads from one text.
    public nonisolated let maximumSequenceLength: Int

    /// The language that the embedder passes to the model,
    /// or `nil` to let the model detect the language of each text.
    public nonisolated let language: NLLanguage?

    private let embedding: NLContextualEmbedding
    private var isLoaded = false

    /// Creates an embedder with the most recent model for a language.
    ///
    /// - Parameter language: The language of the memories and queries.
    ///   The embedder also passes this language to the model with each text.
    public init(language: NLLanguage) throws {
      guard let embedding = NLContextualEmbedding(language: language) else {
        throw Error.modelUnavailable(language.rawValue)
      }
      self.init(embedding: embedding, language: language)
    }

    /// Creates an embedder with the most recent model for a writing system.
    ///
    /// The model detects the language of each text.
    public init(script: NLScript) throws {
      guard let embedding = NLContextualEmbedding(script: script) else {
        throw Error.modelUnavailable(script.rawValue)
      }
      self.init(embedding: embedding, language: nil)
    }

    /// Creates an embedder with a specific model.
    ///
    /// Use this initializer to keep the same model across system updates.
    ///
    /// - Parameters:
    ///   - modelIdentifier: The ``modelIdentifier`` of a model.
    ///   - language: The language to pass to the model with each text,
    ///     or `nil` to let the model detect it.
    public init(modelIdentifier: String, language: NLLanguage? = nil) throws {
      guard let embedding = NLContextualEmbedding(modelIdentifier: modelIdentifier) else {
        throw Error.modelUnavailable(modelIdentifier)
      }
      self.init(embedding: embedding, language: language)
    }

    private init(embedding: NLContextualEmbedding, language: NLLanguage?) {
      self.embedding = embedding
      self.language = language
      modelIdentifier = embedding.modelIdentifier
      revision = embedding.revision
      dimension = embedding.dimension
      maximumSequenceLength = embedding.maximumSequenceLength
      identifier = "NLContextualEmbedding/\(embedding.modelIdentifier)/\(embedding.revision)/mean"
    }

    /// Makes sure that the model assets are on the device, and loads the model.
    ///
    /// If the assets aren't on the device,
    /// this method asks the system to download them,
    /// which can take some time and needs a network connection.
    ///
    /// - Throws: ``Error/assetsUnavailable(modelIdentifier:)``
    ///   if the system can't provide the assets,
    ///   or an error from the framework if the model doesn't load.
    public func prepare() async throws {
      guard !isLoaded else { return }
      if !embedding.hasAvailableAssets {
        let result = try await requestAssets()
        guard result == .available else {
          throw Error.assetsUnavailable(modelIdentifier: modelIdentifier)
        }
      }
      try embedding.load()
      isLoaded = true
    }

    /// Releases the model's memory.
    ///
    /// The next embedding loads the model again.
    public func unload() {
      guard isLoaded else { return }
      embedding.unload()
      isLoaded = false
    }

    public func embed(_ text: String) async throws -> [Float] {
      try await prepare()
      let result = try embedding.embeddingResult(for: text, language: language)
      var sum = [Double](repeating: 0, count: dimension)
      var count = 0
      result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { vector, _ in
        guard vector.count == sum.count else { return true }
        for index in sum.indices {
          sum[index] += vector[index]
        }
        count += 1
        return true
      }
      guard count > 0 else { throw Error.noVectors }
      return sum.map { Float($0 / Double(count)) }
    }

    private func requestAssets() async throws -> NLContextualEmbedding.AssetsResult {
      try await withCheckedThrowingContinuation { continuation in
        embedding.requestAssets { result, error in
          if let error {
            continuation.resume(throwing: error)
          } else {
            continuation.resume(returning: result)
          }
        }
      }
    }
  }
#endif
