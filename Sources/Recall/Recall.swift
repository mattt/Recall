import Foundation

/// Settings that control how ``MemoryStore/recall(_:limit:configuration:)`` ranks memories.
///
/// Recall builds up to three ranked lists of memories
/// and merges them with weighted reciprocal rank fusion.
/// A memory at rank `r` in a list with weight `w`
/// gets `w / (k + r)` from that list,
/// where ranks start at 1.
/// A memory's fused score is the sum of these contributions.
///
/// The lists are:
///
/// - term Keyword: Memories that contain words from the query, ordered by BM25.
///   By default, the list ignores common English words, such as "the" and "my".
/// - term Vector: Memories ordered by the cosine similarity of their vectors to the query's vector.
///   The store makes this list only if it has an ``Embedder``.
/// - term Recency: The memories from the other two lists, newest first.
///   This list only reorders the memories that the query found;
///   it doesn't add memories.
///
/// A list with a weight of zero isn't made,
/// and its rank in each result is `nil`.
/// The recency list has a weight of zero by default.
public struct RecallConfiguration: Hashable, Sendable {
  /// The constant that controls how fast a list's contribution falls with rank.
  ///
  /// A larger value makes the difference between adjacent ranks smaller.
  /// The default is 60.
  public var k: Double

  /// The weight of the keyword list.
  public var keywordWeight: Double

  /// The weight of the vector list.
  public var vectorWeight: Double

  /// The weight of the recency list.
  ///
  /// The default is zero, so recall doesn't make the list.
  /// The list gives points to every candidate by its date,
  /// so a recent memory that matches the query weakly
  /// can move above an older memory that matches it well.
  public var recencyWeight: Double

  /// The maximum number of memories that the keyword and vector lists each contain.
  public var candidateLimit: Int

  /// Whether the keyword list ignores common English words, such as "the", "my", and "when".
  ///
  /// The keyword list matches a memory that contains any word of the query,
  /// so these words can make a question match many memories,
  /// and those weak matches can push a better match from the vector list down.
  /// If every word of the query is a common word,
  /// the keyword list uses them all.
  ///
  /// The stop word list is English only, and the store does not check the language.
  /// Some stop words have meaning in other languages.
  /// For example, "by" means "city" in Danish.
  /// Set this to `false` for memories or queries in other languages or in mixed languages.
  /// The default is `true`.
  public var ignoresStopWords: Bool

  /// The minimum cosine similarity that a memory needs to be in the vector list,
  /// or `nil` to allow every memory with a vector.
  ///
  /// A value must be finite and in the range `-1...1`.
  public var minimumSimilarity: Float?

  /// Creates a configuration.
  ///
  /// With the default weights,
  /// the keyword and vector lists count equally,
  /// and creation dates don't count.
  public init(
    k: Double = 60,
    keywordWeight: Double = 1,
    vectorWeight: Double = 1,
    recencyWeight: Double = 0,
    candidateLimit: Int = 50,
    ignoresStopWords: Bool = true,
    minimumSimilarity: Float? = nil
  ) {
    self.k = k
    self.keywordWeight = keywordWeight
    self.vectorWeight = vectorWeight
    self.recencyWeight = recencyWeight
    self.candidateLimit = candidateLimit
    self.ignoresStopWords = ignoresStopWords
    self.minimumSimilarity = minimumSimilarity
  }

  /// Throws if a value can't produce meaningful scores.
  func validate() throws(MemoryStoreError) {
    guard k.isFinite, k >= 0 else { throw .invalidConfiguration(.k) }
    let weights = [keywordWeight, vectorWeight, recencyWeight]
    for weight in weights {
      guard weight.isFinite, weight >= 0 else { throw .invalidConfiguration(.weight) }
    }
    // The highest possible score puts a memory at rank 1 in every list.
    // Floating-point addition and division are monotonic,
    // so if this sum is finite, every fused score is finite.
    let highestScore = weights.reduce(0) { $0 + $1 / (k + 1) }
    guard highestScore.isFinite else { throw .invalidConfiguration(.scoreOverflow) }
    guard candidateLimit > 0 else { throw .invalidConfiguration(.candidateLimit) }
    if let minimumSimilarity {
      guard minimumSimilarity.isFinite, (-1...1).contains(minimumSimilarity) else {
        throw .invalidConfiguration(.minimumSimilarity)
      }
    }
  }

  /// A reason that a configuration isn't valid.
  public enum Problem: Hashable, Sendable, CustomStringConvertible {
    /// `k` isn't finite, or it is negative.
    case k

    /// A weight isn't finite, or it is negative.
    case weight

    /// The weights are so large that a fused score would overflow to infinity.
    case scoreOverflow

    /// `candidateLimit` isn't greater than zero.
    case candidateLimit

    /// `minimumSimilarity` isn't finite, or it is outside the range `-1...1`.
    case minimumSimilarity

    public var description: String {
      switch self {
      case .k: "k must be a finite number that isn't negative."
      case .weight: "Each weight must be a finite number that isn't negative."
      case .scoreOverflow: "The weights are too large: a fused score would be infinite."
      case .candidateLimit: "candidateLimit must be greater than zero."
      case .minimumSimilarity: "minimumSimilarity must be a finite number from -1 to 1."
      }
    }
  }
}

/// A memory that recall found, with information that explains its rank.
public struct RecallResult: Identifiable, Hashable, Sendable {
  /// The memory.
  public var memory: Memory

  /// The fused score: the sum of `weight / (k + rank)` over the lists that contain the memory.
  ///
  /// Results are sorted by this score, highest first.
  public var score: Double

  /// The rank of the memory in the keyword list, starting at 1,
  /// or `nil` if the list doesn't contain it.
  public var keywordRank: Int?

  /// The rank of the memory in the vector list, starting at 1,
  /// or `nil` if the list doesn't contain it.
  public var vectorRank: Int?

  /// The rank of the memory in the recency list, starting at 1,
  /// or `nil` if the list doesn't contain it.
  public var recencyRank: Int?

  /// The cosine similarity of the memory's vector to the query's vector,
  /// or `nil` if the vector list doesn't contain the memory.
  public var similarity: Float?

  /// The identifier of the memory.
  public var id: UUID { memory.id }

  /// Creates a result.
  public init(
    memory: Memory,
    score: Double,
    keywordRank: Int? = nil,
    vectorRank: Int? = nil,
    recencyRank: Int? = nil,
    similarity: Float? = nil
  ) {
    self.memory = memory
    self.score = score
    self.keywordRank = keywordRank
    self.vectorRank = vectorRank
    self.recencyRank = recencyRank
    self.similarity = similarity
  }
}

/// Weighted reciprocal rank fusion.
enum Fusion {
  /// Returns the fused score of each identifier in `lists`.
  ///
  /// The identifier at index `i` of a list has rank `i + 1`
  /// and gets `weight / (k + i + 1)` from that list.
  static func scores<ID: Hashable>(
    _ lists: [(weight: Double, ids: [ID])],
    k: Double
  ) -> [ID: Double] {
    var scores: [ID: Double] = [:]
    for list in lists {
      for (index, id) in list.ids.enumerated() {
        scores[id, default: 0] += list.weight / (k + Double(index + 1))
      }
    }
    return scores
  }

  /// Returns the rank, starting at 1, of each identifier in `ids`.
  static func ranks<ID: Hashable>(_ ids: [ID]) -> [ID: Int] {
    var ranks: [ID: Int] = [:]
    for (index, id) in ids.enumerated() where ranks[id] == nil {
      ranks[id] = index + 1
    }
    return ranks
  }
}

extension MemoryStore {
  /// Finds the memories that best match a query.
  ///
  /// The store ranks memories by keywords,
  /// by vector similarity if it has an embedder,
  /// and by recency if the configuration gives recency a weight,
  /// then merges the lists as ``RecallConfiguration`` describes.
  /// Before a vector search,
  /// the store embeds any memories that have no vector from the current embedder.
  ///
  /// - Parameters:
  ///   - query: Text that describes what to recall.
  ///   - limit: The maximum number of results.
  ///   - configuration: The settings for ranking.
  /// - Returns: Results with the highest score first.
  ///   Equal scores put newer memories first.
  public func recall(
    _ query: String,
    limit: Int = 10,
    configuration: RecallConfiguration = RecallConfiguration()
  ) async throws -> [RecallResult] {
    try Task.checkCancellation()
    try configuration.validate()
    guard limit > 0 else { return [] }

    // Finish the work that suspends before reading any rankings.
    // Another task can change memories while the embedder runs,
    // so the rankings and the memories must come from one later snapshot.
    let queryVector =
      configuration.vectorWeight > 0 ? try await queryVector(for: query) : nil
    let (keyword, vector, memories) = try database.read {
      let keyword =
        configuration.keywordWeight > 0
        ? try keywordSearch(
          query,
          limit: configuration.candidateLimit,
          ignoringStopWords: configuration.ignoresStopWords
        ) : []
      let vector =
        try queryVector.map {
          try vectorSearch(
            $0,
            limit: configuration.candidateLimit,
            minimumSimilarity: configuration.minimumSimilarity
          )
        } ?? []
      // Load each candidate once.
      var memories: [UUID: Memory] = [:]
      for id in keyword + vector.map(\.id) where memories[id] == nil {
        memories[id] = try memory(id)
      }
      return (keyword, vector, memories)
    }

    let recency =
      configuration.recencyWeight > 0
      ? memories.values.sorted(by: Self.newerFirst).map(\.id) : []

    let keywordRanks = Fusion.ranks(keyword)
    let vectorRanks = Fusion.ranks(vector.map(\.id))
    let recencyRanks = Fusion.ranks(recency)
    let similarities = Dictionary(vector.map { ($0.id, $0.similarity) }) { first, _ in first }
    let scores = Fusion.scores(
      [
        (configuration.keywordWeight, keyword),
        (configuration.vectorWeight, vector.map(\.id)),
        (configuration.recencyWeight, recency),
      ],
      k: configuration.k
    )

    let results = memories.values.map { memory in
      RecallResult(
        memory: memory,
        score: scores[memory.id] ?? 0,
        keywordRank: keywordRanks[memory.id],
        vectorRank: vectorRanks[memory.id],
        recencyRank: recencyRanks[memory.id],
        similarity: similarities[memory.id]
      )
    }
    return Array(
      results.sorted { left, right in
        left.score != right.score
          ? left.score > right.score : Self.newerFirst(left.memory, right.memory)
      }
      .prefix(limit)
    )
  }

  /// Orders memories newest first,
  /// then by identifier when the dates are equal to keep the order stable.
  static func newerFirst(_ left: Memory, _ right: Memory) -> Bool {
    left.createdAt != right.createdAt
      ? left.createdAt > right.createdAt : left.id.uuidString < right.id.uuidString
  }
}
