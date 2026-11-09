import Foundation

/// A store of memories in a SQLite database.
///
/// A store is an actor,
/// so you can share one instance between tasks.
/// Each store has its own connection to the database.
/// Two stores can open the same file;
/// SQLite serializes their writes.
/// The stores can use different embedders:
/// the database keeps one vector for each memory from each embedder,
/// so one store doesn't replace the vectors of another.
///
/// If you supply an ``Embedder``,
/// the store also keeps a vector for each memory
/// and uses it for semantic search.
///
/// ## Cancellation
///
/// Each asynchronous method checks for cancellation before it starts.
/// `remember` and `update` check again inside their write transaction,
/// after they take the write lock and immediately before they commit.
/// If the task is canceled by then,
/// the method rolls back, throws `CancellationError`, and saves nothing.
/// The check before the commit is the decision point:
/// a call canceled after this check still saves its change and returns normally.
/// Re-embedding is the exception that ``reembed()`` describes.
public actor MemoryStore {
  let database: Database

  /// The embedder that the store uses for vector search, if any.
  public nonisolated let embedder: Embedder?

  /// The pass that is embedding stale memories, if one is running.
  private var reembedding: (task: Task<ReembeddingPass, any Error>, retriesFailures: Bool)?

  /// The number of calls that are waiting for ``reembedding``.
  private(set) var reembedWaiterCount = 0

  /// Whether a re-embedding task is running.
  var isReembedding: Bool { reembedding != nil }

  /// Opens or creates a store in the SQLite file at `url`.
  ///
  /// If the file has an older schema,
  /// the store migrates it to the current schema.
  ///
  /// - Parameters:
  ///   - url: A file URL for the database.
  ///   - embedder: The embedder for vector search,
  ///     or `nil` to use keyword search only.
  /// - Throws: ``MemoryStoreError/unsupportedSchemaVersion(_:)``
  ///   if a newer version of Recall created the file,
  ///   or ``MemoryStoreError/sqlite(code:message:)``
  ///   if SQLite can't open it.
  public init(url: URL, embedder: Embedder? = nil) throws {
    try self.init(database: Database(url: url), isFile: true, embedder: embedder)
  }

  /// Creates a store in a private in-memory database.
  ///
  /// The memories exist only as long as the store does.
  ///
  /// - Parameter embedder: The embedder for vector search,
  ///   or `nil` to use keyword search only.
  public static func inMemory(embedder: Embedder? = nil) throws -> MemoryStore {
    try MemoryStore(database: Database(url: nil), isFile: false, embedder: embedder)
  }

  init(
    database: Database,
    isFile: Bool,
    embedder: Embedder?
  ) throws(MemoryStoreError) {
    self.embedder = embedder
    if isFile {
      try database.enableWriteAheadLog()
    }
    try database.execute("PRAGMA foreign_keys = ON")
    try Schema.migrate(database)
    self.database = database
  }

  // MARK: - Writing

  /// Saves a new memory.
  ///
  /// - Parameters:
  ///   - text: The text to remember.
  ///   - source: The name of the client or tool that writes the memory.
  ///   - metadata: String key-value pairs to keep with the memory.
  ///   - createdAt: The creation date of the memory.
  ///     Supply a date to import memories from elsewhere.
  /// - Returns: The saved memory.
  /// - Throws: ``MemoryStoreError/emptyText`` if `text` contains only whitespace,
  ///   `CancellationError` if the task is canceled before the store saves the memory,
  ///   or any error from the embedder.
  ///   If the embedder fails, the store doesn't save the memory.
  @discardableResult
  public func remember(
    _ text: String,
    source: String? = nil,
    metadata: [String: String] = [:],
    createdAt: Date = Date()
  ) async throws -> Memory {
    try Task.checkCancellation()
    try Self.validate(text)
    let memory = Memory(text: text, createdAt: createdAt, source: source, metadata: metadata)
    let vector = try await embedding(for: text)
    try database.cancellableTransaction { () throws(MemoryStoreError) in
      let rowid = try insert(memory)
      if let vector { try saveEmbedding(vector, rowid: rowid) }
    }
    return memory
  }

  /// Saves changes to the text, source, or metadata of a memory.
  ///
  /// The identifier and creation date don't change.
  /// This method replaces all three fields with the values in `memory`,
  /// so the last update to finish overwrites changes that another call saved
  /// after you read `memory`.
  /// To change only the text, call ``update(_:text:)``.
  ///
  /// - Parameter memory: The memory with its new values.
  /// - Returns: The saved memory.
  /// - Throws: ``MemoryStoreError/notFound(_:)`` if the store has no memory with that identifier,
  ///   `CancellationError` if the task is canceled before the store saves the change,
  ///   or any error from the embedder.
  ///   If the embedder fails, the store doesn't save the change.
  @discardableResult
  public func update(_ memory: Memory) async throws -> Memory {
    try Task.checkCancellation()
    try Self.validate(memory.text)
    guard let existing = try self.memory(memory.id) else {
      throw MemoryStoreError.notFound(memory.id)
    }
    let vector = existing.text == memory.text ? nil : try await embedding(for: memory.text)
    try database.cancellableTransaction { () throws(MemoryStoreError) in
      let rowid = try save(memory)
      if let vector { try saveEmbedding(vector, rowid: rowid) }
    }
    return try self.memory(memory.id) ?? memory
  }

  /// Replaces the text of a memory.
  ///
  /// The source and metadata don't change,
  /// even if another call changes them while the embedder runs.
  ///
  /// - Returns: The saved memory.
  /// - Throws: ``MemoryStoreError/notFound(_:)`` if the store has no memory with that identifier,
  ///   `CancellationError` if the task is canceled before the store saves the change,
  ///   or any error from the embedder.
  ///   If the embedder fails, the store doesn't save the change.
  @discardableResult
  public func update(_ id: UUID, text: String) async throws -> Memory {
    try Task.checkCancellation()
    try Self.validate(text)
    guard let existing = try memory(id) else { throw MemoryStoreError.notFound(id) }
    if existing.text != text {
      let vector = try await embedding(for: text)
      try database.cancellableTransaction { () throws(MemoryStoreError) in
        try database.prepare("UPDATE memories SET text = ? WHERE id = ?")
          .bind(.text(text), .text(id.uuidString))
          .step()
        guard database.changes > 0 else { throw .notFound(id) }
        if let vector { try saveEmbedding(vector, rowid: rowid(of: id)) }
      }
    }
    guard let memory = try memory(id) else { throw MemoryStoreError.notFound(id) }
    return memory
  }

  /// Deletes a memory.
  ///
  /// - Returns: `true` if the store had the memory, or `false` if it didn't.
  @discardableResult
  public func forget(_ id: UUID) throws -> Bool {
    try database.prepare("DELETE FROM memories WHERE id = ?")
      .bind(.text(id.uuidString))
      .step()
    return database.changes > 0
  }

  // MARK: - Reading

  /// Returns the memory with an identifier, or `nil` if there isn't one.
  public func memory(_ id: UUID) throws -> Memory? {
    let statement = try database.prepare(
      "SELECT \(Self.columns) FROM memories m WHERE m.id = ?"
    )
    try statement.bind(.text(id.uuidString))
    guard try statement.step() else { return nil }
    return try Self.memory(from: statement)
  }

  /// Returns memories, newest first.
  ///
  /// - Parameters:
  ///   - source: If not `nil`, only memories with this source.
  ///   - limit: The maximum number of memories to return, or `nil` for all.
  ///   - offset: The number of memories to skip.
  public func list(source: String? = nil, limit: Int? = nil, offset: Int = 0) throws -> [Memory] {
    let statement = try database.prepare(
      """
      SELECT \(Self.columns) FROM memories m
      WHERE ?1 IS NULL OR m.source = ?1
      ORDER BY m.created_at DESC, m.rowid DESC
      LIMIT ?2 OFFSET ?3
      """
    )
    try statement.bind(.optionalText(source), .integer(limit ?? -1), .integer(max(offset, 0)))
    var memories: [Memory] = []
    while try statement.step() {
      memories.append(try Self.memory(from: statement))
    }
    return memories
  }

  /// The number of memories in the store.
  public func count() throws -> Int {
    let statement = try database.prepare("SELECT count(*) FROM memories")
    try statement.step()
    return statement.int(at: 0)
  }

  // MARK: - Keyword search

  /// Returns the identifiers of memories that match the words in `query`,
  /// best match first by BM25.
  func keywordSearch(
    _ query: String,
    limit: Int,
    ignoringStopWords: Bool = false
  ) throws -> [UUID] {
    guard limit > 0,
      let match = KeywordQuery.match(for: query, ignoringStopWords: ignoringStopWords)
    else { return [] }
    let statement = try database.prepare(
      """
      SELECT m.id FROM memories_fts
      JOIN memories m ON m.rowid = memories_fts.rowid
      WHERE memories_fts MATCH ?
      ORDER BY bm25(memories_fts), m.created_at DESC, m.rowid DESC
      LIMIT ?
      """
    )
    try statement.bind(.text(match), .integer(limit))
    var ids: [UUID] = []
    while try statement.step() {
      if let id = statement.text(at: 0).flatMap(UUID.init(uuidString:)) {
        ids.append(id)
      }
    }
    return ids
  }

  // MARK: - Vectors

  /// Embeds each memory that has no vector from the current embedder,
  /// and returns the number of vectors it saved.
  ///
  /// Vectors from other embedders stay in the database.
  ///
  /// Recall embeds stale memories before a vector search,
  /// so you don't need to call this method.
  /// Call it after you open a store with a new embedder
  /// to do the work before the first recall,
  /// or to try again the memories that recall skips.
  ///
  /// ## Failures
  ///
  /// This method continues after the embedder fails for one memory,
  /// and it throws ``ReembeddingError`` after it embeds every memory that it can.
  /// Unlike a recall, it also tries the memories that failed earlier.
  ///
  /// An error doesn't say whether one text or the whole embedder caused it.
  /// The store remembers a failure only after the embedder
  /// returns a usable vector for a different text,
  /// so a memory that fails during an outage stays stale and isn't remembered.
  /// After eight consecutive failures with no such vector,
  /// the method stops and throws the embedder's error.
  /// A zero vector or ``ContextualEmbedder/Error/noVectors``
  /// shows that the cause is the text, so it doesn't count toward that limit.
  ///
  /// If another call is already embedding memories,
  /// this call waits for it and returns the same count.
  /// If a recall started that work,
  /// this call waits for it and then tries the memories that failed earlier.
  ///
  /// The work runs in a task that all waiting calls share.
  /// If the task that calls this method is canceled,
  /// this call stops waiting and throws `CancellationError`,
  /// but the shared work continues for the other calls.
  /// The store saves each vector as soon as the embedder returns it,
  /// so if the method throws,
  /// the vectors that the work saved before the error remain,
  /// and a later call embeds only the memories that are still stale.
  ///
  /// - Returns: The number of vectors that the store saved.
  /// - Throws: ``ReembeddingError`` if some memories couldn't be embedded,
  ///   the embedder's error if the embedder keeps failing,
  ///   ``MemoryStoreError/dimensionMismatch(embedder:expected:actual:)``
  ///   if a vector doesn't match the vectors that the store saved from the embedder,
  ///   or `CancellationError` if the task is canceled.
  @discardableResult
  public func reembed() async throws -> Int {
    try Task.checkCancellation()
    guard let embedder else { return 0 }
    var embedded = 0
    // A pass that a recall started skips the memories that failed earlier,
    // so this call can't share it. It can start its own pass when that one ends.
    while let running = reembedding, !running.retriesFailures {
      embedded += try await wait(for: running.task).embedded
    }
    let pass = try await sharedPass(retryingFailures: true, with: embedder)
    embedded += pass.embedded
    guard pass.failures.isEmpty else {
      throw ReembeddingError(embeddedCount: embedded, failures: pass.failures)
    }
    return embedded
  }

  /// Joins the running pass, or starts one if none is running,
  /// and returns the result of the pass.
  private func sharedPass(
    retryingFailures: Bool,
    with embedder: Embedder
  ) async throws -> ReembeddingPass {
    let task: Task<ReembeddingPass, any Error>
    if let reembedding {
      task = reembedding.task
    } else {
      // The task clears itself when it ends,
      // so a waiter that stops waiting doesn't let a second task start.
      task = Task {
        defer { reembedding = nil }
        return try await embedStaleMemories(with: embedder, retryingFailures: retryingFailures)
      }
      reembedding = (task, retryingFailures)
    }
    return try await wait(for: task)
  }

  /// Returns the result of a pass, and counts this call as a waiter until it ends.
  private func wait(for task: Task<ReembeddingPass, any Error>) async throws -> ReembeddingPass {
    reembedWaiterCount += 1
    defer { reembedWaiterCount -= 1 }
    return try await Self.value(of: task)
  }

  /// Returns the value of `task`,
  /// or throws `CancellationError` as soon as the current task is canceled.
  ///
  /// Unlike `Task.value`, this method stops waiting on cancellation.
  /// It doesn't cancel `task`.
  private static func value<T: Sendable>(of task: Task<T, any Error>) async throws -> T {
    let waiter = Waiter<T>()
    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        waiter.wait(continuation)
        Task { waiter.finish(await task.result) }
      }
    } onCancel: {
      waiter.finish(.failure(CancellationError()))
    }
  }

  /// The number of memories that have no vector from the current embedder.
  ///
  /// The count includes memories that the store couldn't embed.
  public func staleEmbeddingCount() throws -> Int {
    guard let embedder else { return 0 }
    let statement = try database.prepare(
      """
      SELECT count(*) FROM memories m
      WHERE NOT EXISTS (
        SELECT 1 FROM embeddings e WHERE e.memory_rowid = m.rowid AND e.embedder = ?
      )
      """
    )
    try statement.bind(.text(embedder.identifier))
    try statement.step()
    return statement.int(at: 0)
  }

  /// The number of consecutive failures with no sign that the text caused them
  /// after which a pass stops.
  private static let maximumConsecutiveUnexplainedFailures = 8

  private func embedStaleMemories(
    with embedder: Embedder,
    retryingFailures: Bool
  ) async throws -> ReembeddingPass {
    let batchSize = 64
    let marker = Self.failureMarker(for: embedder)
    var pass = ReembeddingPass()
    var unexplainedFailures = 0
    // Skip rows that changed during embedding so the loop ends;
    // a later call tries them again.
    var lastRowid = 0
    while true {
      // A pass that retries failures looks only for the embedder's vector,
      // so it doesn't exclude a memory that has a failure marker.
      let statement = try database.prepare(
        """
        SELECT m.rowid, m.id, m.text FROM memories m
        WHERE NOT EXISTS (
          SELECT 1 FROM embeddings e WHERE e.memory_rowid = m.rowid AND e.embedder IN (?1, ?4)
        ) AND m.rowid > ?2
        ORDER BY m.rowid
        LIMIT ?3
        """
      )
      try statement.bind(
        .text(embedder.identifier),
        .integer(lastRowid),
        .integer(batchSize),
        .text(retryingFailures ? embedder.identifier : marker)
      )
      var batch: [StaleRow] = []
      while try statement.step() {
        batch.append(
          StaleRow(
            rowid: statement.int(at: 0),
            id: statement.text(at: 1) ?? "",
            text: statement.text(at: 2) ?? ""
          )
        )
      }
      guard !batch.isEmpty else { return pass }
      for row in batch {
        lastRowid = row.rowid
        let vector: [Float]
        do {
          vector = try await normalizedEmbedding(for: row.text, with: embedder)
        } catch {
          if error is CancellationError { throw error }
          // A failure for text that another connection replaced says nothing about the memory.
          guard try isCurrent(row) else { continue }
          guard let id = UUID(uuidString: row.id) else {
            throw MemoryStoreError.sqlite(
              code: 0, message: "A memory has an identifier that isn't a UUID.")
          }
          pass.failures.append(ReembeddingError.Failure(id: id, error: error))
          pass.unconfirmed.append(row)
          if !Self.isCausedByText(error) {
            unexplainedFailures += 1
            if unexplainedFailures >= Self.maximumConsecutiveUnexplainedFailures { throw error }
          }
          continue
        }
        // Another task or connection can change or delete the memory while the embedder runs.
        // Check it and save the vector in one write transaction,
        // so that no other write can come between them.
        // No caller can cancel this shared task,
        // so the transaction doesn't check for cancellation.
        let saved = try database.transaction { () throws(MemoryStoreError) -> Bool in
          guard try isCurrent(row) else { return false }
          try saveEmbedding(vector, rowid: row.rowid)
          try removeFailureMarker(rowid: row.rowid, marker: marker)
          return true
        }
        if saved { pass.embedded += 1 }
        // The embedder works, so the failures before this vector came from their texts.
        unexplainedFailures = 0
        try rememberFailures(of: pass.unconfirmed, marker: marker)
        pass.unconfirmed = []
      }
    }
  }

  /// The result of one pass over the stale memories.
  private struct ReembeddingPass: Sendable {
    /// The number of vectors that the pass saved.
    var embedded = 0

    /// The memories that the pass couldn't embed.
    var failures: [ReembeddingError.Failure] = []

    /// The memories that failed after the last usable vector from the embedder.
    ///
    /// The store doesn't remember these failures yet,
    /// because the embedder might be unavailable.
    var unconfirmed: [StaleRow] = []
  }

  /// A memory that needs a vector from the current embedder.
  private struct StaleRow: Sendable {
    var rowid: Int
    var id: String
    var text: String
  }

  /// Returns whether an embedding error shows that the text caused it.
  private static func isCausedByText(_ error: any Error) -> Bool {
    if case MemoryStoreError.invalidEmbedding = error { return true }
    return (error as? any TextSpecificEmbeddingError)?.isCausedByText ?? false
  }

  /// Returns the name under which the store marks memories that `embedder` can't embed.
  ///
  /// A marker is a row in the `embeddings` table with this name, no dimension, and no vector.
  /// The trigger and foreign key that delete a memory's vectors also delete its markers,
  /// so a change to the text retries the memory,
  /// and another embedder has a different marker.
  /// The separate name keeps markers out of every query for an embedder's vectors,
  /// including the queries in earlier versions of Recall.
  private static func failureMarker(for embedder: Embedder) -> String {
    "recall.embedding-failed/\(embedder.identifier)"
  }

  /// Marks memories as ones that the embedder can't embed,
  /// so that a recall doesn't try them again.
  private func rememberFailures(
    of rows: [StaleRow],
    marker: String
  ) throws(MemoryStoreError) {
    guard !rows.isEmpty else { return }
    try database.transaction { () throws(MemoryStoreError) in
      for row in rows {
        guard try isCurrent(row) else { continue }
        try database.prepare(
          """
          INSERT OR IGNORE INTO embeddings (memory_rowid, embedder, dimension, vector)
          VALUES (?, ?, 0, x'')
          """
        )
        .bind(.integer(row.rowid), .text(marker))
        .step()
      }
    }
  }

  /// Deletes the mark that the embedder couldn't embed a memory.
  private func removeFailureMarker(rowid: Int, marker: String) throws(MemoryStoreError) {
    try database.prepare("DELETE FROM embeddings WHERE memory_rowid = ? AND embedder = ?")
      .bind(.integer(rowid), .text(marker))
      .step()
  }

  /// Returns whether the row still holds the memory with this identifier and text.
  private func isCurrent(_ row: StaleRow) throws(MemoryStoreError) -> Bool {
    let statement = try database.prepare("SELECT id, text FROM memories WHERE rowid = ?")
    try statement.bind(.integer(row.rowid))
    guard try statement.step() else { return false }
    return statement.text(at: 0) == row.id && statement.text(at: 1) == row.text
  }

  /// Returns the identifiers of the memories whose vectors are most similar to the vector of `query`,
  /// most similar first, with their cosine similarities.
  func vectorSearch(
    _ query: String,
    limit: Int,
    minimumSimilarity: Float? = nil
  ) async throws -> [(id: UUID, similarity: Float)] {
    guard limit > 0, let vector = try await queryVector(for: query) else { return [] }
    return try vectorSearch(vector, limit: limit, minimumSimilarity: minimumSimilarity)
  }

  /// Embeds any stale memories that haven't failed before,
  /// then returns the unit vector of `query`,
  /// or `nil` if the store has no embedder
  /// or the query's vector can't be normalized.
  func queryVector(for query: String) async throws -> [Float]? {
    guard let embedder else { return nil }
    try Task.checkCancellation()
    let pass = try await sharedPass(retryingFailures: false, with: embedder)
    // A query vector that can't be normalized, such as the vector of punctuation,
    // matches nothing; it isn't an error.
    guard let vector = Vector.normalized(try await embedder.embed(query, for: .query)) else {
      return nil
    }
    // The embedder works, so the failures at the end of the pass came from their texts.
    try rememberFailures(of: pass.unconfirmed, marker: Self.failureMarker(for: embedder))
    return vector
  }

  /// Returns the identifiers of the memories whose vectors are most similar to a unit vector,
  /// most similar first, with their cosine similarities.
  func vectorSearch(
    _ vector: [Float],
    limit: Int,
    minimumSimilarity: Float?
  ) throws(MemoryStoreError) -> [(id: UUID, similarity: Float)] {
    guard limit > 0, let embedder else { return [] }
    let results = try nearest(to: vector, embedder: embedder.identifier, limit: limit)
    guard let minimumSimilarity else { return results }
    return results.filter { $0.similarity >= minimumSimilarity }
  }

  /// Returns memory identifiers and cosine similarities, most similar first.
  private func nearest(
    to query: [Float],
    embedder: String,
    limit: Int
  ) throws(MemoryStoreError) -> [(id: UUID, similarity: Float)] {
    // A query vector can't be compared with vectors of another dimension.
    // The store keeps one dimension for each embedder,
    // so a different dimension means that the embedder broke its contract.
    let dimensions = try database.prepare(
      "SELECT DISTINCT dimension FROM embeddings WHERE embedder = ?"
    )
    try dimensions.bind(.text(embedder))
    while try dimensions.step() {
      let dimension = dimensions.int(at: 0)
      guard dimension == query.count else {
        throw .dimensionMismatch(embedder: embedder, expected: dimension, actual: query.count)
      }
    }
    let statement = try database.prepare(
      """
      SELECT m.id, e.vector FROM embeddings e
      JOIN memories m ON m.rowid = e.memory_rowid
      WHERE e.embedder = ? AND e.dimension = ?
      """
    )
    try statement.bind(.text(embedder), .integer(query.count))
    var scored: [(id: UUID, similarity: Float)] = []
    while try statement.step() {
      guard let id = statement.text(at: 0).flatMap(UUID.init(uuidString:)),
        let vector = Vector.decode(statement.data(at: 1), dimension: query.count)
      else { continue }
      scored.append((id, Vector.dot(query, vector)))
    }
    scored.sort { $0.similarity > $1.similarity }
    return Array(scored.prefix(limit))
  }

  /// Returns the unit vector of a memory's text from the store's embedder,
  /// or `nil` if the store has no embedder.
  private func embedding(for text: String) async throws -> [Float]? {
    guard let embedder else { return nil }
    return try await normalizedEmbedding(for: text, with: embedder)
  }

  /// Returns the unit vector of a memory's text, which the store saves.
  private func normalizedEmbedding(
    for text: String,
    with embedder: Embedder
  ) async throws -> [Float] {
    guard let vector = Vector.normalized(try await embedder.embed(text, for: .document)) else {
      throw MemoryStoreError.invalidEmbedding(embedder: embedder.identifier)
    }
    return vector
  }

  /// Saves the vector of a memory from the current embedder.
  ///
  /// The vector replaces an earlier vector from the same embedder
  /// and leaves vectors from other embedders in place.
  private func saveEmbedding(_ vector: [Float], rowid: Int) throws(MemoryStoreError) {
    guard let embedder else { return }
    let identifier = embedder.identifier
    // Vectors of different dimensions can't be compared,
    // so all vectors from one embedder must agree.
    let existing = try database.prepare(
      "SELECT dimension FROM embeddings WHERE embedder = ? AND memory_rowid != ? LIMIT 1"
    )
    try existing.bind(.text(identifier), .integer(rowid))
    if try existing.step(), existing.int(at: 0) != vector.count {
      throw .dimensionMismatch(
        embedder: identifier, expected: existing.int(at: 0), actual: vector.count)
    }
    try database.prepare(
      """
      INSERT OR REPLACE INTO embeddings (memory_rowid, embedder, dimension, vector)
      VALUES (?, ?, ?, ?)
      """
    )
    .bind(.integer(rowid), .text(identifier), .integer(vector.count), .blob(Vector.encode(vector)))
    .step()
  }

  // MARK: - Rows

  /// The memory columns, in the order that ``memory(from:)`` reads them.
  static let columns = "m.id, m.text, m.created_at, m.source, m.metadata"

  static func memory(from statement: Statement) throws(MemoryStoreError) -> Memory {
    guard let id = statement.text(at: 0).flatMap(UUID.init(uuidString:)) else {
      throw .sqlite(code: 0, message: "A memory has an identifier that isn't a UUID.")
    }
    var metadata: [String: String] = [:]
    if let json = statement.text(at: 4) {
      do {
        metadata = try JSONDecoder().decode([String: String].self, from: Data(json.utf8))
      } catch {
        throw .invalidMetadata(id)
      }
    }
    return Memory(
      id: id,
      text: statement.text(at: 1) ?? "",
      createdAt: Date(timeIntervalSinceReferenceDate: statement.double(at: 2)),
      source: statement.text(at: 3),
      metadata: metadata
    )
  }

  static func encode(_ metadata: [String: String]) -> SQLValue {
    guard !metadata.isEmpty else { return .null }
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    // Encoding a dictionary of strings can't fail.
    let data = (try? encoder.encode(metadata)) ?? Data("{}".utf8)
    return .text(String(decoding: data, as: UTF8.self))
  }

  static func validate(_ text: String) throws(MemoryStoreError) {
    guard text.contains(where: { !$0.isWhitespace }) else { throw .emptyText }
  }

  /// Inserts a new memory row and returns its row identifier.
  @discardableResult
  func insert(_ memory: Memory) throws(MemoryStoreError) -> Int {
    try database.prepare(
      """
      INSERT INTO memories (id, text, created_at, source, metadata)
      VALUES (?, ?, ?, ?, ?)
      """
    )
    .bind(
      .text(memory.id.uuidString),
      .text(memory.text),
      .real(memory.createdAt.timeIntervalSinceReferenceDate),
      .optionalText(memory.source),
      Self.encode(memory.metadata)
    )
    .step()
    return try rowid(of: memory.id)
  }

  /// Saves the mutable fields of an existing memory and returns its row identifier.
  @discardableResult
  func save(_ memory: Memory) throws(MemoryStoreError) -> Int {
    try database.prepare(
      "UPDATE memories SET text = ?, source = ?, metadata = ? WHERE id = ?"
    )
    .bind(
      .text(memory.text),
      .optionalText(memory.source),
      Self.encode(memory.metadata),
      .text(memory.id.uuidString)
    )
    .step()
    guard database.changes > 0 else { throw .notFound(memory.id) }
    return try rowid(of: memory.id)
  }

  func rowid(of id: UUID) throws(MemoryStoreError) -> Int {
    let statement = try database.prepare("SELECT rowid FROM memories WHERE id = ?")
    try statement.bind(.text(id.uuidString))
    guard try statement.step() else { throw .notFound(id) }
    return statement.int(at: 0)
  }
}

/// A continuation that resumes once,
/// with the first result that it receives.
private final class Waiter<T: Sendable>: @unchecked Sendable {
  private let lock = NSLock()
  private var continuation: CheckedContinuation<T, any Error>?
  private var result: Result<T, any Error>?

  /// Resumes `continuation` with a result that has arrived,
  /// or keeps it for ``finish(_:)``.
  func wait(_ continuation: CheckedContinuation<T, any Error>) {
    let result = lock.withLock {
      if self.result == nil { self.continuation = continuation }
      return self.result
    }
    if let result { continuation.resume(with: result) }
  }

  /// Records `result` and resumes the waiting continuation,
  /// unless a result has already arrived.
  func finish(_ result: Result<T, any Error>) {
    let continuation = lock.withLock { () -> CheckedContinuation<T, any Error>? in
      guard self.result == nil else { return nil }
      self.result = result
      defer { self.continuation = nil }
      return self.continuation
    }
    continuation?.resume(with: result)
  }
}
