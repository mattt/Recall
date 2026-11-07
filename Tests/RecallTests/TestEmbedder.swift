import Foundation
import Recall

/// A test type that makes vectors and records how the store calls it.
///
/// The store takes an ``Embedder`` value;
/// ``embedder`` wraps the type in one,
/// and the extension on `MemoryStore` below passes it on.
protocol EmbedderSource: Sendable {
  var identifier: String { get }
  func embed(_ text: String, for purpose: EmbeddingPurpose) async throws -> [Float]
}

extension EmbedderSource {
  /// An embedder that calls this value.
  var embedder: Embedder {
    Embedder(identifier: identifier) { text, purpose in
      try await self.embed(text, for: purpose)
    }
  }
}

extension MemoryStore {
  init(url: URL, embedder source: some EmbedderSource) throws {
    try self.init(url: url, embedder: source.embedder)
  }

  static func inMemory(embedder source: some EmbedderSource) throws -> MemoryStore {
    try inMemory(embedder: source.embedder)
  }
}

/// A deterministic embedder for tests.
///
/// It returns the vector in `vectors` for text that it knows.
/// For other text, it counts words into buckets by a stable hash,
/// so texts that share words have similar vectors.
struct TestEmbedder: EmbedderSource {
  let identifier: String
  var dimension = 16
  var vectors: [String: [Float]] = [:]
  let calls = CallCounter()

  init(identifier: String = "test-v1", dimension: Int = 16, vectors: [String: [Float]] = [:]) {
    self.identifier = identifier
    self.dimension = dimension
    self.vectors = vectors
  }

  func embed(_ text: String, for purpose: EmbeddingPurpose) async throws -> [Float] {
    calls.record(text, purpose)
    if let vector = vectors[text] { return vector }
    var vector = [Float](repeating: 0, count: dimension)
    for word in text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }) {
      vector[Int(Self.hash(word) % UInt64(dimension))] += 1
    }
    return vector
  }

  /// The 64-bit FNV-1a hash, which, unlike `Hasher`, is the same in every process.
  static func hash(_ word: Substring) -> UInt64 {
    var hash: UInt64 = 0xcbf2_9ce4_8422_2325
    for byte in word.utf8 {
      hash ^= UInt64(byte)
      hash &*= 0x100_0000_01b3
    }
    return hash
  }
}

/// An embedder that always fails.
struct FailingEmbedder: EmbedderSource {
  struct Failure: Error {}
  let identifier = "failing"
  func embed(_ text: String, for purpose: EmbeddingPurpose) async throws -> [Float] {
    throw Failure()
  }
}

/// An embedder that fails for chosen texts and embeds the others.
struct SelectivelyFailingEmbedder: EmbedderSource {
  let base: TestEmbedder
  let failing: Set<String>

  var identifier: String { base.identifier }

  func embed(_ text: String, for purpose: EmbeddingPurpose) async throws -> [Float] {
    if failing.contains(text) { throw FailingEmbedder.Failure() }
    return try await base.embed(text, for: purpose)
  }
}

/// Records the texts that an embedder receives, with their purposes.
final class CallCounter: @unchecked Sendable {
  struct Call: Hashable {
    var text: String
    var purpose: EmbeddingPurpose
  }

  private let lock = NSLock()
  private var calls: [Call] = []

  func record(_ text: String, _ purpose: EmbeddingPurpose) {
    lock.withLock { calls.append(Call(text: text, purpose: purpose)) }
  }

  var count: Int { lock.withLock { calls.count } }
  var all: [String] { lock.withLock { calls.map(\.text) } }
  var withPurposes: [Call] { lock.withLock { calls } }

  /// Removes the recorded calls.
  func reset() {
    lock.withLock { calls = [] }
  }
}

/// An embedder that pauses on chosen texts until a test opens their gates.
struct GatedEmbedder: EmbedderSource {
  let base: TestEmbedder
  let gates: [String: Gate]

  init(_ base: TestEmbedder = TestEmbedder(), gates: [String: Gate]) {
    self.base = base
    self.gates = gates
  }

  var identifier: String { base.identifier }

  func embed(_ text: String, for purpose: EmbeddingPurpose) async throws -> [Float] {
    await gates[text]?.pass()
    return try await base.embed(text, for: purpose)
  }
}

/// A one-time barrier that records when a task reaches it.
///
/// Waiting doesn't respond to cancellation,
/// so a canceled task stays paused until the test calls ``open()``.
final class Gate: @unchecked Sendable {
  private let lock = NSLock()
  private var arrived = false
  private var isOpen = false
  private var arrivalWaiters: [CheckedContinuation<Void, Never>] = []
  private var openWaiters: [CheckedContinuation<Void, Never>] = []

  /// Records the arrival of a task, then waits until the gate opens.
  func pass() async {
    let waiters = lock.withLock {
      arrived = true
      defer { arrivalWaiters = [] }
      return arrivalWaiters
    }
    for waiter in waiters { waiter.resume() }
    await withCheckedContinuation { continuation in
      let resume = lock.withLock {
        if isOpen { return true }
        openWaiters.append(continuation)
        return false
      }
      if resume { continuation.resume() }
    }
  }

  /// Waits until a task reaches the gate.
  func arrival() async {
    await withCheckedContinuation { continuation in
      let resume = lock.withLock {
        if arrived { return true }
        arrivalWaiters.append(continuation)
        return false
      }
      if resume { continuation.resume() }
    }
  }

  /// Lets every waiting task and every later task through.
  func open() {
    let waiters = lock.withLock {
      isOpen = true
      defer { openWaiters = [] }
      return openWaiters
    }
    for waiter in waiters { waiter.resume() }
  }
}
