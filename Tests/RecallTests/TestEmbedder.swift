import Foundation
import Recall

/// A deterministic embedder for tests.
///
/// It returns the vector in `vectors` for text that it knows.
/// For other text, it counts words into buckets by a stable hash,
/// so texts that share words have similar vectors.
struct TestEmbedder: Embedder {
  let identifier: String
  var dimension = 16
  var vectors: [String: [Float]] = [:]
  let calls = CallCounter()

  init(identifier: String = "test-v1", dimension: Int = 16, vectors: [String: [Float]] = [:]) {
    self.identifier = identifier
    self.dimension = dimension
    self.vectors = vectors
  }

  func embed(_ text: String) async throws -> [Float] {
    calls.increment(text)
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
struct FailingEmbedder: Embedder {
  struct Failure: Error {}
  let identifier = "failing"
  func embed(_ text: String) async throws -> [Float] { throw Failure() }
}

/// Records the texts that an embedder receives.
final class CallCounter: @unchecked Sendable {
  private let lock = NSLock()
  private var texts: [String] = []

  func increment(_ text: String) {
    lock.withLock { texts.append(text) }
  }

  var count: Int { lock.withLock { texts.count } }
  var all: [String] { lock.withLock { texts } }
}
