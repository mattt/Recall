import Foundation

#if canImport(Accelerate)
  import Accelerate
#endif

/// Operations on embedding vectors.
enum Vector {
  /// Returns `vector` scaled to unit length,
  /// or `nil` if it is empty, has zero length, or contains a value that isn't finite.
  static func normalized(_ vector: [Float]) -> [Float]? {
    guard !vector.isEmpty, vector.allSatisfy(\.isFinite) else { return nil }
    let length = dot(vector, vector).squareRoot()
    guard length.isNormal else { return nil }
    return vector.map { $0 / length }
  }

  /// Returns the dot product of two vectors that have the same number of elements.
  static func dot(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count, "The vectors have different dimensions.")
    #if canImport(Accelerate)
      var result: Float = 0
      vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
      return result
    #else
      var result: Float = 0
      for index in a.indices {
        result += a[index] * b[index]
      }
      return result
    #endif
  }

  /// Returns the bytes of `vector` in host byte order.
  static func encode(_ vector: [Float]) -> Data {
    vector.withUnsafeBufferPointer { Data(buffer: $0) }
  }

  /// Returns the vector in `data`,
  /// or `nil` if `data` doesn't contain exactly `dimension` values.
  static func decode(_ data: Data, dimension: Int) -> [Float]? {
    guard dimension > 0, data.count == dimension * MemoryLayout<Float>.stride else { return nil }
    return [Float](unsafeUninitializedCapacity: dimension) { buffer, count in
      count = data.copyBytes(to: buffer) / MemoryLayout<Float>.stride
    }
  }
}
