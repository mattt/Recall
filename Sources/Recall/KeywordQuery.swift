/// Converts user text into an FTS5 query that can't fail to parse.
///
/// FTS5 gives meaning to quotation marks, parentheses, `*`, `^`, `:`, `-`, `+`,
/// and the words `AND`, `OR`, `NOT`, and `NEAR`.
/// To keep user text from using that syntax,
/// this type splits the text into runs of letters and numbers,
/// puts each run in double quotation marks,
/// and joins the runs with `OR`.
/// A quoted run is an FTS5 string, so FTS5 reads it only as a term to match.
///
/// Because the runs are joined with `OR`,
/// a common word such as "my" matches most memories.
/// With `ignoringStopWords`,
/// the query leaves out the words in ``stopWords``,
/// unless every word of the text is a stop word.
enum KeywordQuery {
  /// Returns an FTS5 `MATCH` expression for `text`,
  /// or `nil` if the text contains no letters or numbers.
  ///
  /// - Parameters:
  ///   - text: The user's text.
  ///   - ignoringStopWords: Whether to leave out the words in ``stopWords``.
  ///     If every word is a stop word, the query keeps them all.
  static func match(for text: String, ignoringStopWords: Bool = false) -> String? {
    var seen: Set<String> = []
    var terms: [String] = []
    var contentTerms: [String] = []
    for run in text.split(whereSeparator: { !($0.isLetter || $0.isNumber) }) {
      let term = String(run)
      let folded = term.lowercased()
      guard seen.insert(folded).inserted else { continue }
      terms.append(quoted(term))
      if !stopWords.contains(folded) {
        contentTerms.append(quoted(term))
      }
    }
    let chosen = ignoringStopWords && !contentTerms.isEmpty ? contentTerms : terms
    guard !chosen.isEmpty else { return nil }
    return chosen.joined(separator: " OR ")
  }

  /// Common English words that say little about which memory a query wants.
  ///
  /// The list has articles, pronouns, auxiliary verbs, prepositions, conjunctions,
  /// question words, and the pieces of contractions that remain
  /// after the text is split at apostrophes, such as the "s" of "Mirela's".
  /// It is in lowercase, and the comparison ignores case.
  static let stopWords: Set<String> = [
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "been", "before", "being", "but", "by", "can", "could", "d", "did", "do",
    "does", "doing", "for", "from", "had", "has", "have", "having", "he", "her",
    "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "just", "ll", "m", "me", "mine", "my", "of", "on", "or", "our", "ours", "re", "s",
    "she", "should", "so", "some", "t", "than", "that", "the", "their", "theirs",
    "them", "then", "there", "these", "they", "this", "those", "to", "too", "us",
    "ve", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
  ]

  /// Returns `term` as an FTS5 string,
  /// with each double quotation mark in it doubled.
  static func quoted(_ term: String) -> String {
    "\"" + term.replacing("\"", with: "\"\"") + "\""
  }
}
