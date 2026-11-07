# Recall

Recall is a local memory store for Swift.
It keeps an assistant's explicit memories:
short text records that a user or an agent asks it to keep,
such as "The user prefers tea to coffee."
Recall saves them in a SQLite file
and finds them again with keyword search,
optional vector search,
and recency,
merged with weighted reciprocal rank fusion.

Recall has no dependencies outside the Swift toolchain and the operating system.
It uses the system SQLite library and its FTS5 full-text search module.
It requires macOS 14 or iOS 17 or later
and Swift 6.

## Remembering and recalling

```swift
import Recall

let directory = URL.applicationSupportDirectory
try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
let url = directory.appending(path: "memories.sqlite")
let embedder = try ContextualEmbedder(language: .english)
let store = try MemoryStore(url: url, embedder: embedder)

try await store.remember(
    "The user prefers tea to coffee.",
    source: "chat",
    metadata: ["topic": "drinks"]
)

for result in try await store.recall("favorite beverage", limit: 5) {
    print(result.memory.text, result.score)
}
```

The store also lists, updates, and forgets memories:

```swift
let recent = try await store.list(limit: 20)
try await store.update(recent[0].id, text: "The user prefers green tea.")
try await store.forget(recent[0].id)
```

`MemoryStore` is an actor,
so one store can serve many tasks.
Each store has its own SQLite connection,
and two stores can open the same file.
The stores can use different embedders:
the file keeps one vector for each memory from each embedder,
so one store doesn't replace the vectors of another.
To keep memories only in memory,
use `MemoryStore.inMemory(embedder:)`.

The embedder is optional.
Without one, recall uses keywords and recency only.

## How ranking works

Recall makes up to three ranked lists for a query:

- **Keyword**: memories that contain words from the query,
  ordered by the FTS5 `bm25()` function.
  The words are stemmed and compared without case or diacritics,
  so "Running" matches "runs" and "cafe" matches "café".
  Recall quotes each word,
  so FTS5 operators and punctuation in a query can't cause an error.
- **Vector**: memories ordered by the cosine similarity
  between their vectors and the query's vector.
  Recall compares the query with every stored vector,
  which is fast enough for thousands of memories.
  It uses Accelerate where it is available.
- **Recency**: the memories from the other two lists, newest first.
  This list only reorders memories that the query found;
  it never adds a memory that matches nothing.

Each list contributes `weight / (k + rank)` to a memory's score,
where ranks start at 1.
The score is the sum of these values.
`RecallConfiguration` sets the constant `k` (default 60)
and the weight of each list
(keyword 1, vector 1, and recency 0.5 by default).
It also limits the number of candidates in each list
and can set a minimum cosine similarity.

```swift
let results = try await store.recall(
    "tea",
    configuration: RecallConfiguration(recencyWeight: 0)
)
```

A list with a weight of zero isn't made.
For example, a vector weight of zero means that recall doesn't call the embedder.

Each `RecallResult` contains the memory, the fused score,
the memory's rank in each list (or `nil`),
and its cosine similarity,
so a client can explain why a memory appears where it does.

## Embedders

`ContextualEmbedder` uses `NLContextualEmbedding` from the NaturalLanguage framework.
It takes the mean of the model's token vectors to get one vector for each text.
The system downloads the model's assets on demand;
call `prepare()` to download and load the model before the first recall.

To use another model,
conform a type to `Embedder`:

```swift
struct ServiceEmbedder: Embedder {
    let identifier = "example-embedding-model-v1"
    let client: EmbeddingClient

    func embed(_ text: String) async throws -> [Float] {
        try await client.embedding(for: text)
    }
}
```

The identifier must change whenever the vectors would change,
for example when you change the model or its version.
The store saves each vector with the embedder's identifier and dimension.
When you open a store with an embedder that has a different identifier,
the store embeds each memory that has no vector from that embedder
before its next vector search.
Call `reembed()` to do this work earlier.
Vectors from other embedders stay in the file,
and a change to the text of a memory deletes all of its vectors.
All vectors from one embedder must have the same number of elements.

If an embedder throws while the store saves a memory,
the store doesn't save the memory.

## Cancellation

Each asynchronous method of `MemoryStore` checks for cancellation before it starts.
`remember` and `update` check again inside their write transaction,
after they take the write lock and immediately before they commit.
A call that is canceled by then rolls back,
throws `CancellationError`, and saves nothing.
The check before the commit is the decision point:
a call that is canceled after it saves its change and returns normally.

Re-embedding is shared work.
When several calls to `reembed()` or `recall` need it at the same time,
they wait for one task.
If one of those calls is canceled,
it stops waiting and throws `CancellationError`,
but the work continues for the others.
The store saves each vector as soon as the embedder returns it,
so the vectors saved before an embedder error remain.
