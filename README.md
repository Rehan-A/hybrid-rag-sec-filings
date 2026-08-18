# Hybrid RAG Pipeline for SEC Filings

A hybrid search RAG system over SEC 10-K/10-Q filings — dense (bge-m3) + BM25 retrieval, RRF fusion, cross-encoder reranking, and citation-verified generation.

Built end-to-end: ingestion → cleaning → structure-aware chunking → hybrid indexing → reranking → citation-grounded QA → RAGAS evaluation.

## Why SEC filings

Financial filings are a strong, defensible test case for hybrid search specifically because dense-only retrieval genuinely struggles here: queries with exact numbers, tickers, and fiscal quarters need lexical precision that embeddings blur, while paraphrased/semantic queries (risk sentiment, strategy language) need dense retrieval. This is a real, benchmarked tradeoff, not a contrived one — [TODO: cite the financial-document hybrid search benchmark you referenced early on].

## Dataset

- 40 filings (10 10-Ks + 30 10-Qs) across 6 tech companies: AAPL, MSFT, GOOGL, NVDA, CRM, ADBE
- Pulled from SEC EDGAR via the [sec-api](https://sec-api.io) Query API for filing discovery, downloaded directly from `sec.gov` (not via sec-api's PDF conversion — native HTML, closer to the real messy source)
- 2024–2025 filing dates

## Architecture

```
Raw HTML (EDGAR)
    ↓
BeautifulSoup cleaning (strip hidden iXBRL data islands, unwrap ix: tags)
    ↓
Section extraction (regex on "Item X." boundaries, TOC vs. real header disambiguation)
    ↓
Sentence splitting (regex sentence boundaries)
    ↓
Token-budget chunking (bge-m3 tokenizer, ~500 tokens, sentence-snapped)
    ↓
Overlap (capped backward walk, ~60 tokens target)
    ↓
[TODO Phase 2] Dense embedding (bge-m3) + BM25 sparse index
    ↓
[TODO Phase 2] RRF fusion
    ↓
[TODO Phase 3] Cross-encoder reranking (bge-reranker-v2-m3)
    ↓
[TODO Phase 4] Citation-grounded generation + NLI-based citation verification
    ↓
[TODO Phase 5] RAGAS evaluation (dense-only vs. hybrid vs. hybrid+rerank)
```

## Design decisions

This section is the actual differentiator of this project — every choice below was made deliberately, with a specific problem behind it, not copied from a tutorial.

### iXBRL cleaning
Modern EDGAR filings embed machine-readable financial tags (`<ix:nonnumeric>`, hidden `display:none` divs full of `fasb.org` taxonomy URLs) directly in the visual HTML. Naive `get_text()` extraction pulls in this invisible metadata as if it were content, polluting the text with a wall of taxonomy URLs. Fix: strip all `display:none` elements before extraction, then unwrap (not delete) `ix:` tags to keep their visible text content.

### Section boundary detection (TOC vs. real header)
Filings have no reliable semantic HTML structure — no real `<h1>`/`<h2>` tags, section titles are just styled spans. "Item X." labels appear at least twice per filing: once in the table of contents, once as the real section header. Disambiguation rule: group all occurrences of each item number by position, take the *last* occurrence as the real header (TOC entries cluster near the start of the document; real content is spread throughout). Validated manually against Apple's 2024 10-K before scaling to all 40 files.

### Fixed-size chunking over structural chunking
Investigated whether real paragraph/subsection structure survived text flattening — it largely doesn't (no reliable `\n\n` or heading markers within a section body). Fell back to fixed-size, token-budget chunking rather than forcing a structural approach the data doesn't support.

### Chunking unit: tokens (bge-m3 tokenizer), not characters
Character count doesn't reliably predict token count — dense numeric content (financial tables) and prose tokenize differently. Chunking against the actual embedding model's tokenizer ensures chunks stay within its effective context, rather than hoping a character-based proxy holds.

### Sentence-boundary snapping (never split mid-sentence)
Chunks always end on a complete sentence, even if that means landing under the token target rather than exactly at it. Rationale: a chunk that's cited to a user should never read as a broken fragment — this matters directly for the citation-verification stage.

### Overlap: token-budget walk, capped
~60-token overlap between consecutive chunks, computed by walking backward through the closing chunk's sentences until the budget is met. **Edge case found and fixed during development**: SEC filings' financial-statement sections (Item 8) and auditor-opinion boilerplate (Item 9A) contain very few sentence-ending periods (long clause-chained legal/tabular text), which the sentence splitter treats as single oversized "sentences" — some over 2,600 tokens. This caused two downstream problems, both fixed:
1. **Oversized single-sentence chunks**: added a token-level hard-split fallback (`split_oversized_sentence`) that only activates when a sentence alone exceeds the chunk budget — the only place in the pipeline where a sentence is deliberately split mid-way, and only as a last resort.
2. **Unbounded overlap**: the backward overlap walk could pull in one of these oversized "sentences" and blow past the 60-token target by 8x. Fixed by capping the size of any single sentence eligible for overlap (120 tokens) — if every candidate sentence exceeds the cap, overlap is skipped for that boundary rather than forced. Affects ~5% of boundaries (reduced overlap) and <1% (zero overlap) — a deliberate "best effort, bounded" tradeoff over "guaranteed overlap at any cost."

### Known limitations
- **Item 8 financial tables are extracted as flattened text**, not structured data — numbers from different fiscal years can appear ambiguously column-adjacent once flattened (e.g. "$294,866 $298,085 $316,199" with no inline label of which year is which). Planned fix: pull structured figures via SEC's XBRL `companyfacts` API instead of relying on text extraction for tabular financial data; treat Item 8 narrative text and its tables as separate extraction paths.
- A small residual (~0.15% of chunks) can still land moderately over the token budget when a single new sentence pushes an already-near-full chunk over the limit — a distinct, smaller mechanism than the two issues above, left unaddressed as a low-priority known case.

## Pipeline stats

| Stage | Result |
|---|---|
| Filings ingested | 40 (10 10-K, 30 10-Q) across 6 tickers |
| Sections extracted | 442 |
| Total chunks | 3,371 |
| Avg tokens/chunk | ~449 (pre-fix); 400.6 (post-fix) |
| Chunks requiring hard-split fallback | 101 (2.9% of pre-fix chunks) |
| Chunk boundaries with reduced/no overlap | 135 reduced (~5%), 18 zero (<1%) |

## Evaluation

[TODO Phase 5 — RAGAS Faithfulness / Context Precision / Context Recall / Answer Relevancy, compared across dense-only, hybrid (RRF), and hybrid+rerank configurations]

## Stack

- **Parsing/cleaning**: BeautifulSoup
- **Embeddings**: BAAI/bge-m3
- **Sparse retrieval**: BM25 [TODO: rank_bm25 or Qdrant native sparse — decide in Phase 2]
- **Vector store**: [TODO: Qdrant]
- **Fusion**: Reciprocal Rank Fusion (RRF)
- **Reranking**: BAAI/bge-reranker-v2-m3
- **Generation**: Ollama [TODO: confirm model]
- **Evaluation**: RAGAS

## Setup

[TODO — fill in once Phase 2+ dependencies are locked down: requirements.txt, .env setup for SEC_API_KEY, how to run the pipeline end to end]
