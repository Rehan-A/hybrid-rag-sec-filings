# Hybrid RAG Pipeline for SEC Filings

A production-style hybrid search RAG system over SEC 10-K/10-Q filings — dense (bge-m3) + BM25 retrieval, hand-derived RRF fusion, cross-encoder reranking, citation-grounded generation with two compared citation-verification approaches, and a RAGAS evaluation comparing retrieval configurations.

Built and evaluated end-to-end: ingestion → cleaning → structure-aware chunking → hybrid indexing → RRF fusion → reranking → citation-grounded generation + verification → RAGAS evaluation.

Every design decision below was made deliberately in response to a concrete problem hit during development, not copied from a tutorial — including several genuine bugs found, diagnosed, and fixed along the way. That process is documented rather than hidden, because it's a better demonstration of understanding than a pipeline that "just worked."

## Why SEC filings

Financial filings are a strong, defensible test case for hybrid search specifically because dense-only retrieval genuinely struggles here: queries with exact numbers, tickers, and fiscal quarters need lexical precision that embeddings blur, while paraphrased/semantic queries (risk sentiment, strategy language) need dense retrieval. Prior benchmarks on financial documents have found BM25 outperforming dense retrieval on several metrics, with hybrid + reranking giving the largest gains — this project's own evaluation (see below) partially confirms and partially complicates that story on a small local eval set.

## Dataset

- 40 filings (10 10-Ks + 30 10-Qs) across 6 tech companies: AAPL, MSFT, GOOGL, NVDA, CRM, ADBE
- Filed 2024–2025
- Discovered via the [sec-api](https://sec-api.io) Query API, downloaded directly from `sec.gov` as native HTML (not via sec-api's PDF conversion — closer to the real, messy source format and avoids a lossy conversion step)
- 3,371 chunks after cleaning, section extraction, and chunking (see Pipeline Stats below)

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
Token-budget chunking (bge-m3 tokenizer, ~500 tokens, sentence-snapped, capped overlap)
    ↓
Dense embedding (bge-m3, 1024-dim) + sparse embedding (BM25 via fastembed)
    ↓
Qdrant (embedded local mode) — both vectors stored on the same point
    ↓
Hand-derived Reciprocal Rank Fusion (dense + sparse → single ranked candidate set)
    ↓
Cross-encoder reranking (bge-reranker-v2-m3) on top candidates
    ↓
Citation-grounded generation (Ollama, llama3.1:8b)
    ↓
Citation verification — two compared methods: strict NLI entailment vs. LLM-as-judge
    ↓
RAGAS evaluation — Faithfulness / Context Precision / Context Recall / Answer Relevancy,
compared across dense-only, hybrid (RRF), and hybrid+rerank configurations
```

## Design decisions

### iXBRL cleaning
Modern EDGAR filings embed machine-readable financial tags (`<ix:nonnumeric>`, hidden `display:none` divs full of `fasb.org` taxonomy URLs) directly in the visual HTML. Naive `get_text()` extraction pulls this invisible metadata in as if it were content, polluting the text with a wall of taxonomy URLs. Fix: strip all `display:none` elements before extraction, then unwrap (not delete) `ix:` tags to keep their visible text content.

### Section boundary detection (TOC vs. real header)
Filings have no reliable semantic HTML structure — no real `<h1>`/`<h2>` tags; section titles are just styled spans. "Item X." labels appear at least twice per filing: once in the table of contents, once as the real section header. Disambiguation rule: group all occurrences of each item number by position, take the *last* occurrence as the real header (TOC entries cluster near the start of the document; real content is spread throughout the rest). Validated manually against Apple's 2024 10-K before scaling to all 40 files via a Claude Code batch script.

### Fixed-size chunking over structural chunking
Investigated whether real paragraph/subsection structure survived text flattening — it largely doesn't (no reliable `\n\n` or heading markers within a section body). Fell back to fixed-size, token-budget chunking rather than forcing a structural approach the data doesn't support.

### Chunking unit: tokens (bge-m3 tokenizer), not characters
Character count doesn't reliably predict token count — dense numeric content (financial tables) and prose tokenize differently. Chunking against the actual embedding model's tokenizer keeps chunks within its effective context, rather than hoping a character-based proxy holds.

### Sentence-boundary snapping (never split mid-sentence, with one deliberate exception)
Chunks always end on a complete sentence, even if that means landing under the token target rather than exactly at it — a chunk cited to a user should never read as a broken fragment, which matters directly for citation verification. The one exception: a small number of sentences (financial-table rows, auditor-opinion boilerplate) have no internal sentence-ending punctuation and come back as single "sentences" thousands of tokens long. For these, and only these, a token-level hard split is applied as a last resort (see Overlap section below for how this was found).

### Overlap: token-budget walk, capped — a real debugging story
~60-token overlap between consecutive chunks, computed by walking backward through the closing chunk's sentences until the budget is met. Two real bugs were found and fixed during development, both traced to the same root cause:

**Root cause**: SEC filings' financial-statement sections (Item 8) and auditor-opinion boilerplate (Item 9A) contain very long clause-chained text with few sentence-ending periods, which the sentence splitter treats as single oversized "sentences" — some over 2,600 tokens.

1. **Oversized single-sentence chunks.** 101 of the original 3,525 chunks (2.9%) were single sentences alone exceeding the 500-token chunk budget, topping out at 2,602 tokens (NVIDIA's Item 9A auditor report). Fixed with `split_oversized_sentence()`: encode the sentence with the tokenizer, slice the token IDs into budget-sized pieces, decode each piece back to text. This is the only place in the pipeline a sentence is deliberately split mid-way.

2. **Unbounded overlap.** The backward overlap walk had no cap on how large a single sentence it would pull in — one of the same oversized "sentences" could get pulled into overlap and blow the ~60-token target by up to 8x (measured: 485 tokens observed in one case, chunks up to 901 tokens as a result). The first attempted fix (reordering the stop-condition check) was traced, tested, and shown *not* to fix it — a single oversized sentence still gets fully included regardless of when the check runs, since the design already committed to never splitting a sentence mid-way. The actual fix: cap the size of any single sentence eligible for overlap (120 tokens); if every candidate sentence exceeds the cap, overlap is skipped for that boundary rather than forced. This is a deliberate "best-effort, bounded" tradeoff over "guaranteed overlap at any cost" — affects ~5% of chunk boundaries (reduced overlap) and <1% (zero overlap).

Net result after both fixes: 3,371 final chunks, max token count reduced from 901 to a small residual of 5 chunks (0.15%) still moderately over budget via a distinct, smaller, unaddressed mechanism (a new sentence alone pushing an already-near-full chunk over the limit) — documented as a known limitation rather than chased further, given diminishing returns.

### RRF fusion, derived by hand rather than using a library
Reciprocal Rank Fusion combines dense and sparse rankings using only rank *position*, not raw scores — this sidesteps the problem that dense (cosine similarity) and sparse (BM25) scores live on incompatible scales. For each document, `RRF_score = Σ 1/(k + rank_i)` summed across every ranked list it appears in (k=60, standard default); a document absent from a list simply contributes nothing from that list — no penalty term. This means a document ranked well by *both* methods scores meaningfully higher than one ranked #1 by only one method, which is exactly the desired behavior: reward retrieval methods agreeing, don't just reward any single method's top pick.

**Verified on a real query** ("what are apple's main supply chain risks," fetch_k=20): only 4 of 20 candidates were found by both dense and sparse retrieval. The top 4 fused results were exactly those 4 overlapping chunks (RRF scores ≈0.032, consistent with two summed contributions); rank 5 (an NVDA chunk, found by sparse alone via loose keyword overlap on "compete"/"demand") scored ≈0.016 — consistent with a single contribution (1/61 ≈ 0.0164). The formula's behavior was directly confirmed against real retrieval data, not just assumed from the math.

### Cross-encoder reranking
Dense and sparse retrieval are both effectively bi-encoders: query and document are scored independently, so the model never sees them together — fast enough to run at index scale, but structurally unable to reason about whether a document *actually answers* a specific query. A cross-encoder (bge-reranker-v2-m3) fixes this by scoring `(query, document)` pairs jointly, at the cost of needing one forward pass per pair — feasible only on the small candidate set (top ~20) that RRF already narrowed down from the full 3,371-chunk corpus. This means reranking can only reorder what retrieval already surfaced; it cannot rescue a relevant chunk that both dense and sparse missed entirely — a retrieval-stage failure that reranking cannot fix downstream.

**Verified with a real before/after comparison** on the same supply-chain query: reranking correctly demoted the NVDA chunk that RRF's sparse-side keyword overlap had pulled into the top 5 (dropped off the list entirely — it wasn't actually about Apple's supply chain), and promoted two chunks that RRF had ranked outside its top 5 (positions 11 and 15) but were genuinely relevant, just under-scored by both cheap methods individually. Only the #1 result held its rank across both orderings — clear evidence the reranker is doing real, non-trivial work rather than rubber-stamping the fusion order.

### Citation-grounded generation
Every retrieved source is presented to the generation model (Ollama, llama3.1:8b) as a numbered source; the model is explicitly instructed to cite `[N]` after every factual claim and to state explicitly when retrieved sources don't cover the question, rather than answering from general knowledge.

### Citation verification — two methods compared, not one declared "correct"
This is the piece most portfolio RAG projects skip entirely.

**Method 1 — strict NLI entailment.** Each generated claim (sentence containing a `[N]` marker) is checked against its cited source using a pretrained NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`). Initial version ran NLI against the *whole* source chunk as premise and found near-zero entailment across the board — traced to a real limitation, not a bug: NLI models are trained on short, single-sentence premise/hypothesis pairs, and are out-of-distribution against a 400–500 token chunk as premise. Fixed by decomposing each source chunk into its own sentences and taking the max entailment score across them — this immediately surfaced genuine, high-confidence entailment (0.9966 on a near-verbatim match), proving the mechanism works when decomposed correctly. Even after the fix, overall verification rate stayed low (1/8 claims kept on a test query) — NLI's strict, closer-to-verbatim notion of entailment routinely rejects valid paraphrases and summaries, which is a calibration property of the model class, not a further bug.

**Method 2 — LLM-as-judge.** The same claims judged instead by an Ollama call (llama3.1:8b, structured output via Pydantic) explicitly instructed to accept paraphrase and summarization as valid support. On the same query: 8/8 claims kept — a stark contrast to NLI's 1/8. Worth scrutinizing rather than accepting as simply "better": the judge is the same model family that generated the claims, which is a plausible self-preference/leniency bias, unverified against independent ground truth.

**Neither method is declared correct.** This is a real, open tension in RAG faithfulness measurement — NLI is precise but too strict for legitimate summarization; LLM-as-judge is more semantically reasonable but its leniency is itself unverified. Both implementations are kept in the codebase for direct comparison rather than picking a winner and discarding the other.

## Pipeline stats

| Stage | Result |
|---|---|
| Filings ingested | 40 (10 10-K, 30 10-Q) across 6 tickers |
| Total chunks (final, post-fixes) | 3,371 |
| Chunks requiring hard-split fallback (oversized sentences) | 101 of original 3,525 (2.9%) |
| Chunk boundaries with reduced overlap (cap triggered) | 135 of 2,691 boundaries (~5%) |
| Chunk boundaries with zero overlap (cap fully triggered) | 18 of 2,691 boundaries (<1%) |
| Max token count, before overlap-cap fix | 901 |
| Max token count, after all fixes | 616 (5 chunks, 0.15%, via a distinct unaddressed mechanism) |
| RRF fusion, real-query overlap (fetch_k=20) | 4/20 candidates found by both dense and sparse |

## Evaluation (RAGAS)

15–17 hand-written questions (mix of exact-lookup, semantic/conceptual, and cross-document), with ground truth verified by hand against the actual indexed chunks — not fabricated from general knowledge. Evaluated across three retrieval configurations using a fully local judge (Ollama, llama3.1:8b via LangChain wrapper — no external API dependency).

| Config | Faithfulness | Context Precision | Context Recall | Answer Relevancy |
|---|---|---|---|---|
| Dense-only | 0.8241 | 0.4824 | 0.7647 | 0.6491 |
| Hybrid (RRF) | 0.8009 | 0.5354 | 0.6732 | 0.6979 |
| Hybrid + Rerank | 0.7889 | 0.5000 | 0.6987 | 0.7273 |

**This is not a clean "hybrid wins everything" result, reported honestly rather than smoothed over:**
- **Answer Relevancy**: hybrid+rerank wins clearly (0.7273 vs. 0.6491 dense-only) — the expected direction.
- **Context Precision**: both hybrid variants beat dense-only, but plain RRF edges out reranking (0.5354 vs. 0.5000) — reranking did not improve precision on this eval set.
- **Faithfulness and Context Recall**: dense-only actually wins both, the opposite of the expected direction.

**Discussion.** A plausible explanation: the eval set deliberately includes conceptual/semantic questions dense embeddings are well-suited for, and with only top-5 contexts passed to generation, hybrid's BM25 component can pull lexically-matching-but-topically-adjacent chunks into that small window (the same failure mode the NVDA example demonstrated qualitatively), diluting an otherwise clean dense-only top-5 even where hybrid's broader candidate set shows higher precision before the final cutoff. With only 15–17 questions and a single local LLM judge (2 of 204 metric values came back NaN during the run, excluded from the aggregate via `nanmean` rather than silently miscounted), some of this spread is very plausibly noise rather than a robust effect — a larger eval set and a stronger/external judge model would be needed to say more with confidence. This is reported as a genuine, mixed empirical finding rather than adjusted to match the expected narrative.

Full per-question results (question, retrieved contexts, generated answer, all four metric scores, per config) are saved in `eval_results.json` for inspection beyond the aggregate table.

## Known limitations

- **Item 8 financial tables are extracted as flattened text**, not structured data — numbers from different fiscal years can appear ambiguously column-adjacent once flattened (e.g. "$294,866 $298,085 $316,199" with no inline label of which year is which). Planned fix: pull structured figures via SEC's XBRL `companyfacts` API instead of relying on text extraction for tabular financial data; treat Item 8 narrative text and its tables as separate extraction paths.
- 5 chunks (0.15%) can still land moderately over the token budget (up to 616 tokens) via a mechanism distinct from the two fixed overlap/oversized-sentence issues — a new sentence alone pushing an already-near-full chunk over the limit. Left as a documented, low-priority known case.
- Citation verification has two implementations with materially different verification rates (1/8 vs. 8/8 on the same test query) and neither is validated against independent human judgment — a small human-eval spot-check would be the natural next step to adjudicate between them.
- RAGAS evaluation uses a fully local LLM judge (llama3.1:8b) rather than a stronger external model, and the eval set is small (15–17 questions) — the mixed Phase 5 results should be read with real sample-size and judge-variance caveats, not treated as a definitive verdict on hybrid vs. dense-only retrieval.

## Stack

- **Parsing/cleaning**: BeautifulSoup
- **Dense embeddings**: BAAI/bge-m3 (via FlagEmbedding)
- **Sparse retrieval**: BM25 via fastembed (`Qdrant/bm25`)
- **Vector store**: Qdrant, embedded local mode (no Docker/server dependency)
- **Fusion**: Reciprocal Rank Fusion (hand-implemented)
- **Reranking**: BAAI/bge-reranker-v2-m3 (via FlagEmbedding)
- **Generation**: Ollama, llama3.1:8b
- **Citation verification**: `cross-encoder/nli-deberta-v3-base` (strict NLI) and Ollama llama3.1:8b with structured output (LLM-as-judge) — both implemented and compared
- **Evaluation**: RAGAS, with a local Ollama judge via LangChain wrappers (no external API dependency)

## Setup notes

- `transformers` must be pinned to `<5.0` (tested on `4.57.6`) — `FlagEmbedding`'s reranker code calls `prepare_for_model`, which was removed in `transformers>=5.0`; this is a real upstream incompatibility, not a local misconfiguration.
- Qdrant runs in embedded local mode (`QdrantClient(path="./qdrant_local_db")`) — no Docker required. Only one process can hold the local DB lock at a time; indexing and querying scripts must be run sequentially, not concurrently.
- Requires Ollama running locally with `llama3.1:8b` pulled (`ollama pull llama3.1:8b`).
- `SEC_API_KEY` required as an environment variable (`.env` file) for filing discovery via sec-api's Query API; document download itself is direct from `sec.gov` and requires no key, only a compliant `User-Agent` header.

[TODO: requirements.txt / environment setup, exact run order for the scripts]