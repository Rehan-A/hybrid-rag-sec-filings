from collections import defaultdict

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

QDRANT_PATH = "./qdrant_local_db"
COLLECTION_NAME = "sec_filings"
DENSE_MODEL_NAME = "BAAI/bge-m3"
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = defaultdict(float)
    for rank, result in enumerate(dense_results):
        scores[result.id] += 1 / (k + rank + 1)
    for rank, result in enumerate(sparse_results):
        scores[result.id] += 1 / (k + rank + 1)
    return scores


def _format_result(result, rrf_score=None, rerank_score=None):
    return {
        "chunk_id": result.payload["chunk_id"],
        "ticker": result.payload["ticker"],
        "item_num": result.payload["item_num"],
        "text": result.payload["text"],
        "rrf_score": rrf_score,
        "rerank_score": rerank_score,
    }


def hybrid_search(query, client, dense_model, sparse_model, reranker=None, top_k=5, fetch_k=20, k=60,
                   rerank_candidates=20, mode="rerank"):
    # mode="dense": dense retrieval only, no fusion, no reranking
    # mode="rrf": dense + sparse fused with RRF, top_k taken straight from fusion (no reranking)
    # mode="rerank" (default): full pipeline — RRF fusion, then cross-encoder reranking

    # 1. dense query embedding
    dense_query_vec = dense_model.encode(
        [query], return_dense=True, return_sparse=False, return_colbert_vecs=False
    )["dense_vecs"][0].tolist()

    # 2. dense retrieval
    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_query_vec,
        using=DENSE_VECTOR_NAME,
        limit=top_k if mode == "dense" else fetch_k,
        with_payload=True,
    ).points

    if mode == "dense":
        return [_format_result(r) for r in dense_results[:top_k]]

    # 3. sparse query embedding
    sparse_query_vec_raw = list(sparse_model.embed([query]))[0]
    sparse_query_vec = models.SparseVector(
        indices=sparse_query_vec_raw.indices.tolist(),
        values=sparse_query_vec_raw.values.tolist(),
    )

    # 4. sparse retrieval
    sparse_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_query_vec,
        using=SPARSE_VECTOR_NAME,
        limit=fetch_k,
        with_payload=True,
    ).points

    # --- DEBUG: check overlap between dense and sparse result sets ---
    dense_ids = {r.id for r in dense_results}
    sparse_ids = {r.id for r in sparse_results}
    both = dense_ids & sparse_ids
    print(f"[debug] dense returned {len(dense_ids)}, sparse returned {len(sparse_ids)}, overlap: {len(both)}")
    # -------------------------------------------------------------

    # 5. fuse
    fused_scores = reciprocal_rank_fusion(dense_results, sparse_results, k=k)

    # 6. build a lookup from chunk_id -> full result object (payload included)
    #    so we can return complete records, not just scores
    id_to_result = {}
    for result in dense_results + sparse_results:
        id_to_result[result.id] = result

    if mode == "rrf":
        top_ids = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [_format_result(id_to_result[chunk_point_id], rrf_score=rrf_score) for chunk_point_id, rrf_score in top_ids]

    # mode == "rerank": sort chunk_ids by fused RRF score, descending — take a
    # larger candidate set than top_k, so the reranker has real options to work with
    candidate_ids = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:rerank_candidates]

    # --- DEBUG: original RRF-only order, before reranking ---
    print(f"[debug] RRF-only top {top_k} (pre-rerank):")
    for chunk_point_id, rrf_score in candidate_ids[:top_k]:
        result = id_to_result[chunk_point_id]
        print(f"  rrf={rrf_score:.5f}  {result.payload['ticker']} {result.payload['item_num']}  {result.payload['chunk_id']}")
    # ---------------------------------------------------------

    # 7. rerank the candidate set with a cross-encoder, scored in one batched call
    candidates = [id_to_result[chunk_point_id] for chunk_point_id, _ in candidate_ids]
    pairs = [[query, candidate.payload["text"]] for candidate in candidates]
    rerank_scores = reranker.compute_score(pairs, normalize=True)

    # 8. sort candidates by rerank score, descending — RRF score is discarded
    #    for ordering purposes but kept in the result for transparency
    reranked = sorted(
        zip(candidates, [rrf_score for _, rrf_score in candidate_ids], rerank_scores),
        key=lambda item: item[2],
        reverse=True,
    )[:top_k]

    return [_format_result(result, rrf_score=rrf_score, rerank_score=rerank_score)
            for result, rrf_score, rerank_score in reranked]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter query: ")

    client = QdrantClient(path=QDRANT_PATH)
    dense_model = BGEM3FlagModel(DENSE_MODEL_NAME, use_fp16=False)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    reranker = FlagReranker(RERANKER_MODEL_NAME, use_fp16=False)

    results = hybrid_search(query, client, dense_model, sparse_model, reranker, top_k=5, rerank_candidates=20)

    print(f"\nQuery: {query!r}\n")
    print("Top 5 hybrid results (RRF candidates, reranked):\n")
    for r in results:
        preview = r["text"][:150].replace("\n", " ")
        print(f"rerank={r['rerank_score']:.4f}  rrf={r['rrf_score']:.5f}  {r['ticker']} {r['item_num']}  {r['chunk_id']}")
        print(f"  {preview}...")
        print()

    client.close()