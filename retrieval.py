from collections import defaultdict

from FlagEmbedding import BGEM3FlagModel
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

QDRANT_PATH = "./qdrant_local_db"
COLLECTION_NAME = "sec_filings"
DENSE_MODEL_NAME = "BAAI/bge-m3"
SPARSE_MODEL_NAME = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = defaultdict(float)
    for rank, result in enumerate(dense_results):
        scores[result.id] += 1 / (k + rank + 1)
    for rank, result in enumerate(sparse_results):
        scores[result.id] += 1 / (k + rank + 1)
    return scores


def hybrid_search(query, client, dense_model, sparse_model, top_k=5, fetch_k=20, k=60):
    # 1. dense query embedding
    dense_query_vec = dense_model.encode(
        [query], return_dense=True, return_sparse=False, return_colbert_vecs=False
    )["dense_vecs"][0].tolist()

    # 2. sparse query embedding
    sparse_query_vec_raw = list(sparse_model.embed([query]))[0]
    sparse_query_vec = models.SparseVector(
        indices=sparse_query_vec_raw.indices.tolist(),
        values=sparse_query_vec_raw.values.tolist(),
    )

    # 3. dense retrieval
    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_query_vec,
        using=DENSE_VECTOR_NAME,
        limit=fetch_k,
        with_payload=True,
    ).points

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

    # sort chunk_ids by fused RRF score, descending, take top_k
    ranked_ids = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    final_results = []
    for chunk_point_id, rrf_score in ranked_ids:
        result = id_to_result[chunk_point_id]
        final_results.append({
            "chunk_id": result.payload["chunk_id"],
            "ticker": result.payload["ticker"],
            "item_num": result.payload["item_num"],
            "text": result.payload["text"],
            "rrf_score": rrf_score,
        })
    
    

    return final_results


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter query: ")

    client = QdrantClient(path=QDRANT_PATH)
    dense_model = BGEM3FlagModel(DENSE_MODEL_NAME, use_fp16=False)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    results = hybrid_search(query, client, dense_model, sparse_model, top_k=5)

    print(f"Query: {query!r}\n")
    print("Top 5 hybrid (RRF-fused) results:\n")
    for r in results:
        preview = r["text"][:150].replace("\n", " ")
        print(f"rrf={r['rrf_score']:.5f}  {r['ticker']} {r['item_num']}  {r['chunk_id']}")
        print(f"  {preview}...")
        print()

    client.close()