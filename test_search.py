# Run this only after index_chunks.py has finished and exited — Qdrant's
# embedded local mode allows only one process to hold the qdrant_local_db
# folder lock at a time.
import sys

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient

sys.stdout.reconfigure(encoding="utf-8")

QDRANT_PATH = "./qdrant_local_db"
COLLECTION_NAME = "sec_filings"
DENSE_MODEL_NAME = "BAAI/bge-m3"
DENSE_VECTOR_NAME = "dense"
TOP_K = 5


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter query: ")

    client = QdrantClient(path=QDRANT_PATH)
    dense_model = BGEM3FlagModel(DENSE_MODEL_NAME, use_fp16=False)

    query_vector = dense_model.encode(
        [query], return_dense=True, return_sparse=False, return_colbert_vecs=False
    )["dense_vecs"][0].tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=TOP_K,
        with_payload=True,
    )

    print(f"Query: {query!r}")
    print(f"Top {TOP_K} dense-only results:\n")
    for point in results.points:
        payload = point.payload
        preview = payload["text"][:150].replace("\n", " ")
        print(f"score={point.score:.4f}  {payload['ticker']} {payload['item_num']}  {payload['chunk_id']}")
        print(f"  {preview}...")
        print()

    client.close()


if __name__ == "__main__":
    main()
