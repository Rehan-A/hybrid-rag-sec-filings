import glob
import json
import os
import sys
import uuid

from FlagEmbedding import BGEM3FlagModel
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

sys.stdout.reconfigure(encoding="utf-8")

CHUNKS_DIR = "chunks"
QDRANT_PATH = "./qdrant_local_db"
COLLECTION_NAME = "sec_filings"
DENSE_MODEL_NAME = "BAAI/bge-m3"
SPARSE_MODEL_NAME = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_SIZE = 1024
BATCH_SIZE = 32


def chunk_id_to_point_id(chunk_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def load_all_chunks(chunks_dir):
    chunks = []
    for filepath in sorted(glob.glob(os.path.join(chunks_dir, "*.json"))):
        with open(filepath, "r", encoding="utf-8") as f:
            chunks.extend(json.load(f))
    return chunks


def ensure_collection(client):
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists — reusing it.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(size=DENSE_SIZE, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(),
        },
    )
    print(f"Created collection '{COLLECTION_NAME}'.")


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    print(f"Using embedded Qdrant local mode — data path: {QDRANT_PATH}")
    client = QdrantClient(path=QDRANT_PATH)
    ensure_collection(client)

    print("Loading dense model (BAAI/bge-m3 via FlagEmbedding)...")
    dense_model = BGEM3FlagModel(DENSE_MODEL_NAME, use_fp16=False)
    print("Loading sparse model (Qdrant/bm25 via fastembed)...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    chunks = load_all_chunks(CHUNKS_DIR)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_DIR}/")

    succeeded = 0
    failed = []
    processed = 0

    for batch_num, batch in enumerate(batched(chunks, BATCH_SIZE), start=1):
        try:
            texts = [c["text"] for c in batch]
            dense_vectors = dense_model.encode(
                texts, batch_size=len(texts), return_dense=True, return_sparse=False, return_colbert_vecs=False
            )["dense_vecs"]
            sparse_vectors = list(sparse_model.embed(texts))

            points = []
            for chunk, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
                points.append(models.PointStruct(
                    id=chunk_id_to_point_id(chunk["chunk_id"]),
                    vector={
                        DENSE_VECTOR_NAME: dense_vec.tolist(),
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload=chunk,
                ))

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            succeeded += len(points)

        except Exception as e:
            chunk_ids = [c["chunk_id"] for c in batch]
            failed.append({"batch_num": batch_num, "chunk_ids": chunk_ids, "error": str(e)})
            print(f"Batch {batch_num}: FAILED — {e}")

        processed += len(batch)
        print(f"Batch {batch_num}: {processed} chunks processed so far")

    print(f"\n{succeeded} points indexed successfully, {len(failed)} batches failed")
    if failed:
        print("Failed batches:")
        for f in failed:
            print(f"  Batch {f['batch_num']} ({len(f['chunk_ids'])} chunks): {f['error']}")
            print(f"    chunk_ids: {f['chunk_ids']}")

    # close the writing client to release the local-mode storage lock, then
    # re-open a fresh connection to prove the data actually persisted to disk
    # rather than just existing in this process's memory
    client.close()

    verify_client = QdrantClient(path=QDRANT_PATH)
    count_result = verify_client.count(collection_name=COLLECTION_NAME, exact=True)
    print(f"\nRe-opened fresh connection to {QDRANT_PATH} to verify persistence.")
    print(f"Collection '{COLLECTION_NAME}' point count (from disk): {count_result.count}")
    verify_client.close()


if __name__ == "__main__":
    main()
