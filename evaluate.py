import json
import sys

import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

# Faithfulness and ContextPrecision decompose into several LLM calls per row
# (statement generation + per-statement verification; per-context relevance
# judging), unlike ContextRecall/AnswerRelevancy's one call per row. RAGAS's
# default RunConfig (timeout=180s, max_workers=16) fires 16 of these
# concurrently against a single local Ollama server that can't actually run
# them in parallel, so most queue past the 180s timeout and silently come
# back as NaN. max_workers=1 serializes requests to match what Ollama can
# actually do; timeout is raised generously for slow CPU-bound inference.
EVAL_RUN_CONFIG = RunConfig(timeout=900, max_workers=1)

from generation import generate_answer
from retrieval import (
    QDRANT_PATH,
    DENSE_MODEL_NAME,
    SPARSE_MODEL_NAME,
    RERANKER_MODEL_NAME,
    QdrantClient,
    BGEM3FlagModel,
    FlagReranker,
    SparseTextEmbedding,
    hybrid_search,
)

QUESTIONS_PATH = "eval_questions.json"
RESULTS_PATH = "eval_results.json"

JUDGE_MODEL = "llama3.1:8b"
JUDGE_EMBEDDING_MODEL = "nomic-embed-text"  # llama3.1:8b is a chat model, not tagged
# for embeddings in current Ollama — it rejects /api/embed with a 501 for any
# model lacking that capability, so a real embedding model is required here.

CONFIGS = [
    ("dense_only", "dense"),
    ("hybrid_rrf", "rrf"),
    ("hybrid_rerank", "rerank"),
]

METRIC_NAMES = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]


def build_metrics():
    # fresh instances per run — ragas.evaluate() wires in judge_llm/judge_embeddings itself
    return [Faithfulness(), ContextPrecision(), ContextRecall(), AnswerRelevancy()]


def run_config(config_label, mode, questions, client, dense_model, sparse_model, reranker):
    print(f"\n=== Config: {config_label} (mode={mode}) — retrieval + generation ===")
    rows = []
    for i, q in enumerate(questions, start=1):
        print(f"[{config_label}] question {i}/{len(questions)}: {q['question'][:80]}")
        sources = hybrid_search(q["question"], client, dense_model, sparse_model, reranker, top_k=5, mode=mode)
        contexts = [s["text"] for s in sources]
        answer = generate_answer(q["question"], sources)
        rows.append({
            "user_input": q["question"],
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": q["ground_truth"],
            "category": q["category"],
        })

    print(f"\n[{config_label}] running RAGAS evaluation ({len(rows)} questions x {len(METRIC_NAMES)} metrics)...")
    ds = EvaluationDataset.from_list([
        {"user_input": r["user_input"], "retrieved_contexts": r["retrieved_contexts"],
         "response": r["response"], "reference": r["reference"]}
        for r in rows
    ])

    result = evaluate(
        dataset=ds,
        metrics=build_metrics(),
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=EVAL_RUN_CONFIG,
    )

    per_question = []
    for row, scores in zip(rows, result.scores):
        record = {
            "config": config_label,
            "category": row["category"],
            "question": row["user_input"],
            "ground_truth": row["reference"],
            "response": row["response"],
            "retrieved_contexts": row["retrieved_contexts"],
        }
        record.update(scores)
        per_question.append(record)

    aggregates = {}
    for metric in METRIC_NAMES:
        values = [r.get(metric) for r in per_question if r.get(metric) is not None]
        aggregates[metric] = float(np.nanmean(values)) if values else float("nan")

    print(f"[{config_label}] done: " + ", ".join(f"{m}={v:.4f}" for m, v in aggregates.items()))
    return per_question, aggregates


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} eval questions from {QUESTIONS_PATH}")

    global judge_llm, judge_embeddings
    judge_llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=JUDGE_EMBEDDING_MODEL))

    client = QdrantClient(path=QDRANT_PATH)
    dense_model = BGEM3FlagModel(DENSE_MODEL_NAME, use_fp16=False)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    reranker = FlagReranker(RERANKER_MODEL_NAME, use_fp16=False)

    all_results = []
    all_aggregates = {}

    for config_label, mode in CONFIGS:
        per_question, aggregates = run_config(
            config_label, mode, questions, client, dense_model, sparse_model, reranker
        )
        all_results.extend(per_question)
        all_aggregates[config_label] = aggregates

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_results)} per-question results to {RESULTS_PATH}")

    print("\n=== Final comparison ===")
    header = f"| {'Config':<16} | {'Faithfulness':>12} | {'Context Precision':>18} | {'Context Recall':>15} | {'Answer Relevancy':>17} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    label_map = {"dense_only": "Dense-only", "hybrid_rrf": "Hybrid (RRF)", "hybrid_rerank": "Hybrid + Rerank"}
    for config_label, _ in CONFIGS:
        a = all_aggregates[config_label]
        print(
            f"| {label_map[config_label]:<16} | {a['faithfulness']:>12.4f} | {a['context_precision']:>18.4f} "
            f"| {a['context_recall']:>15.4f} | {a['answer_relevancy']:>17.4f} |"
        )

    client.close()


if __name__ == "__main__":
    main()
