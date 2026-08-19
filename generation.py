import re
from collections import defaultdict
from typing import Literal

import numpy as np
import ollama
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from chunking import split_into_sentences
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

OLLAMA_MODEL = "llama3.1:8b"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
NLI_LABELS = ["contradiction", "entailment", "neutral"]  # model card's fixed output order

nli_model = CrossEncoder(NLI_MODEL_NAME)

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer ONLY using the numbered sources provided. "
    "Cite the source number in square brackets immediately after every factual claim, e.g. "
    "\"Apple's revenue grew 2% in 2024 [1].\" A claim supported by multiple sources cites all of them, "
    "e.g. \"[1][3]\". Every sentence that states a fact must carry at least one citation. "
    "If the sources do not contain the information needed to answer the question, say so explicitly. "
    "Do not use outside knowledge. Do not state anything not directly supported by the sources. "
    "Write your answer as plain prose sentences only — no markdown, no numbered or bulleted lists, "
    "no headers. Each sentence must stand on its own so it can be checked independently."
)


class CitationVerdict(BaseModel):
    verdict: Literal["supported", "unsupported", "partially_supported"]
    reasoning: str  # one short sentence explaining the verdict


JUDGE_PROMPT_TEMPLATE = (
    "You are verifying whether a source document supports a specific claim.\n\n"
    "Source:\n{source_text}\n\n"
    "Claim:\n{claim_sentence}\n\n"
    "Judge whether the source supports the claim.\n"
    "\"supported\" means the source directly backs up this specific claim — paraphrase and "
    "summarization ARE acceptable as support, this is not strict verbatim matching.\n"
    "\"unsupported\" means the source does not contain this information, or contradicts it.\n"
    "\"partially_supported\" means the source backs up part of the claim but not all of it."
)


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def build_prompt(query, sources):
    source_lines = [f"Source [{i}]: {source['text']}" for i, source in enumerate(sources, start=1)]
    return f"Question: {query}\n\nSources:\n" + "\n\n".join(source_lines)


def generate_answer(query, sources):
    prompt = build_prompt(query, sources)
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Ollama generation failed (model={OLLAMA_MODEL}): {e}") from e

    return response["message"]["content"]


def extract_claims(generated_text):
    claims = []
    for sentence in split_into_sentences(generated_text):
        source_numbers = [int(n) for n in re.findall(r'\[(\d+)\]', sentence)]
        if not source_numbers:
            continue
        claims.append((sentence, source_numbers))
    return claims


def verify_claims(claims, sources):
    # NLI cross-encoders trained on SNLI/MultiNLI expect short, single-clause
    # premises. A full ~500-token source chunk almost never scores as
    # "entailment" against a summary sentence even when well-supported, because
    # entailment there means "restates the whole premise," not "is backed by
    # part of it." So each source chunk is decomposed into its own sentences,
    # NLI is run against each one, and the sentence giving the strongest
    # entailment signal represents that (claim, source) pair.
    records = []
    pairs = []
    pair_group_id = []
    group_meta = []

    for sentence, source_numbers in claims:
        for source_number in source_numbers:
            idx = source_number - 1
            if idx < 0 or idx >= len(sources):
                records.append({
                    "sentence": sentence,
                    "source_number": source_number,
                    "chunk_id": None,
                    "label": "invalid_source",
                    "confidence": None,
                })
                continue

            source = sources[idx]
            premise_sentences = split_into_sentences(source["text"]) or [source["text"]]

            group_id = len(group_meta)
            group_meta.append((sentence, source_number, source["chunk_id"]))
            for premise_sentence in premise_sentences:
                pairs.append((premise_sentence, sentence))
                pair_group_id.append(group_id)

    if pairs:
        raw_scores = nli_model.predict(pairs)
        probs = softmax(np.asarray(raw_scores), axis=1)
        entailment_idx = NLI_LABELS.index("entailment")

        best_per_group = {}
        for group_id, prob_row in zip(pair_group_id, probs):
            best = best_per_group.get(group_id)
            if best is None or prob_row[entailment_idx] > best[entailment_idx]:
                best_per_group[group_id] = prob_row

        for group_id, (sentence, source_number, chunk_id) in enumerate(group_meta):
            best_probs = best_per_group[group_id]
            label_idx = int(np.argmax(best_probs))
            records.append({
                "sentence": sentence,
                "source_number": source_number,
                "chunk_id": chunk_id,
                "label": NLI_LABELS[label_idx],
                "confidence": float(best_probs[label_idx]),
            })

    return records


def verify_claims_llm_judge(claims, sources, model="llama3.1:8b"):
    records = []

    for sentence, source_numbers in claims:
        for source_number in source_numbers:
            idx = source_number - 1
            if idx < 0 or idx >= len(sources):
                records.append({
                    "sentence": sentence,
                    "source_number": source_number,
                    "chunk_id": None,
                    "verdict": "unsupported",
                    "reasoning": "cited source number is out of range",
                })
                continue

            source = sources[idx]
            prompt = JUDGE_PROMPT_TEMPLATE.format(source_text=source["text"], claim_sentence=sentence)

            try:
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    format=CitationVerdict.model_json_schema(),
                    options={"temperature": 0},
                )
                verdict = CitationVerdict.model_validate_json(response.message.content)
                records.append({
                    "sentence": sentence,
                    "source_number": source_number,
                    "chunk_id": source["chunk_id"],
                    "verdict": verdict.verdict,
                    "reasoning": verdict.reasoning,
                })
            except Exception as e:
                print(f"[llm_judge] verification call failed for source [{source_number}]: {e}")
                records.append({
                    "sentence": sentence,
                    "source_number": source_number,
                    "chunk_id": source["chunk_id"],
                    "verdict": "unsupported",
                    "reasoning": f"verification call failed: {e}",
                })

    return records


def _is_supported(record, method):
    if method == "nli":
        return record["label"] == "entailment"
    elif method == "llm_judge":
        return record["verdict"] in ("supported", "partially_supported")
    raise ValueError(f"Unknown verification_method: {method!r}")


def filter_verified_answer(generated_text, verification_records, method="nli"):
    records_by_sentence = defaultdict(list)
    for record in verification_records:
        records_by_sentence[record["sentence"]].append(record)

    kept_sentences = []
    for sentence in split_into_sentences(generated_text):
        records = records_by_sentence.get(sentence)
        if not records:
            # no citations in this sentence — nothing to verify, keep as-is
            # (this also preserves an uncited "sources don't cover this" disclaimer)
            kept_sentences.append(sentence)
            continue
        if any(_is_supported(record, method) for record in records):
            kept_sentences.append(sentence)
        # else: every cited source failed to support this sentence — drop

    return " ".join(kept_sentences)


def answer_with_verified_citations(query, verification_method: Literal["nli", "llm_judge"] = "llm_judge"):
    sources = hybrid_search(query, client, dense_model, sparse_model, reranker, top_k=5)
    raw_answer = generate_answer(query, sources)
    claims = extract_claims(raw_answer)

    if verification_method == "nli":
        verification_log = verify_claims(claims, sources)
    elif verification_method == "llm_judge":
        verification_log = verify_claims_llm_judge(claims, sources)
    else:
        raise ValueError(f"Unknown verification_method: {verification_method!r}")

    verified_answer = filter_verified_answer(raw_answer, verification_log, method=verification_method)

    return {
        "query": query,
        "raw_answer": raw_answer,
        "verified_answer": verified_answer,
        "verification_log": verification_log,
        "sources_used": sources,
    }


def compare_verification_methods(query):
    sources = hybrid_search(query, client, dense_model, sparse_model, reranker, top_k=5)
    raw_answer = generate_answer(query, sources)
    claims = extract_claims(raw_answer)

    nli_records = verify_claims(claims, sources)
    llm_records = verify_claims_llm_judge(claims, sources)

    nli_lookup = {(r["sentence"], r["source_number"]): r for r in nli_records}
    llm_lookup = {(r["sentence"], r["source_number"]): r for r in llm_records}

    print(f"\nQuery: {query!r}\n")
    print("--- Raw answer ---")
    print(raw_answer)

    print("\n--- Side-by-side verification (NLI vs. LLM judge) ---")
    for sentence, source_numbers in claims:
        preview = sentence[:80]
        for source_number in source_numbers:
            nli_record = nli_lookup[(sentence, source_number)]
            llm_record = llm_lookup[(sentence, source_number)]
            print(f"[{source_number}] NLI={nli_record['label']:<13} LLM={llm_record['verdict']:<19} {preview}")

    total = len({s for s, _ in claims})
    nli_kept = sum(
        1 for sentence in {s for s, _ in claims}
        if any(_is_supported(r, "nli") for r in nli_records if r["sentence"] == sentence)
    )
    llm_kept = sum(
        1 for sentence in {s for s, _ in claims}
        if any(_is_supported(r, "llm_judge") for r in llm_records if r["sentence"] == sentence)
    )
    print(f"\nNLI: {nli_kept}/{total} kept, LLM judge: {llm_kept}/{total} kept")

    return {
        "query": query,
        "raw_answer": raw_answer,
        "claims": claims,
        "nli_records": nli_records,
        "llm_records": llm_records,
        "sources_used": sources,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    client = QdrantClient(path=QDRANT_PATH)
    dense_model = BGEM3FlagModel(DENSE_MODEL_NAME, use_fp16=False)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    reranker = FlagReranker(RERANKER_MODEL_NAME, use_fp16=False)

    query = "what are apple's main supply chain risks"
    compare_verification_methods(query)

    client.close()
