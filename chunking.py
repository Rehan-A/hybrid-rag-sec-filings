import json
import os
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")


def split_into_sentences(text):
    sentence = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s for s in sentence if s.strip()]


def count_tokens(text):
    return len(tokenizer.encode(text))


def split_oversized_sentence(sentence, max_tokens=500):
    token_ids = tokenizer.encode(sentence)
    pieces = []
    for i in range(0, len(token_ids), max_tokens):
        piece_ids = token_ids[i:i + max_tokens]
        piece_text = tokenizer.decode(piece_ids)
        pieces.append(piece_text)
    return pieces


def chunk_sentences(sentences, max_tokens=500):
    all_chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if sentence_tokens > max_tokens:
            if current_chunk_sentences:
                all_chunks.append(current_chunk_sentences)
                overlap = get_overlap_sentences(current_chunk_sentences)
                current_chunk_sentences = overlap
                current_chunk_tokens = count_tokens(" ".join(overlap))

            for piece in split_oversized_sentence(sentence, max_tokens):
                all_chunks.append([piece])

            current_chunk_sentences = []
            current_chunk_tokens = 0
            continue

        if current_chunk_tokens + sentence_tokens <= max_tokens:
            current_chunk_sentences.append(sentence)
            current_chunk_tokens = current_chunk_tokens + sentence_tokens
        else:
            all_chunks.append(current_chunk_sentences)
            overlap = get_overlap_sentences(current_chunk_sentences)
            current_chunk_sentences = overlap
            current_chunk_tokens = count_tokens(" ".join(overlap))
            current_chunk_sentences.append(sentence)
            current_chunk_tokens = current_chunk_tokens + sentence_tokens

    if current_chunk_sentences:
        all_chunks.append(current_chunk_sentences)

    return all_chunks

def get_overlap_sentences(chunk_sentences, overlap_tokens=60):
    max_overlap_sentence_tokens = overlap_tokens * 2
    carry_over = []
    accumulated_tokens = 0

    for sentence in reversed(chunk_sentences):
        sentence_tokens = count_tokens(sentence)
        if sentence_tokens > max_overlap_sentence_tokens:
            continue
        accumulated_tokens += sentence_tokens
        carry_over.append(sentence)
        if accumulated_tokens >= overlap_tokens:
            break

    carry_over = list(reversed(carry_over))

    return carry_over


def process_all_files(sections_dir="sections", output_dir="chunks", max_tokens=500, overlap_tokens=60):
    os.makedirs(output_dir, exist_ok=True)

    filenames = sorted(f for f in os.listdir(sections_dir) if f.lower().endswith(".json"))

    succeeded = []
    failed = []
    total_chunks = 0
    all_token_counts = []

    for filename in filenames:
        try:
            in_path = os.path.join(sections_dir, filename)
            with open(in_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            ticker = data["ticker"]
            form_type = data["form_type"]
            filing_date = data["filing_date"]
            sections = data["sections"]

            file_chunks = []
            for item_num, section_text in sections.items():
                sentences = split_into_sentences(section_text)
                if not sentences:
                    continue

                chunks = chunk_sentences(sentences, max_tokens)
                for chunk_index, chunk in enumerate(chunks):
                    chunk_text = " ".join(chunk)
                    file_chunks.append({
                        "chunk_id": f"{ticker}_{form_type}_{filing_date}_{item_num}_{chunk_index}",
                        "ticker": ticker,
                        "form_type": form_type,
                        "filing_date": filing_date,
                        "item_num": item_num,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "token_count": count_tokens(chunk_text),
                    })

            out_name = os.path.splitext(filename)[0] + ".json"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(file_chunks, f, indent=2, ensure_ascii=False)

            total_chunks += len(file_chunks)
            all_token_counts.extend(c["token_count"] for c in file_chunks)
            succeeded.append(filename)
            print(f"{filename}: {len(sections)} sections, {len(file_chunks)} chunks")

        except Exception as e:
            failed.append((filename, str(e)))
            print(f"{filename}: FAILED — {e}")

    print(f"\n{len(succeeded)} files processed, {len(failed)} failed")
    if failed:
        print("Failed files:")
        for filename, err in failed:
            print(f"  {filename}: {err}")

    if succeeded:
        print(f"Total chunks generated: {total_chunks}")
        print(f"Average chunks per file: {total_chunks / len(succeeded):.1f}")

    if all_token_counts:
        avg_tokens = sum(all_token_counts) / len(all_token_counts)
        print(f"Average token_count per chunk: {avg_tokens:.1f} (target <= {max_tokens})")
        print(f"Min token_count: {min(all_token_counts)}")
        print(f"Max token_count: {max(all_token_counts)}")


if __name__ == "__main__":
    process_all_files()