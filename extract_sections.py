import json
import os
import re
from collections import defaultdict

from bs4 import BeautifulSoup

FILINGS_DIR = "filings"
SECTIONS_DIR = "sections"

ITEM_PATTERN = re.compile(r'\bItem\s*(\d+[A-Z]?)\s*\.\s*', re.IGNORECASE)


def parse_filename(filename):
    stem = os.path.splitext(filename)[0]
    ticker, form_type, filing_date = stem.split("_")
    return ticker, form_type, filing_date


def extract_sections(html):
    soup = BeautifulSoup(html, "html.parser")

    # strip hidden XBRL data islands
    hidden = soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", ""))
    for tag in hidden:
        tag.decompose()

    # unwrap ix: namespace tags but keep their visible text
    for tag in soup.find_all(lambda t: t.name and t.name.startswith("ix:")):
        tag.unwrap()

    clean_text = soup.get_text(separator=" ", strip=True)

    matches = list(ITEM_PATTERN.finditer(clean_text))

    # group all match positions by item number, in document order
    by_item = defaultdict(list)
    for m in matches:
        by_item[m.group(1)].append(m.start())

    # for each item number, the LAST occurrence is the real section header
    real_headers = {item_num: positions[-1] for item_num, positions in by_item.items()}

    # sort by position so we know section order and can compute section boundaries
    sorted_items = sorted(real_headers.items(), key=lambda x: x[1])

    sections = {}
    for i, (item_num, start) in enumerate(sorted_items):
        if i < len(sorted_items) - 1:
            end = sorted_items[i + 1][1]
        else:
            end = len(clean_text)
        sections[item_num] = clean_text[start:end]

    return sections


def main():
    os.makedirs(SECTIONS_DIR, exist_ok=True)

    filenames = sorted(f for f in os.listdir(FILINGS_DIR) if f.lower().endswith(".html"))

    succeeded = []
    failed = []

    for filename in filenames:
        try:
            ticker, form_type, filing_date = parse_filename(filename)

            in_path = os.path.join(FILINGS_DIR, filename)
            with open(in_path, "r", encoding="utf-8") as f:
                html = f.read()

            sections = extract_sections(html)

            record = {
                "source_file": filename,
                "ticker": ticker,
                "form_type": form_type,
                "filing_date": filing_date,
                "sections": sections,
            }

            out_name = os.path.splitext(filename)[0] + ".json"
            out_path = os.path.join(SECTIONS_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

            total_chars = sum(len(text) for text in sections.values())
            succeeded.append((filename, len(sections), total_chars))
            print(f"{filename}: {len(sections)} sections found")

        except Exception as e:
            failed.append((filename, str(e)))
            print(f"{filename}: FAILED — {e}")

    print(f"\n{len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed files:")
        for filename, err in failed:
            print(f"  {filename}: {err}")

    if succeeded:
        print("\nSummary:")
        print(f"{'file':<40} {'items':>6} {'chars':>10}")
        for filename, item_count, total_chars in succeeded:
            print(f"{filename:<40} {item_count:>6} {total_chars:>10}")


if __name__ == "__main__":
    main()
