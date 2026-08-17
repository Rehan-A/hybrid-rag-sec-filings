import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sec_api import QueryApi
from collections import defaultdict

load_dotenv()

# SEC filings contain characters (e.g. checkbox glyphs) outside the
# Windows console's default cp1252 codepage — force UTF-8 stdout so
# printing filing text doesn't crash mid-script.
sys.stdout.reconfigure(encoding="utf-8")

api_key = os.getenv("SEC_API_KEY")
if not api_key:
    raise ValueError("Missing SEC_API_KEY in the environment (.env) file.")

queryApi = QueryApi(api_key=api_key)

# pick your sector — example: 6 tech companies
tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "CRM", "ADBE"]

REQUEST_HEADERS = {"User-Agent": "Rehan Sayyad rehansayyaduk@gmail.com"}


def fetch_filings(form_type, tickers, start_date, end_date, size=50):
    ticker_query = " OR ".join(f'ticker:{t}' for t in tickers)
    query = {
        "query": f'({ticker_query}) AND formType:"{form_type}" '
                 f'AND filedAt:[{start_date} TO {end_date}]',
        "from": "0",
        "size": str(size),
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    return queryApi.get_filings(query)


def download_filings(filings_response, out_dir="filings"):
    os.makedirs(out_dir, exist_ok=True)
    saved, failed = [], []
    for filing in filings_response["filings"]:
        url = filing["linkToFilingDetails"]
        fname = f'{filing["ticker"]}_{filing["formType"]}_{filing["filedAt"][:10]}.html'
        path = os.path.join(out_dir, fname)
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()
            with open(path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            saved.append(path)
        except Exception as e:
            print(f"Failed {fname}: {e}")
            failed.append(fname)
        time.sleep(0.2)  # be polite to SEC's servers
    return saved, failed


filings_10k = fetch_filings("10-K", tickers, "2024-01-01", "2025-12-31")
filings_10q = fetch_filings("10-Q", tickers, "2024-01-01", "2025-12-31")

print(f"10-K: {filings_10k['total']['value']} found")
print(f"10-Q: {filings_10q['total']['value']} found")

paths_10k, failed_10k = download_filings(filings_10k)
paths_10q, failed_10q = download_filings(filings_10q)

print(f"Saved {len(paths_10k)} 10-Ks and {len(paths_10q)} 10-Qs to disk")
if failed_10k or failed_10q:
    print(f"Failed: {len(failed_10k)} 10-Ks, {len(failed_10q)} 10-Qs")

for f in sorted(os.listdir("filings"))[:5]:
    size = os.path.getsize(os.path.join("filings", f))
    print(f"{f}: {size/1024:.1f} KB")


# --- Parse a single filing ---
sample_path = "filings/AAPL_10-K_2024-11-01.html"
with open(sample_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

raw_text_length = len(soup.get_text(separator=" ", strip=True))

# strip hidden XBRL data island
hidden = soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", ""))
print(f"Hidden elements found: {len(hidden)}")
for tag in hidden:
    tag.decompose()

# unwrap ix: namespace tags but keep their visible text
for tag in soup.find_all(lambda t: t.name and t.name.startswith("ix:")):
    tag.unwrap()

clean_text = soup.get_text(separator=" ", strip=True)
print(f"Text length after cleanup: {len(clean_text)} chars (was {raw_text_length})")
print(clean_text[:1500])

# 1. How many tables are in here, and how big are they?
tables = soup.find_all("table")
print(f"\nTotal <table> tags: {len(tables)}")
for i, t in enumerate(tables[:5]):
    rows = t.find_all("tr")
    print(f"  Table {i}: {len(rows)} rows")

# 2. What does a real financial table actually look like as raw HTML?
print("\n--- Sample table (first one with >5 rows) ---")
for t in tables:
    if len(t.find_all("tr")) > 5:
        print(t.prettify()[:1500])
        break

# 3. What headers/structure exist for prose sections?
header_tags = soup.find_all(["h1", "h2", "h3", "b", "strong"])
print(f"\nHeader-like tags found: {len(header_tags)}")
for h in header_tags[:20]:
    text = h.get_text(strip=True)
    if text:
        print(f"  {h.name}: {text[:80]}")


# --- Chunking exploration on the cleaned text ---

# 1. Find Item section boundaries — this is how you'll split into logical chunks
item_pattern = re.compile(r'\bItem\s+\d+[A-Z]?\.\s')
matches = list(item_pattern.finditer(clean_text))
print(f"\n'Item X.' matches found: {len(matches)}")
for m in matches[:15]:
    snippet = clean_text[m.start():m.start()+60]
    print(f"  @ char {m.start()}: {snippet}")

# 2. Check for leftover ix: residue anywhere in the clean text
residue = clean_text.count("fasb.org") + clean_text.count("<ix:")
print(f"\nLeftover XBRL residue count: {residue}")

# 3. Natural paragraph/sentence density — split on periods followed by space+capital
# (rough proxy since there's no real paragraph markup)
sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean_text)
print(f"\nApprox sentence count: {len(sentences)}")
print(f"Avg sentence length (chars): {sum(len(s) for s in sentences)/len(sentences):.0f}")

# 4. Peek at the actual Risk Factors section if found
risk_start = clean_text.find("Item 1A")
if risk_start != -1:
    print(f"\n--- Item 1A section preview ---")
    print(clean_text[risk_start:risk_start+1200])


item_pattern = re.compile(r'\bItem\s*(\d+[A-Z]?)\s*\.\s*', re.IGNORECASE)
matches = list(item_pattern.finditer(clean_text))

# group all match positions by item number, in document order
by_item = defaultdict(list)
for m in matches:
    by_item[m.group(1)].append(m.start())

# for each item number, take the LAST occurrence as the real section header
real_headers = {}
for item_num, positions in by_item.items():
    real_headers[item_num] = positions[-1]  # works whether there are 1 or many occurrences

# sort by position so we know section order and can compute section boundaries
sorted_items = sorted(real_headers.items(), key=lambda x: x[1])

print("Real section headers found:")
for item_num, pos in sorted_items:
    print(f"  Item {item_num} @ char {pos}")


sections = {} 

for i in range(len(sorted_items)):
    item_num, start = sorted_items[i]  # unpack tuple: item_num = sorted_items[i][0], start = sorted_items[i][1]

    if i < len(sorted_items) - 1:
        # not the last item — end where the NEXT item's position starts
        _, next_start = sorted_items[i + 1]
        end = next_start
    else:
        # last item — run to the end of the document
        end = len(clean_text)

    sections[item_num] = clean_text[start:end]

# quick check
for item_num, text in sections.items():
    print(f"Item {item_num}: {len(text)} chars — preview: {text[:100]!r}")



