import json
import gzip
from pathlib import Path
from tqdm import tqdm


INPUT_GZ = Path("C:/Users/Admin/Project/Multi-hop-QA/data/wiki_cleaned_from_dump/enwiki-latest.jsonl.gz") 

TITLES_FILE = Path("C:/Users/Admin/Project/Multi-hop-QA/data/urls/unique_titles_lower.txt")
OUTPUT_FILTERED = Path("C:/Users/Admin/Project/Multi-hop-QA/data/processed/filtered_wiki.jsonl")

# Load set titles 
with open(TITLES_FILE, 'r', encoding='utf-8') as f:
    wanted_titles = {line.strip() for line in f if line.strip()}



extracted = 0
skipped_invalid = 0


with gzip.open(INPUT_GZ, 'rt', encoding='utf-8') as f_in, \
        open(OUTPUT_FILTERED, 'w', encoding='utf-8') as f_out:

    for line in tqdm(f_in, desc="Filtering articles"):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
            title = doc.get('title', '').strip().lower()
            if title in wanted_titles:
                f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
                extracted += 1
        except json.JSONDecodeError:
            skipped_invalid += 1
        except Exception as e:
            print(f"Lỗi dòng: {e}")
            skipped_invalid += 1



print(f"\nCompleted!")
print(f"→ Extracted: {extracted:,} articles with my unique list")
print(f"→ Skipped invalid lines: {skipped_invalid}")
print(f"→ File output: {OUTPUT_FILTERED}")
print(f"→ Estimated size: ~{extracted * 30 / 1024 / 1024:.1f}–{extracted * 80 / 1024 / 1024:.1f} MB ")