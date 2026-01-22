# scripts/2_extract_unique_pages.py
"""
Extract unique Wikipedia page titles từ:
- IIRC: field 'links' trong context
- MuSiQue: field 'supporting_facts' (list [title, sent_id])

Lưu vào: data/urls/unique_pages_iirc_musique.jsonl
"""
import json
from pathlib import Path
#from collections import set
from tqdm import tqdm

DATA_ROOT = Path("data")
URL_DIR = DATA_ROOT / "urls"
URL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = URL_DIR / "unique_pages_iirc_musique.jsonl"

IIRC_DIR = DATA_ROOT / "raw" / "iirc"
MUSIQUE_DIR = DATA_ROOT / "raw" / "musique"

def extract_from_iirc(file_path: Path) -> set:
    pages = set()
    if not file_path.exists():
        print(f"Skip IIRC file: {file_path}")
        return pages

    # Giả định file là JSON array (mỗi item là một example)
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f) if file_path.suffix == ".json" else [json.loads(line) for line in f]

    for ex in tqdm(data, desc=f"Extracting {file_path.name}"):
        # IIRC: links là list of dict {'text': ..., 'target': title}
        for link in ex.get("links", []):
            target = link.get("target") or link.get("page")
            if target:
                title = target.strip().replace(" ", "_").replace("/", "_")
                if title:
                    pages.add(title)
    return pages

def extract_from_musique(file_path: Path) -> set:
    pages = set()
    if not file_path.exists():
        print(f"Skip MuSiQue file: {file_path}")
        return pages

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc=f"Extracting {file_path.name}")):
            try:
                ex = json.loads(line)
                paragraphs = ex.get("paragraphs", [])
                for para in paragraphs:
                    title = para.get("title")
                    if title:
                        clean_title = title.strip().replace(" ", "_").replace("/", "_")
                        if clean_title:
                            pages.add(clean_title)
            except json.JSONDecodeError:
                print(f"JSON error at line {line_num+1} in {file_path}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num+1}: {e}")
                continue
    return pages

if __name__ == "__main__":
    all_pages = set()

    print("Extracting from IIRC...")
    iirc_pages = set()
    iirc_pages |= extract_from_iirc(IIRC_DIR / "train.json")
    iirc_pages |= extract_from_iirc(IIRC_DIR / "dev.json")
    print(f"→ IIRC unique pages: {len(iirc_pages):,}")

    print("\nExtracting from MuSiQue...")
    musique_pages = set()
    musique_pages |= extract_from_musique(MUSIQUE_DIR / "train.jsonl")
    musique_pages |= extract_from_musique(MUSIQUE_DIR / "dev.jsonl")
    print(f"→ MuSiQue unique pages: {len(musique_pages):,}")

    all_pages = iirc_pages.union(musique_pages)
    print(f"\nTotal unique Wikipedia titles: {len(all_pages):,}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for title in sorted(all_pages):
            f.write(json.dumps({"title": title}) + "\n")

    print(f"Saved → {OUTPUT_PATH}")