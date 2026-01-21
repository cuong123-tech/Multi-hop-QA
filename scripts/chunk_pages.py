# scripts/3_chunk_pages.py
"""
Chunk passages từ IIRC và MuSiQue (nếu dataset có field passages/context/paragraphs)
Lưu chunks vào data/processed/chunks.jsonl
"""
import json
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_ROOT = Path("data")
PROCESSED_DIR = DATA_ROOT / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = PROCESSED_DIR / "chunks.jsonl"

IIRC_DIR = DATA_ROOT / "raw" / "iirc"
MUSIQUE_DIR = DATA_ROOT / "raw" / "musique"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=70,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def process_dataset(file_path: Path, dataset_name: str):
    if not file_path.exists():
        print(f"Skip: {file_path}")
        return

    chunks_written = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        if file_path.suffix == ".jsonl":
            with open(file_path, encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Chunking {dataset_name} {file_path.name}"):
                    ex = json.loads(line)
                    # Thử lấy passages / context
                    passages = ex.get("paragraphs", []) or ex.get("context", []) or ex.get("text", [])
                    if not passages:
                        continue

                    for line in tqdm(f, desc=f"Chunking {file_path.name}"):
                        try:
                            ex = json.loads(line.strip())
                            paragraphs = ex.get("paragraphs", []) or ex.get("context", [])
                            
                            for para in paragraphs:
                                title = para.get("title", "NoTitle").strip().replace(" ", "_").replace("/", "_")
                                
                                # Lấy text an toàn
                                raw_text = para.get("paragraph_text") or para.get("text") or para.get("content") or ""
                                
                                # Xử lý nếu raw_text là dict (lỗi phổ biến)
                                if isinstance(raw_text, dict):
                                    raw_text = raw_text.get("text", "") or raw_text.get("paragraph_text", "") or str(raw_text)
                                
                                if not isinstance(raw_text, str):
                                    raw_text = str(raw_text)
                                
                                text = raw_text.strip()
                                
                                if len(text) < 50:
                                    continue
                                
                                chunks = text_splitter.split_text(text)
                                for i, chunk in enumerate(chunks):
                                    chunk_data = {
                                        "chunk_id": f"{dataset_name}_{title}_{i}",
                                        "title": title.replace("_", " "),
                                        "text": chunk.strip(),
                                        "source": dataset_name,
                                        "example_id": ex.get("id", "no_id"),
                                        "is_supporting": para.get("is_supporting", False)  # thêm nếu muốn
                                    }
                                    out_f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
                                    chunks_written += 1
                                    
                        except Exception as e:
                            print(f"Lỗi xử lý dòng trong {file_path.name}: {str(e)}")
        else:
            # .json (IIRC)
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            for ex in tqdm(data, desc=f"Chunking {dataset_name} {file_path.name}"):
                # Tương tự logic trên...
                # (copy logic chunking nếu cần)

                print(f"→ Wrote {chunks_written:,} chunks from {file_path.name}")

if __name__ == "__main__":
    print("Starting chunking...")

    process_dataset(IIRC_DIR / "train.json", "IIRC")
    process_dataset(IIRC_DIR / "dev.json", "IIRC")
    process_dataset(MUSIQUE_DIR / "train.jsonl", "MuSiQue")
    process_dataset(MUSIQUE_DIR / "dev.jsonl", "MuSiQue")

    print(f"\nAll chunks saved to: {OUTPUT_PATH}")