# embed_and_index.py
# Script embed chunks Wikipedia → FAISS index
# Đã sửa key từ 'content' → 'text' dựa trên sample chunk của bạn

import os
import json
import warnings
import numpy as np
from functools import lru_cache

# Suppress NumPy warnings (MinGW on Windows)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", message="Numpy built with MINGW-W64")

# Fix OpenMP duplicate (nếu có torch/MKL conflict)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Script started - warnings suppressed and env set")

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# ================= CONFIG =================
MODEL_NAME = 'all-MiniLM-L6-v2'          # Nhỏ, nhanh để test
# MODEL_NAME = 'all-mpnet-base-v2'       # Chất lượng cao hơn, dùng sau khi test OK

CHUNKS_FILE    = 'D:/Project/Multi-hop-QA/data/processed/full_wiki_chunks.jsonl'
INDEX_FOLDER   = 'faiss_index_wiki'
BATCH_SIZE     = 128                   # Tùy CPU/RAM, có thể giảm xuống 16 nếu RAM thấp
# ==========================================

# ================= BƯỚC 1: LOAD CHUNKS =================
print("Loading chunks from file...")
chunks = []
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            chunk = json.loads(line.strip())
            if 'text' not in chunk:
                print(f"WARNING: Chunk at line {line_num} missing 'text' key! Skipping...")
                continue
            chunks.append(chunk)
        except json.JSONDecodeError:
            print(f"Error decoding JSON at line {line_num}, skipping...")

print(f"Loaded {len(chunks):,} valid chunks")
chunks = chunks[:50000]

# Kiểm tra sample chunk đầu tiên
if chunks:
    print("Sample chunk keys:", list(chunks[0].keys()))
    print("Sample text snippet:", chunks[0]['text'][:150] + "...")

# ================= BƯỚC 2: KHỞI TẠO MODEL =================
print(f"\nInitializing SentenceTransformer model: {MODEL_NAME}")
print("(May take 1-5 minutes first time due to download)")
model = SentenceTransformer(MODEL_NAME, model_kwargs={"torch_dtype": "float16"})

print("Model loaded successfully!")

# ================= BƯỚC 3: EMBEDDING (BATCH + SEQUENTIAL) =================
print("\nStarting embedding (sequential + batch mode for stability)...")

embedded_chunks = []
embeddings_list = []

for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
    batch = chunks[i:i + BATCH_SIZE]
    batch_texts = [chunk['text'] for chunk in batch]

    try:
        batch_embs = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True
        )
    except Exception as e:
        print(f"Error encoding batch starting at {i}: {e}")
        continue

    for j, (chunk, emb) in enumerate(zip(batch, batch_embs)):
        chunk['embedding'] = emb.tolist()  # Lưu dưới dạng list để dễ JSON nếu cần sau
        embedded_chunks.append(chunk)
        embeddings_list.append(emb)

    processed = i + len(batch)
    if processed % 1000 == 0:
        print(f"  → Processed {processed:,} / {len(chunks):,} chunks ({processed/len(chunks)*100:.1f}%)")

print(f"Finished embedding {len(embedded_chunks):,} chunks")

# Chuẩn bị dữ liệu cho FAISS
texts = [chunk['text'] for chunk in embedded_chunks]
embeddings = np.array(embeddings_list).astype('float32')
metadatas = [
    {
        'chunk_id': chunk['chunk_id'],
        'title': chunk['title'],
        'url': chunk['url'],
        'section_title': chunk.get('section_title', ''),
        'source': chunk.get('source', '')
    }
    for chunk in embedded_chunks
]

print(f"Embeddings shape: {embeddings.shape}")
if texts:
    print(f"Sample text: {texts[0][:150]}...")

# ================= BƯỚC 4: TẠO VÀ LƯU FAISS INDEX =================
print("\nCreating FAISS vectorstore...")
embeddings_func = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},  # 'cuda' nếu có GPU
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings_func,
    metadatas=metadatas,
    ids=[meta['chunk_id'] for meta in metadatas]
)

print(f"Indexed {vectorstore.index.ntotal:,} chunks successfully")

print(f"Saving FAISS index to: {INDEX_FOLDER}")
vectorstore.save_local(INDEX_FOLDER)

print("\n=== HOÀN TẤT ===\nFAISS index đã sẵn sàng!")
print(f"Folder lưu: {INDEX_FOLDER}")
print("Để load lại sau này:")
print(f"vectorstore = FAISS.load_local('{INDEX_FOLDER}', embeddings_func, allow_dangerous_deserialization=True)")