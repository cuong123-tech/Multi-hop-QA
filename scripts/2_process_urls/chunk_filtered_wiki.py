import json
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=70,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator=True  
)

INPUT_JSONL = Path("C:/Users/Admin/Project/Multi-hop-QA/data/processed/filtered_wiki.jsonl")
OUTPUT_CHUNKS = Path("C:/Users/Admin/Project/Multi-hop-QA/data/processed/full_wiki_chunks.jsonl")

# Tạo thư mục nếu chưa có
OUTPUT_CHUNKS.parent.mkdir(parents=True, exist_ok=True)

total_chunks = 0

with open(INPUT_JSONL, 'r', encoding='utf-8') as f_in, \
     open(OUTPUT_CHUNKS, 'w', encoding='utf-8') as f_out:

    for line_num, line in enumerate(tqdm(f_in, desc="Chunking filtered Wikipedia")):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
            title = doc.get('title', 'NoTitle')
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

            section_titles = doc.get('section_titles', [])
            section_texts = doc.get('section_texts', [])

            if not section_texts:
                continue

            for sec_idx, (sec_title, sec_text) in enumerate(zip(section_titles, section_texts)):
                # Clean text cơ bản: loại bỏ nhiều \n đầu/cuối
                sec_text = sec_text.strip()
                if len(sec_text) < 100:
                    continue

                chunks = text_splitter.split_text(sec_text)

                for chunk_idx, chunk in enumerate(chunks):
                    chunk_data = {
                        "chunk_id": f"wiki_{title.replace(' ', '_')}_sec{sec_idx}_{chunk_idx}",
                        "title": title,
                        "section_title": sec_title,
                        "url": url,
                        "text": chunk.strip(),
                        "source": "wikipedia_gensim_filtered",
                        "section_index": sec_idx,
                        "chunk_in_section": chunk_idx,
                        "text_length": len(chunk)
                    }
                    f_out.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
                    total_chunks += 1

        except json.JSONDecodeError:
            print(f"Skip invalid JSON at line {line_num+1}")
        except Exception as e:
            print(f"Error at line {line_num+1}: {e}")

print(f"\nChunking completed!")
print(f"Total chunks: {total_chunks:,}")
print(f"File output: {OUTPUT_CHUNKS}")