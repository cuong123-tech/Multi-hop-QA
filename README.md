# Multi-Hop RAG Dataset Generation on Wikipedia

Tạo dataset câu hỏi multi-hop chất lượng cao từ Wikipedia để train/eval RAG models (dựa trên ý tưởng HotpotQA / MuSiQue).
Building a multi-doc QA datasetby retrieving similar wikipedia articles via elasticsearch, sentence transformer

## Mục tiêu
- Ingestion ~10k–20k Wikipedia articles/paragraphs
- Generate 1k–5k multi-hop questions needing reasoning by using 2-4 documents
- Evaluation (manual + LLM-as-judge)
- Demo RAG chain to answer multi-hop

## Cấu trúc dự án

## Cài đặt nhanh (5 phút)

1. Clone repo
   ```bash
   git clone https://github.com/cuong123-tech/Multi-hop-QA.git
   cd Multi-hop-QA