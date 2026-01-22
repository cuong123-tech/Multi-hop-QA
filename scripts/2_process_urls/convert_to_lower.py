import json
from pathlib import Path

unique_file = Path("data/urls/unique_pages_iirc_musique.jsonl")
plain_titles = Path("data/urls/unique_titles_lower.txt")

with open(unique_file, 'r', encoding='utf-8') as f_in, open(plain_titles, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        obj = json.loads(line.strip())
        title = obj["title"].replace("_", " ").strip().lower()
        f_out.write(title + "\n")