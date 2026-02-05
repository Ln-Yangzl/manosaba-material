## Usage

- Run with a single query:

```bash
python semantic_search.py --jsonl Sherry/sherry_extracts.jsonl --query "我接受你的友谊，很高兴成为你的朋友"
```

- Run with multiple queries from a file (separated by semicolons or newlines):

```bash
python semantic_search.py --jsonl Sherry/sherry_extracts.jsonl --query_file queries.txt --top_k 10 --threshold 0.5
```

- Disable built-in expansion (uses acceptance patterns) to only search with provided queries:

```bash
python semantic_search.py --jsonl Sherry/sherry_extracts.jsonl --query_file queries.txt --no_expand
```

The script will print grouped results per query, showing `id`, `field`, cosine `score`, `label`, and a preview of the matched text.
# Local Semantic Search for Friendship Acceptance

This sample reads JSONL (`Sherry/sherry_extracts.jsonl`) with per-line objects having keys: `id`, `orig` (Japanese), `trans` (Chinese). It runs local multilingual embeddings and cosine similarity to find texts that accept friendship.

## Install

Windows PowerShell:

```powershell
# From workspace root
python -m pip install -r requirements.txt
```

If you don't have `python` in PATH, use your environment's interpreter.

## Run

```powershell
# Basic run with query (Chinese)
python semantic_search.py --use_trans_only --query "我接受你的友谊，很高兴成为你的朋友" --top_k 10 --threshold 0.5

# Use all fields (Chinese + Japanese)
python semantic_search.py --query "我接受你的友谊，很高兴成为你的朋友" --top_k 10 --threshold 0.5

# Disable query expansion (exact intent only)
python semantic_search.py --use_trans_only --no_expand --query "我接受你的友谊" --top_k 10 --threshold 0.55

# Switch to a stronger but heavier model (downloads larger weights)
python semantic_search.py --model "BAAI/bge-m3" --use_trans_only --query "我们做朋友吧" --top_k 10 --threshold 0.5
```

## Notes
- Default model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` is small and supports Chinese/Japanese; for better accuracy use `BAAI/bge-m3` (larger download).
- Cosine similarity with normalized embeddings; increase `--threshold` for stricter results.
- A simple pattern-based label is included: `AcceptFriendship` / `RejectFriendship` / `Neutral`.
- For large datasets, consider FAISS for faster indexing; this demo uses NumPy for simplicity.

E:/file/素材/manosaba/textWorkSpace/.venv/Scripts/python.exe semantic_search.py --jsonl Sherry/sherry_extracts.jsonl --query_file queries.txt --top_k 5 --threshold 0.3 --no_expand --use_trans_only