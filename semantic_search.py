import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)
    return records


def get_texts(records: List[Dict], use_trans: bool = True) -> Tuple[List[str], List[Tuple[str, str]]]:
    texts = []
    meta = []  # (id, field)
    for r in records:
        rid = str(r.get("id", ""))
        if use_trans and r.get("trans"):
            texts.append(str(r["trans"]))
            meta.append((rid, "trans"))
        else:
            if r.get("trans"):
                texts.append(str(r["trans"]))
                meta.append((rid, "trans"))
            if r.get("orig"):
                texts.append(str(r["orig"]))
                meta.append((rid, "orig"))
    return texts, meta


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def build_embeddings(model_name: str, texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=64, show_progress_bar=False)
    emb = np.array(emb, dtype=np.float32)
    emb = normalize(emb)
    return emb


def pool_queries(model_name: str, queries: List[str]) -> np.ndarray:
    model = SentenceTransformer(model_name)
    q_emb = model.encode(queries, batch_size=32, show_progress_bar=False)
    q_emb = np.array(q_emb, dtype=np.float32)
    q_emb = normalize(q_emb)
    pooled = np.mean(q_emb, axis=0, keepdims=True)
    pooled = normalize(pooled)
    return pooled


def cosine_topk(doc_emb: np.ndarray, q_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    sims = doc_emb @ q_emb.T
    sims = sims.reshape(-1)
    if k >= sims.shape[0]:
        order = np.argsort(-sims)
    else:
        order = np.argpartition(-sims, k)[:k]
        order = order[np.argsort(-sims[order])]
    scores = sims[order]
    return order, scores


def accept_patterns() -> Tuple[List[str], List[str]]:
    positives = [
        "接受你的友谊",
        "我接受你的朋友邀请",
        "很高兴成为你的朋友",
        "我们做朋友吧",
        "愿意和你做朋友",
        "乐意交个朋友",
        "我愿意和你成为朋友",
        "成为朋友我很高兴",
        "同意当朋友",
        "交个朋友吧",
        "当然可以做朋友",
    ]
    negatives = [
        "不想做朋友",
        "拒绝你的朋友邀请",
        "不愿意和你做朋友",
        "算了吧",
        "以后再说",
        "不接受",
    ]
    return positives, negatives


def classify_text(text: str) -> str:
    pos, neg = accept_patterns()
    t = text.strip()
    for n in neg:
        if n in t:
            return "RejectFriendship"
    for p in pos:
        if p in t:
            return "AcceptFriendship"
    return "Neutral"


def load_queries(path: str) -> List[str]:
    """Load queries from a file, split by semicolons or newlines."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Split on semicolons or any newline variant
    import re
    parts = re.split(r"[;\r\n]+", content)
    queries = [p.strip() for p in parts if p.strip()]
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for q in queries:
        # Ignore comment lines starting with '//'
        if q.startswith("//"):
            continue
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default=os.path.join("Sherry", "sherry_extracts.jsonl"))
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--query", type=str, default="我接受你的友谊，很高兴成为你的朋友")
    parser.add_argument("--query_file", type=str, help="Path to a file containing queries separated by ';' or newlines")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_trans_only", action="store_true")
    parser.add_argument("--no_expand", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(args.jsonl)
    texts, meta = get_texts(records, use_trans=args.use_trans_only)

    model_name = args.model
    doc_emb = build_embeddings(model_name, texts)

    # Determine sources of queries
    if args.query_file:
        query_list = load_queries(args.query_file)
    else:
        query_list = [args.query]

    # Ignore commented queries starting with '//'
    query_list = [q for q in query_list if not q.strip().startswith("//")]
    if not query_list:
        print("No valid queries after filtering comments.")
        return

    # Process each query independently and print grouped results
    for base_query in query_list:
        query_variants = [base_query]
        if not args.no_expand:
            # Optional expansion using built-in acceptance patterns
            pos, _ = accept_patterns()
            query_variants = list(set([base_query] + pos))

        q_emb = pool_queries(model_name, query_variants)
        idxs, scores = cosine_topk(doc_emb, q_emb, args.top_k * 2)

        results = []
        for i, s in zip(idxs, scores):
            rid, field = meta[i]
            text = texts[i]
            label = classify_text(text)
            results.append({
                "query": base_query,
                "id": rid,
                "field": field,
                "score": float(s),
                "label": label,
                "text": text,
            })

        # Filter, deduplicate by (id, field), and sort
        results = [r for r in results if r["score"] >= args.threshold]
        seen = set()
        dedup = []
        for r in results:
            key = (r["id"], r["field"])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(r)

        dedup = sorted(dedup, key=lambda x: -x["score"])[: args.top_k]

        # Print a header per query and its top matches
        print(f"\n=== Query: {base_query} ===")
        for r in dedup:
            preview = r["text"][:120].replace("\n", " ")
            print(
                f"id={r['id']} field={r['field']} score={r['score']:.3f} label={r['label']} text={preview}..."
            )


if __name__ == "__main__":
    main()
