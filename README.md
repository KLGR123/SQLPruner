### SQLPruner

Lightweight tools to prune large DB schemas for Text-to-SQL:
- Hybrid recall: BM25 (sparse) + BGE-M3 (dense) → **RRF** fusion
- Optional rerank: **BGE cross-encoder**
- Pruning with per-table caps and (optional) minimal FK-connected subgraph
- **Minimal mode**: whitelist columns and print the compact schema—no graph logic

#### Install
```bash
pip install rank-bm25 sentence-transformers transformers torch
```
(If you only need minimal mode, rank-bm25 and the heavy deps are optional.)

#### Quickstart
```
from schema_pruner import (
  parse_schema_text, prune_schema, format_schema_text,
  PruneConfig, prune_schema_min, format_schema_min
)

question = "What is the highest eligible free rate for K-12 students in Alameda County?"
evidence = "Eligible free rate = `Free Meal Count (K-12)` / `Enrollment (K-12)`"

schema_text = """... your schema text ..."""
schema = parse_schema_text(schema_text)

# Full pipeline
cfg = PruneConfig(dense_model="BAAI/bge-m3", reranker_model="BAAI/bge-reranker-base")
pruned, _ = prune_schema(question, evidence, schema, cfg)
print(format_schema_text(pruned))

# Minimal mode (no graph logic)
kept_cols = {"frpm": ["Free Meal Count (K-12)", "Enrollment (K-12)"], "schools": ["County", "School", "CDSCode"]}
pruned_min = prune_schema_min(schema, kept_cols)
print(format_schema_min(pruned_min))
```

#### Offline tip

Download once, then load locally:
`
huggingface-cli download BAAI/bge-m3 --local-dir /models/bge-m3 --local-dir-use-symlinks False
`
```
import os; os.environ["HF_HUB_OFFLINE"]="1"
from sentence_transformers import SentenceTransformer
SentenceTransformer("/models/bge-m3", local_files_only=True)
```