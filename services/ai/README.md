# ContextOpti research core

Python implementation of the research claim in
[`docs/paper2_claim.md`](../../docs/paper2_claim.md): **structure-conditioned selective
and budgeted context selection**.

## Setup

```bash
# from the repository root
python -m venv .venv
.venv/Scripts/python -m pip install -e services/ai[dev]     # Windows
# source .venv/bin/activate && pip install -e services/ai[dev]   # Linux / macOS
```

`networkx` and `pyyaml` are the only hard dependencies. No Postgres, pgvector, Redis, or
Neo4j is required for M1-M5.

## Milestones

| Milestone | Command | Output |
|-----------|---------|--------|
| M1 graph index | `python services/ai/scripts/m1_build_graph.py` | `outputs/m1_graph.json` |
| M2 semantic baseline | `python services/ai/scripts/m2_semantic_baseline.py` | `outputs/m2_semantic.csv` |
| M3 always-on structure | `python services/ai/scripts/m3_always_on_structure.py` | `outputs/m3_structure.csv` |
| M4 selective policy | `python services/ai/scripts/m4_selective_policy.py` | `outputs/m4_decisions.jsonl` |
| M5 main table | `python services/ai/scripts/m5_pareto_and_table.py` | `outputs/m5_main_table.md` |

Milestones M2-M5 are not implemented yet. M1 is.

## Tests

```bash
.venv/Scripts/python -m pytest services/ai/tests -q
```

Tests never touch the network; the LLM provider is mocked.

## Layout

```text
contextopti/
  index/      AST parsing + code graph          (M1, implemented)
  embed/      chunking + embeddings             (M2)
  state/      cursor + ego-graph features       (M4)
  policy/     retrieve? hops? tok?              (M4, the contribution)
  retrieve/   structural + semantic retrieval   (M2/M3)
  rank/       hybrid weighting                  (M5, ablation only)
  optimize/   token-budget assembly             (M3)
  generate/   LLM provider abstraction          (M2, frozen in experiments)
  train/      policy fitting                    (M4)
  eval/       metrics harness                   (M5)
```

Configuration lives in [`configs/default.yaml`](configs/default.yaml); override the path
with the `CONTEXTOPTI_CONFIG` environment variable.
