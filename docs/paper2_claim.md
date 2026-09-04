# ContextOpti — Paper-2 Claim (locked)

**Status:** locked as of M0. Changes to this file require a corresponding README update.

---

## 1. The claim

> **Structure-conditioned selective and budgeted context selection.**
>
> Given an unfinished code region in a repository, decide **whether** to retrieve at all,
> **where** on the repository structure/dependency graph to retrieve from (graph locus and
> hop radius), and **how much** to retrieve (token budget) — under fixed IDE latency and
> token budgets — and beat both (a) always-on hybrid/structure RAG and (b) selective
> policies that operate over plain text chunks with no graph.

The contribution is the **policy** `s -> (retrieve, hops, tok)` conditioned on graph state,
not the retriever and not the ranker.

### Why this and not "hybrid retrieval"

The originally proposed novelty — *hybrid semantic + structural + dependency + data-flow
retrieval* — is already well covered by published work (see §5). A system that fuses those
four signals with tuned weights is an engineering combination of known parts, and reviewers
will read it that way. What is **not** settled is the control problem sitting on top of any
such retriever: retrieval is not free, it is not always useful, and the right amount of it
varies per completion request. That decision problem is where ContextOpti stakes its claim.

Hybrid signals remain in the codebase as **optional features and ablations** — they are
inputs to the policy and rows in the results table, never the headline.

---

## 2. Research questions

**RQ1 — Does selectivity pay off on a code graph?**
Under a fixed token budget, does a policy that *skips* retrieval on a subset of completion
requests match or beat always-on structure retrieval on completion quality while spending
strictly fewer retrieval calls and tokens?

**RQ2 — Does graph state carry the decision signal?**
Is a policy conditioned on ego-graph features of the cursor (in/out degree of the enclosing
entity, unresolved references, cross-file call/import edges, local-context coverage of the
referenced symbols) better at the whether/how-much decision than an equivalent selective
policy over plain chunks with no structural features? This is the isolation experiment that
separates ContextOpti from RepoFormer-style selective retrieval.

**RQ3 — Is the budget worth conditioning on state?**
Does per-request allocation of `(hops, tok)` dominate the best single fixed `(hops, tok)`
setting on the quality-vs-tokens Pareto frontier, or does one well-tuned constant suffice?
A negative result here is publishable and must be reported as such.

**RQ4 — Where does the policy's advantage come from?**
Which state features drive the decisions, and on which task categories (cross-file call,
local completion, unresolved import, intra-function) does selectivity help versus hurt?

---

## 3. Baselines

Every number in the main table comes from the same frozen generator and the same task set.

| # | System | Retrieval | Purpose |
|---|--------|-----------|---------|
| 1 | No retrieval | none — local context only | floor; also the "skip" arm of the policy |
| 2 | Semantic only | top-k chunks, always on | classic RAG baseline (README Phase 1) |
| 3 | Always-on structure | graph, fixed hops + fixed tok | shows structure helps, and its cost |
| 4 | Selective over chunks | semantic, gated by a no-graph heuristic | **RepoFormer-style proxy — the key contrast for RQ2** |
| 5 | **ContextOpti** | graph, policy-chosen `(retrieve, hops, tok)` | the claim |
| 6 | Always-on hybrid *(ablation)* | semantic + structure fused | evidence that hybrid alone is not the story |

Baseline 4 is not optional. Without it the claim collapses to "selective retrieval works",
which is already published.

---

## 4. Explicit non-claims

ContextOpti does **not** claim to be:

- the first hybrid semantic + structural retrieval system;
- the first selective / adaptive retrieval system (RepoFormer, RLCoder, Self-RAG);
- the first structure-aware or data-flow-aware retrieval system for code (DraCo, GraphCoder);
- a SWE-bench-style repository agent — agentic exploration remains future scope;
- a contribution to code generation modeling — the generator is frozen and treated as a
  black box in all policy experiments.

Stating these up front is deliberate. The claim is narrow on purpose so that it is defensible.

---

## 5. Near-miss related work

To cite and position against in the paper. These are **not** to be reimplemented in full;
the proxies in §3 stand in for them.

| Work | What it does | How ContextOpti differs |
|------|--------------|-------------------------|
| **RepoFormer** | Learns *whether* to retrieve for repo-level completion | Selective, but over plain chunks; no graph locus, no budget allocation |
| **RLCoder** | RL-trained retriever for repo completion, no labels | Optimizes *what* to retrieve; retrieval stays always-on |
| **DraCo** | Dataflow-guided repository context | Always-on, hand-built dataflow retrieval; no decision policy |
| **GraphCoder** | Code-context graph + structural retrieval | Always-on graph retrieval; fixed budget |
| **RepoFuse / CodeRAG / AIRCoder / GRACE / Hydra** | Hybrid semantic + structural fusion | Fusion and ranking; orthogonal to the whether/where/how-much decision |
| **RepoShapley** | Attributes value of retrieved context post hoc | Analysis tool; ContextOpti decides *before* paying the cost |
| **Self-RAG / adaptive RAG (NL)** | Adaptive retrieval in open-domain QA | No repository structure; different cost model and latency regime |

The honest summary of the gap: selectivity has been done over chunks, structure has been
done always-on, and nobody has conditioned the selectivity **on** the structure while also
allocating the budget.

---

## 6. What would falsify the claim

The experiments must be able to come back negative. Specifically:

- **RQ1 falsified** if always-on structure retrieval matches the policy at equal or lower
  token cost across the budget sweep.
- **RQ2 falsified** if selective-over-chunks (baseline 4) matches ContextOpti — that would
  mean the graph features add nothing to the *decision*, only to the retrieval.
- **RQ3 falsified** if a single tuned `(hops, tok)` sits on or above the policy's Pareto
  frontier everywhere.

If RQ2 falsifies, the paper becomes a negative result about graph-conditioned selectivity,
which is still worth writing — but the headline claim changes and this document gets rewritten.

---

## 7. Evaluation protocol

- **Generator frozen** across all policy experiments. Any generator change invalidates the
  table and requires a full re-run of every row.
- **Same task set, same seed** for every row; `max_eval_n` and `seed` live in
  `services/ai/configs/default.yaml`.
- **Primary axes:** completion quality (EM / edit-similarity) versus context tokens and
  end-to-end latency, reported as a Pareto frontier over a budget sweep — not as single
  points, which are easy to cherry-pick.
- **Secondary:** retrieval rate (fraction of requests where the policy chose to retrieve),
  mean hops, mean tokens, and per-category breakdowns for RQ4.
- **Data:** CrossCodeEval-style cross-file completion tasks. If the dataset cannot be
  obtained, synthetic cross-file tasks generated from `data/fixtures/` are used and the
  substitution is documented prominently in the results — never silently.

---

## 8. Milestones backing the claim

| Milestone | Produces | Answers |
|-----------|----------|---------|
| M1 graph index | entity graph over a cross-file repo | prerequisite |
| M2 semantic baseline | `outputs/m2_semantic.csv` | baseline 2 |
| M3 always-on structure | `outputs/m3_structure.csv` | baseline 3 |
| M4 selective policy | `outputs/m4_decisions.jsonl` | RQ1, RQ3 |
| M5 main table | `outputs/m5_main_table.md` | RQ2, RQ4, full comparison |

---

## 9. Scope note: language

M1–M5 run on **Python** fixtures, using the standard-library `ast` module. This is a
deliberate scope reduction from the README's original JavaScript/TypeScript-first plan:
the research claim is about the *policy*, and a language with a batteries-included parser
gets to the policy experiments fastest. Tree-sitter-based JS/TS parsing is a follow-up
milestone (M6) once the selective policy is validated; the graph schema in
`contextopti/index/` is language-agnostic so that the parser is the only piece that changes.
