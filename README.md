# ContextOpti

### Structure-Conditioned Selective Context Selection for LLM-Based Code Completion

ContextOpti is a repository-aware AI code completion system built around a single question that
sits *on top of* retrieval rather than inside it:

> **Retrieval is not free. Given an unfinished piece of code in a repository, should we retrieve
> repository context at all — and if so, from where on the code graph, and how much of it?**

Most repository-level code assistants retrieve on every request, at a fixed budget. But many
completions need nothing beyond the current file, and among those that do need cross-file
context, the useful amount varies enormously from request to request. Retrieving anyway costs
tokens, latency, and money, and can actively degrade the completion by burying the local context
in noise.

ContextOpti learns to make that decision using the **structure of the repository itself** — the
entity, import, call, and data-flow graph around the cursor — as the state it conditions on.

The project combines **AI engineering, code intelligence, information retrieval, and research
experimentation** into a single system.

---

## Research Problem

Given a code-completion request inside a large repository, a retrieval system must answer three
questions before it can help:

1. **Whether** cross-file context is needed at all for this request.
2. **Where** to look — which region of the repository graph, and how many hops out from the cursor.
3. **How much** to bring back — the token budget to spend on this particular completion.

Always-on retrieval answers all three with constants. ContextOpti answers them per request, using
repository structure as evidence.

### Research Question

> **Can a policy conditioned on repository graph state around the cursor decide whether to
> retrieve, where on the graph to retrieve from, and how large a token budget to spend — such
> that, under fixed IDE latency and token budgets, it outperforms both always-on structural /
> hybrid retrieval and selective retrieval policies that operate over plain text chunks?**

The full claim, research questions, baselines, and falsification conditions are locked in
[`docs/paper2_claim.md`](docs/paper2_claim.md).

### What this project does *not* claim

The earlier framing of this project — *hybrid semantic + structural + dependency + data-flow
retrieval* — is **already published** work (RepoFuse, GraphCoder, CodeRAG, AIRCoder, GRACE,
Hydra, and others). It is retained here as a set of **optional features and ablations**, not as
the research contribution. Specifically, ContextOpti does not claim to be:

- the first hybrid semantic + structural retrieval system;
- the first selective / adaptive retrieval system (RepoFormer, RLCoder, Self-RAG);
- the first structure- or data-flow-aware retrieval system for code (DraCo, GraphCoder);
- a SWE-bench-style repository agent — agentic behaviour is future scope only.

---

## Core Idea

Instead of retrieving on every request at a fixed budget:

```text
Code Completion Request
        ↓
Semantic Search              ← always on, fixed k
        ↓
Top-K Code Chunks
        ↓
LLM
        ↓
Completion
```

ContextOpti puts a **decision** in front of retrieval, and conditions that decision on the
repository graph:

```text
Code Completion Request
        ↓
Encode State  ←──────  Ego-graph around the cursor
        │              (entity degree, unresolved refs,
        │               cross-file edges, local coverage)
        ↓
   ┌────────────────────────────────┐
   │   Policy:  retrieve? hops? tok?│   ← the research contribution
   └────────────────────────────────┘
        │
   ┌────┴──────────────────┐
   │                       │
  no                      yes
   │                       ↓
   │            Structure Retrieval (hops)
   │                       │
   │            + Semantic  (optional, ablation)
   │                       ↓
   │            Context Optimization (tok)
   │                       │
   └───────────┬───────────┘
               ↓
              LLM
               ↓
         Code Completion
```

The goal is **not to retrieve more context**, and not even simply to retrieve *better* context.

The goal is to **spend retrieval only where it pays for itself** — and to prove that repository
structure is what tells you where that is.

### The inference loop

```text
Input: unfinished code X, repo graph G, budgets (T_max, L_max)

s <- EncodeState(X, Ego(G, X))
(retrieve, hops, tok) <- Policy(s)           # ContextOpti core
if not retrieve:
    return Generate(X)                       # local context only

C_struct <- StructureRetrieve(G, X, hops)
C_sem    <- SemanticRetrieve(X)              # optional; ablation only
C        <- OptimizeContext(merge(C_struct, C_sem), tok)
Y        <- Generate(prompt(X, C))
return Y
```

---

## Quick Start

The research core is Python and has two hard dependencies (`networkx`, `pyyaml`). No
Postgres, pgvector, Redis, Neo4j, or Node toolchain is needed for M1–M5.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e services/ai[dev]      # Windows
# source .venv/bin/activate && pip install -e services/ai[dev]    # Linux / macOS

# M1 — build the repository graph over the toy cross-file fixture
.venv/Scripts/python services/ai/scripts/m1_build_graph.py

# inspect a single relationship type
.venv/Scripts/python services/ai/scripts/m1_build_graph.py --show-edges calls

# index any other Python repository
.venv/Scripts/python services/ai/scripts/m1_build_graph.py --repo path/to/repo

# tests (no network; the LLM provider is mocked)
.venv/Scripts/python -m pytest services/ai/tests -q
```

## Repository Layout

```text
docs/paper2_claim.md          locked research claim, RQs, baselines, non-claims
services/ai/                  the research core (Python)
  contextopti/index/            AST + code graph            (M1, implemented)
  contextopti/state/            ego-graph state encoding    (M4)
  contextopti/policy/           retrieve? hops? tok?        (M4, the contribution)
  contextopti/retrieve/         structural + semantic       (M2/M3)
  contextopti/rank/             hybrid weighting            (M5, ablation only)
  contextopti/optimize/         token-budget assembly       (M3)
  contextopti/generate/         LLM provider, frozen        (M2)
  contextopti/eval/             metrics harness             (M5)
  scripts/                      m1…m5 milestone entry points
  configs/default.yaml          budgets, action space, seed, paths
  tests/                        pytest suite
data/fixtures/toy_repo/       cross-file Python fixture (layered shop app)
outputs/                      experiment artifacts (gitignored — regenerate them)
apps/                         product path: Node API + React/Monaco (phases 7+, not built)
```

---

# Key Features

## 1. Repository Indexing

Users can provide a repository for analysis.

The indexing pipeline extracts:

* source files
* functions
* classes
* methods
* variables
* imports
* exports
* function calls
* references
* dependencies
* code relationships

---

## 2. Semantic Code Retrieval

Repository code is converted into embeddings and indexed for semantic retrieval.

Given a completion request, the system retrieves semantically relevant code candidates.

This provides the project's **baseline RAG system**.

---

## 3. Structural Code Retrieval

The system analyzes the structural relationships within the repository using AST-based code analysis.

Example:

```text
Controller
    ↓ calls
Service
    ↓ calls
Repository
```

This allows ContextOpti to retrieve code based on relationships rather than text similarity alone.

---

## 4. Dependency-Aware Retrieval

The system builds relationships between repository components.

For example:

```text
OrderController
       ↓
OrderService
       ↓
PaymentService
       ↓
PaymentRepository
```

Dependency information becomes an additional retrieval signal.

---

## 5. Data-Flow-Aware Retrieval

ContextOpti can identify how values move through the codebase.

Example:

```text
userId
  ↓
getUser()
  ↓
user.accountId
  ↓
getAccount()
  ↓
account.balance
```

Code participating in the relevant data flow can therefore receive a higher retrieval score.

---

## 6. Hybrid Retrieval *(ablation, not the contribution)*

> This was the project's original headline and is now a **supporting ablation** — it is
> already published work (RepoFuse, GraphCoder, CodeRAG, AIRCoder, GRACE, Hydra). It
> answers *what* to retrieve once the policy has decided to retrieve, which is orthogonal
> to the whether/where/how-much claim. It appears in the results as row 6 of the main table.

Multiple retrieval signals are combined:

```text
Semantic relevance
        +
Structural relevance
        +
Dependency relevance
        +
Data-flow relevance
```

A candidate can therefore be ranked using multiple sources of evidence.

Example scoring formulation:

```text
Score =
    α × Semantic
  + β × Structural
  + γ × Dependency
  + δ × Data-flow
```

The weights are experimental parameters and should be determined through evaluation rather than assumed beforehand.

---

## 7. Context Optimization

Retrieved candidates are ranked and filtered under a configurable context/token budget.

Example:

```text
Initial candidates
15 files
12,000 tokens
        ↓
Ranking + filtering
        ↓
Selected context
4 files
2,800 tokens
        ↓
LLM
```

The system measures whether context can be reduced without degrading completion quality.

---

## 8. LLM Code Completion

The optimized context is combined with:

* current file
* cursor position
* surrounding code
* repository context
* completion instructions

and supplied to an LLM.

The system then returns the generated completion.

---

# Research Evaluation

Every row below is run against the **same task set, the same seed, and the same frozen
generator**. Changing the generator invalidates the table and requires re-running every row.

### The main comparison table (`outputs/m5_main_table.md`)

| # | System | Retrieval behaviour | Role |
|---|--------|--------------------|------|
| 1 | No retrieval | local context only | floor — and the "skip" arm of the policy |
| 2 | Semantic only | top-k chunks, always on | classic RAG baseline |
| 3 | Always-on structure | graph, fixed hops + fixed tok | shows structure helps, and what it costs |
| 4 | Selective over **chunks** | semantic, gated by a no-graph heuristic | RepoFormer-style proxy |
| 5 | **ContextOpti** | graph, policy-chosen `(retrieve, hops, tok)` | **the claim** |
| 6 | Always-on hybrid *(ablation)* | semantic + structure fused | shows hybrid alone is not the story |

Row 4 is the load-bearing baseline. Beating rows 2 and 3 only shows that selectivity and
structure each help — both already known. Beating row 4 is what shows that conditioning the
*decision* on graph structure is the contribution.

Row 6 exists to make the negative point explicitly: if always-on hybrid retrieval matched the
policy, the original "hybrid ranker" framing would have been the right one. Reporting it is how
the paper earns the right to claim otherwise.

### Evaluation dimensions

**Completion quality**

* Exact Match
* Edit similarity (ES)
* CodeBLEU / pass@k where the task set supports it

**Context efficiency — reported jointly with quality, never alone**

* context tokens spent (retrieved and selected)
* retrieval rate — the fraction of requests where the policy chose to retrieve
* mean hops, mean token budget
* retrieval latency and end-to-end latency
* estimated inference cost

**Retrieval quality (diagnostic, not headline)**

* Recall@K, Precision@K, MRR, NDCG

### Reporting format

Results are reported as a **Pareto frontier over a budget sweep** — quality against tokens and
against latency — not as single operating points. A single point is easy to cherry-pick; the
frontier is what shows whether per-request allocation actually dominates a well-tuned constant.

A result where one fixed `(hops, tok)` setting sits on the policy's frontier everywhere is a
**negative result for RQ3**, and is reported as such rather than tuned around.

---

# Ablation Studies

The ablations answer *which part of the decision matters*, not which retrieval signals to fuse.

### Policy ablations (primary)

```text
A. Full policy                     (retrieve?, hops, tok)  ← ContextOpti
B. Gate only                       (retrieve?, fixed hops, fixed tok)
C. Budget only                     (always retrieve, policy hops + tok)
D. No graph features in state      (chunk-level state, same action space)
E. Random gate at matched rate     (control for "retrieving less is just cheaper")
```

Ablation D is the RQ2 isolation experiment. Ablation E is the control that stops a token
reduction from being mistaken for a decision quality improvement — a random gate that retrieves
at the same rate as the policy must do measurably worse.

### State-feature ablations

Which parts of the ego-graph carry the decision signal:

```text
- entity in/out degree
- unresolved reference count
- cross-file call / import edge count
- local-context coverage of referenced symbols
- data-flow reachability
```

### Retrieval-signal ablations (secondary — the old headline, demoted)

Retained for completeness and to support row 6 of the main table. These vary *what* is retrieved
once the policy has decided to retrieve, and are **not** the contribution:

```text
A. Structure only
B. Structure + semantic
C. Structure + dependency
D. Structure + dependency + data-flow
```

A complicated system is not automatically a better system. The experiments exist to find out
which components carry their own weight — including the possibility that some do not.

---

# System Architecture

```text
                         ┌─────────────────────┐
                         │      Web IDE        │
                         │ React + Monaco      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Node.js API      │
                         │     TypeScript      │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
              Repository       Completion      Evaluation
                Service          Service          Service
                    │               │
                    ▼               │
             Python AI Service     │
                    │               │
          ┌─────────┼─────────┐     │
          ▼         ▼         ▼     │
         AST      Graph    Data-flow│
          │         │         │     │
          └─────────┼─────────┘     │
                    ▼               │
             Hybrid Retrieval       │
                    ▼               │
              Context Ranking       │
                    ▼               │
             Context Optimizer       │
                    │               │
                    └───────┬───────┘
                            ▼
                           LLM
                            │
                            ▼
                     Code Completion
```

The initial implementation should use a **modular architecture rather than unnecessary microservices**.

---

# Technology Stack

## Frontend

* React
* TypeScript
* Monaco Editor

Monaco provides the editor experience required for a realistic developer tool, including syntax highlighting, cursor handling, file navigation, and editor integration.

## Backend

* Node.js
* TypeScript
* Express

Responsibilities:

* authentication
* project management
* repository management
* completion APIs
* request orchestration
* database communication

## AI / Code Intelligence

* Python
* FastAPI
* AST parser / Tree-sitter or language-specific compiler APIs
* embedding models
* retrieval and ranking algorithms
* evaluation tooling

## Database

* PostgreSQL
* pgvector

PostgreSQL stores application and repository metadata while pgvector provides vector similarity search.

## Infrastructure

* Redis
* BullMQ

Redis can support:

* caching
* rate limiting
* temporary state

BullMQ can process repository indexing asynchronously.

## LLM

The system should support an external LLM API such as:

* Gemini
* OpenAI
* Claude

The architecture should keep the LLM layer replaceable rather than tightly coupling the system to one provider.

---

# Development Roadmap

The roadmap has two layers. The **research milestones (M0–M5)** below are what the paper depends
on and are executed first. The **product phases (0–11)** further down describe the eventual
developer tool; phases 7–10 are deliberately blocked until the M5 results table exists.

**Research first. Product second.** The research result determines what deserves to become a
product feature.

---

## Research Milestones (M0–M5) — current work

All research code lives under [`services/ai/`](services/ai/) and is **Python**. Each milestone
ends in a commit-ready state, prints how to run it, and writes metrics under `outputs/`.

| Milestone | Goal | Script | Output |
|-----------|------|--------|--------|
| **M0** | Align docs to the selective × structure × budget claim | — | `docs/paper2_claim.md` |
| **M1** | Repository entity/dependency graph over a cross-file fixture | `m1_build_graph.py` | `outputs/m1_graph.json` |
| **M2** | Semantic RAG baseline (README Phase 1) | `m2_semantic_baseline.py` | `outputs/m2_semantic.csv` |
| **M3** | Always-on structure retrieval at fixed hops/budget | `m3_always_on_structure.py` | `outputs/m3_structure.csv` |
| **M4** | **Selective / budgeted policy — the paper contribution** | `m4_selective_policy.py` | `outputs/m4_decisions.jsonl` |
| **M5** | Main comparison table + quality/token Pareto sweep | `m5_pareto_and_table.py` | `outputs/m5_main_table.md` |

M4 ships a heuristic policy first, with a hook for a learned policy trained against
`quality - λ · cost`. The heuristic is the honest v1: if a heuristic over graph features already
beats always-on retrieval, that is the result, and the learned policy is an improvement on a
demonstrated effect rather than a search for one.

### Scope decisions for M1–M5

* **Python fixtures, not JavaScript/TypeScript.** The original plan was JS/TS-first. The research
  claim is about the *policy*, and Python's standard-library `ast` module reaches the policy
  experiments fastest. Tree-sitter-based JS/TS parsing becomes **M6**, after the policy is
  validated; the graph schema is language-agnostic so the parser is the only piece that changes.
* **No Postgres, pgvector, Redis, or Neo4j for M1–M5.** Graphs persist as JSON; retrieval runs
  in-process. The database and queue stack belongs to the product path, not the research path.
* **No Monaco UI until the M5 table exists.**
* **Mocked LLM in tests, frozen generator in experiments.** No network access in unit tests.
* **If CrossCodeEval cannot be downloaded**, synthetic cross-file tasks generated from
  `data/fixtures/` are used, and that substitution is documented in the results — never silently.

---

## Product Phases (0–11) — after M5

The phases below describe the full developer tool. They remain the long-term plan, with two
amendments from the research pivot: **Phase 4's hybrid ranker is now an ablation rather than the
core contribution**, and **Phases 7–10 do not start until the M5 comparison table exists**.

---

## Phase 0 — Project Foundation

**Goal:** Establish the repository and development environment.

### Tasks

* initialize monorepo
* configure TypeScript
* configure Python service
* create Node.js API
* create React application
* establish database
* establish basic CI
* define API contracts

**Estimated time:** 2–3 days

---

# Phase 1 — Research Baseline

**Goal:** Build the simplest possible code-completion retrieval system.

### Tasks

* repository ingestion
* code chunking
* embeddings
* vector indexing
* semantic search
* basic LLM completion
* token measurement

Architecture:

```text
Repository
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Search
    ↓
Top-K Context
    ↓
LLM
    ↓
Completion
```

### Deliverable

A working **Semantic RAG baseline**.

**Estimated time:** 5–7 days

---

# Phase 2 — Code Intelligence

**Goal:** Understand repository structure.

### Tasks

* AST parsing
* function extraction
* class extraction
* import/export extraction
* function-call extraction
* reference extraction
* dependency graph
* code entity metadata

Architecture:

```text
Repository
    ↓
Parser
    ↓
AST
    ↓
Code Entities
    ↓
Relationship Graph
```

**Estimated time:** 1–2 weeks

---

# Phase 3 — Data-Flow Analysis

**Goal:** Add information about how values move through the repository.

### Tasks

* identify variables
* track definitions
* track references
* identify relevant flows
* construct data-flow relationships
* connect data-flow information to repository entities

**Estimated time:** 1–2 weeks

> Keep this phase scoped. Full static data-flow analysis for arbitrary programming languages is a much larger problem than this project needs.

Start with **JavaScript/TypeScript** and a clearly defined subset of language constructs.

---

# Phase 4 — Hybrid Retrieval *(ablation, not the contribution)*

**Goal:** Combine multiple retrieval signals.

> **Status after the research pivot:** hybrid ranking is an *ablation* supporting row 6 of
> the main table, not the headline claim. It answers "what do we retrieve once the policy
> has decided to retrieve?" — a question orthogonal to the contribution. Build it only
> after M4, and only to the depth the ablation needs.

### Tasks

* semantic candidate generation
* structural candidate generation
* dependency candidate generation
* data-flow candidate generation
* candidate deduplication
* scoring
* ranking
* configurable retrieval weights

Architecture:

```text
             Query
               ↓
 ┌─────────────┼─────────────┐
 ▼             ▼             ▼
Semantic    Structural    Relationship
Search      Search        Search
 └─────────────┼─────────────┘
               ▼
        Candidate Pool
               ↓
         Hybrid Ranking
```

**Estimated time:** 1 week

---

# Phase 5 — Context Optimization

**Goal:** Reduce unnecessary context.

### Tasks

* token counting
* context budgets
* candidate ranking
* context pruning
* file-level selection
* function-level selection
* redundancy removal
* context assembly

Target:

```text
Retrieved context
        ↓
12,000 tokens
        ↓
Optimization
        ↓
2,800 tokens
```

Then test whether completion quality remains stable or improves.

**Estimated time:** 1–2 weeks

---

# Phase 6 — Research Evaluation

**Goal:** Determine whether the proposed approach actually works.

> This phase is executed as milestones **M2–M5** under `services/ai/scripts/`. The primary
> comparison is no longer "Semantic RAG vs ContextOpti" but the six-row table above — in
> particular ContextOpti versus *selective-over-chunks*, which is the baseline that isolates
> the contribution.

### Tasks

* create benchmark dataset
* define completion tasks
* run baseline experiments
* run hybrid experiments
* calculate retrieval metrics
* calculate completion metrics
* measure token reduction
* measure latency
* perform ablation studies
* generate experiment reports

### Primary comparison

```text
Semantic RAG
      VS
ContextOpti
```

### Deliverable

A reproducible experiment pipeline and quantitative results.

**Estimated time:** 1–2 weeks

---

# Phase 7 — Developer Interface

**Blocked until `outputs/m5_main_table.md` exists.** Only after the policy has been validated against all six baselines.

### Tasks

* Monaco editor
* repository explorer
* file tabs
* code completion UI
* completion suggestions
* loading/error states
* context inspection panel

Example:

```text
Context Used

Semantic candidates:       12
Structural candidates:      7
Data-flow candidates:       5

Final files:                4
Context tokens:          2,740
```

**Estimated time:** 1 week

---

# Phase 8 — Research Visualization

This is a major differentiator for the project.

Build a **Compare Retrieval** interface.

Example:

```text
┌─────────────────────────────────────────────────────┐
│              Retrieval Comparison                   │
├────────────────────┬─────────────┬──────────────────┤
│                    │ Semantic RAG│ ContextOpti      │
├────────────────────┼─────────────┼──────────────────┤
│ Files retrieved    │ 14          │ 5                │
│ Context tokens     │ 8,420       │ 2,740            │
│ Retrieval latency  │ 210 ms      │ 260 ms           │
│ Completion score   │ 0.71        │ 0.75             │
│ Context reduction  │ —           │ 67.5%            │
└────────────────────┴─────────────┴──────────────────┘
```

The numbers above are examples only; actual values must come from experiments.

**Estimated time:** 4–7 days

---

# Phase 9 — Production Engineering

### Tasks

* authentication
* project isolation
* rate limiting
* Redis caching
* asynchronous repository indexing
* background jobs
* logging
* error handling
* request IDs
* usage tracking
* model/token tracking

**Estimated time:** 1 week

---

# Phase 10 — Deployment

### Target architecture

```text
React
 ↓
Frontend Hosting

Node.js API
 ↓
Backend Hosting

Python AI Service
 ↓
AI/Analysis Hosting

PostgreSQL + pgvector
 ↓
Managed Database

Redis
 ↓
Managed Redis

LLM API
 ↓
External Model Provider
```

**Estimated time:** 3–5 days

---

# Phase 11 — Final Research & Documentation

### Deliverables

* research paper
* system architecture
* methodology
* experiment results
* ablation study
* README
* API documentation
* deployment documentation
* demo video
* presentation
* resume description

**Estimated time:** 1 week

---

# Overall Timeline

| Stage                   | Estimated Time |
| ----------------------- | -------------: |
| Foundation              |       2–3 days |
| Semantic RAG baseline   |       5–7 days |
| AST + code intelligence |      1–2 weeks |
| Data-flow analysis      |      1–2 weeks |
| Hybrid retrieval        |        ~1 week |
| Context optimization    |      1–2 weeks |
| Evaluation              |      1–2 weeks |
| Developer interface     |        ~1 week |
| Research dashboard      |       4–7 days |
| Production engineering  |        ~1 week |
| Deployment              |       3–5 days |
| Documentation           |        ~1 week |

### Realistic target

**Research prototype:** 4–6 weeks

**Strong complete project:** 8–12 weeks

**Polished research + placement project:** 12–16 weeks

These estimates assume consistent development. Note that the **research core (M0–M5) targets
Python fixtures**, not JavaScript/TypeScript — see the scope decisions above. JS/TS support
moves to M6, after the selective policy is validated.

---

# Recommended Development Strategy

Do **not** build everything simultaneously.

Follow this order:

```text
                RESEARCH
                   │
                   ▼
           Define hypothesis
                   │
                   ▼
           Build baseline RAG
                   │
                   ▼
        Build code intelligence
                   │
                   ▼
       Build hybrid retrieval
                   │
                   ▼
        Build context optimizer
                   │
                   ▼
              Evaluate
                   │
             ┌─────┴─────┐
             │           │
          Works?       Doesn't?
             │           │
             ▼           ▼
         Product      Iterate
         Layer        Research
             │
             ▼
          Web IDE
             │
             ▼
      Comparison UI
             │
             ▼
        Production
             │
             ▼
        Deployment
```

**Research first. Product second.**

The research result determines what deserves to become a product feature.

---

# Future Scope

After the core system is validated, ContextOpti can be extended with:

* multi-language support
* agentic repository exploration
* tool calling
* automatic context refinement
* learned retrieval ranking
* query expansion
* context compression
* local LLM support
* IDE extensions
* GitHub integration
* personalized repository indexing

Agentic capabilities should remain **future scope initially**. The core research contribution is efficient context selection, not building a generic coding agent.

---

# Project Goals

ContextOpti aims to demonstrate that effective LLM code completion does not require retrieving
repository context on every request — and that the repository's own structure is what tells
you which requests need it, where to look, and how much to spend.

The project investigates whether **better context selection can achieve a more favorable trade-off between:**

```text
Context Relevance
        ↕
Completion Quality
        ↕
Token Usage
        ↕
Latency
        ↕
Inference Cost
```

---

# Why ContextOpti?

The project combines four areas:

### AI Engineering

* LLM APIs
* embeddings
* RAG
* context engineering
* LLM evaluation

### Software Engineering

* AST analysis
* dependency graphs
* data-flow analysis
* repository indexing

### Backend Engineering

* APIs
* PostgreSQL
* Redis
* queues
* caching
* authentication
* deployment

### Research

* baseline comparison
* controlled experiments
* ablation studies
* retrieval metrics
* completion metrics
* efficiency analysis

ContextOpti is therefore designed not simply as an AI application, but as a **research-backed developer tool demonstrating efficient repository-level context selection for LLM code completion.**
