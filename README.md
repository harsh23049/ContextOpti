# ContextOpti

### Efficient Repository Context Selection for LLM-Based Code Completion

ContextOpti is a repository-aware AI code completion system designed to address a fundamental problem in LLM-powered software development:

> **How can we provide an LLM with the most relevant repository context while minimizing unnecessary context and token consumption?**

Modern code-completion systems can benefit from repository-level context, but retrieving too much irrelevant code increases token usage, latency, and cost while potentially reducing generation quality.

ContextOpti investigates a **hybrid context-retrieval approach** that combines semantic similarity with structural, dependency, and data-flow relationships within a codebase.

The project combines **AI engineering, code intelligence, information retrieval, and research experimentation** into a single system.

---

## Research Problem

Given a code-completion request inside a large repository, traditional semantic retrieval may retrieve code that is textually or semantically similar but structurally unrelated to the current code.

ContextOpti investigates whether repository structure and code relationships can improve context selection.

### Research Question

> **Can repository context selection for LLM-based code completion be improved by combining semantic retrieval with structural, dependency, and data-flow information while reducing unnecessary context and token consumption?**

---

## Core Idea

Instead of:

```text
Code Completion Request
        ↓
Semantic Search
        ↓
Top-K Code Chunks
        ↓
LLM
        ↓
Completion
```

ContextOpti uses:

```text
Code Completion Request
        ↓
Query / Context Analysis
        ↓
 ┌──────────────┬──────────────┬──────────────┐
 │              │              │              │
Semantic     Structural     Dependency    Data-flow
Retrieval    Retrieval      Retrieval     Retrieval
 │              │              │              │
 └──────────────┴──────────────┴──────────────┘
                       ↓
                Candidate Pool
                       ↓
                 Hybrid Ranking
                       ↓
                Context Selection
                       ↓
                Context Optimization
                       ↓
                  Token Budget
                       ↓
                      LLM
                       ↓
                Code Completion
```

The goal is **not to retrieve more context**.

The goal is to retrieve **better context with less unnecessary information**.

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

## 6. Hybrid Retrieval

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

ContextOpti will compare multiple retrieval strategies.

### Baseline

```text
Semantic Retrieval → LLM
```

### Proposed Approach

```text
Semantic
   +
Structural
   +
Dependency
   +
Data-flow
   ↓
Hybrid Retrieval
   ↓
Context Optimization
   ↓
LLM
```

Evaluation will consider three dimensions.

### Retrieval Quality

* Recall@K
* Precision@K
* MRR
* NDCG where appropriate

### Code Completion Quality

* Exact Match
* CodeBLEU
* pass@k where applicable

### Context Efficiency

* retrieved tokens
* selected tokens
* context reduction percentage
* retrieval latency
* end-to-end latency
* LLM token consumption
* estimated inference cost

---

# Ablation Studies

To understand which components actually contribute to performance, ContextOpti will evaluate combinations such as:

```text
A. Semantic only

B. Semantic + Structural

C. Semantic + Dependency

D. Semantic + Data-flow

E. Semantic + Structural + Dependency

F. Semantic + Structural + Dependency + Data-flow
```

This is important for the research because a complicated system is not automatically a better system.

The experiments should identify **which signals provide measurable improvement**.

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

The roadmap is intentionally divided into **Research → Core Engine → Product → Production**.

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

# Phase 4 — Hybrid Retrieval

**Goal:** Combine multiple retrieval signals.

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

**Only after the retrieval approach has been validated.**

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

These estimates assume consistent development and that the scope remains focused on JavaScript/TypeScript initially.

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

ContextOpti aims to demonstrate that effective LLM code completion does not necessarily require providing an LLM with large amounts of repository context.

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
