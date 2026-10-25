# Architecture

`clinical-research-assistant` (package `cra`) is an agentic RAG service that
answers multi-step questions about clinical-research documents with cited
evidence, and explicitly refuses when the evidence is insufficient.

## High-level flow

```mermaid
flowchart TD
    subgraph Client
        U[User question]
    end

    U -->|POST /ask| API[FastAPI app]
    API -->|redact PII, set request_id| ORC[Orchestrator]

    subgraph Graph[LangGraph pipeline]
        direction LR
        R[Research agent] --> S[Summarize agent] --> P[Respond agent]
    end

    ORC --> Graph
    R -->|decompose <= 3 sub-queries| LLM[(LLM: Nova Pro / Fake)]
    R -->|retrieve top-k per sub-query, dedup| VS[(FAISS vector store)]
    S -->|summarize per sub-query, only evidence| LLM
    P -->|compose cited answer OR refuse| LLM

    VS -. loaded at startup .- IDX[(Persisted index + chunk metadata)]
    P --> RESP[AskResponse: answer, citations, trace, latency, tokens, cost]
    RESP -->|JSON| U

    ING[Ingest CLI] -->|load, chunk, embed| EMB[(Embeddings: all-MiniLM / Hashing)]
    EMB --> IDX
```

## Components

| Layer | Package | Responsibility |
|---|---|---|
| Config | `cra.config` | Pydantic-settings; all tunables from `CRA_*` env vars. Bedrock model ID + region are injected, never hard-coded. |
| Ingestion | `cra.ingest` | Load docs, parse front-matter + sections, section-aware chunking, build & persist the FAISS index. |
| Retrieval | `cra.retrieval` | Embedding backends (real + hashing fake), FAISS store, multi-sub-query retrieval with de-duplication. |
| LLM | `cra.llm` | Backend abstraction, Bedrock Nova Pro adapter, deterministic fake, LRU cache + per-request call limit, token/cost accounting. |
| Agents | `cra.agents` | Research / summarize / respond nodes, prompts, prompt-injection guard. |
| Graph | `cra.graph` | Typed `GraphState` contract, LangGraph wiring, request orchestrator. |
| API | `cra.api` | FastAPI `POST /ask` + `GET /health`, request/response schemas. |
| Observability | `cra.observability` | Structured JSON logging with request-ID propagation, PII redaction. |
| Evals | `cra.evals` | Golden set, metrics, threshold gate, Markdown reports. |

## The three agents

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant R as Research
    participant Se as Summarize
    participant P as Respond
    participant L as LLM
    participant V as FAISS

    O->>R: GraphState(question)
    R->>L: decompose (<= 3 sub-queries)
    L-->>R: sub-queries
    loop per sub-query
        R->>V: retrieve top-k
        V-->>R: chunks + scores
    end
    R->>R: de-duplicate, compute confidence (cosine x grounding)
    R-->>Se: evidence, sub_queries
    loop per sub-query
        Se->>L: summarize (evidence only, injection-guarded)
        L-->>Se: grounded summary + chunk IDs
    end
    Se-->>P: summaries
    alt confidence >= threshold
        P->>L: compose answer with [doc_id:section] citations
        L-->>P: grounded answer
    else confidence < threshold
        P->>P: "insufficient evidence" + what was found
    end
    P-->>O: answer, citations, trace, usage
```

### Research agent
Decomposes the question into at most `CRA_MAX_SUB_QUERIES` sub-queries, retrieves
`CRA_RETRIEVAL_TOP_K` chunks per sub-query, de-duplicates (best score per chunk),
and writes evidence + a blended confidence score into graph state. Confidence is
`mean_cosine x lexical_grounding`, where grounding is the fraction of the
question's salient terms present in the retrieved evidence (and titles). This
makes the refusal gate robust even when the embedder is weak or the question is
off-topic.

### Summarize agent
Produces one summary per sub-query using **only** the retrieved evidence. Each
summary carries the chunk IDs it drew from. Evidence is sanitized by the
prompt-injection guard and presented inside a clearly delimited untrusted-data
block.

### Respond agent
If confidence is below `CRA_CONFIDENCE_THRESHOLD` (or there is no evidence),
returns an explicit *insufficient evidence* response describing what little was
found — it never guesses. Otherwise it composes the final answer with
`[doc_id:section]` citations. Every answer carries a *not medical advice* note.

## Typed state contract

`cra.graph.state.GraphState` (Pydantic) is the single contract threaded between
agents. Each agent reads the fields it needs and writes only the fields it owns,
so evidence and summaries pass through without loss or duplication. This
invariant is covered by dedicated tests (`tests/test_graph_state_contract.py`).

## Safety

- **PII redaction** on the inbound question and on log messages (emails, phone
  numbers, long ID-like numbers), preserving clinical numbers like doses and
  percentages.
- **Prompt-injection guard**: retrieved text is always treated as data, wrapped
  in an untrusted-data block, and common injection markers are redacted before
  reaching the model.
- **Not medical advice**: every answer carries the disclaimer.
- **Insufficient-evidence refusal**: the service refuses rather than
  hallucinating when retrieval confidence is low.

## LLM backends

- **Bedrock** (`CRA_LLM_PROVIDER=bedrock`): Amazon Nova Pro via `langchain-aws`,
  model ID + region from env, with retries. Real usage metadata drives cost.
- **Fake** (`CRA_LLM_PROVIDER=fake`, default): deterministic, offline,
  evidence-grounded backend driven by `TASK:` tags. Used for all tests, CI, and
  offline eval runs so nothing calls AWS by accident.

An LRU cache (process-global) short-circuits identical prompts; a per-request
budget caps non-cached LLM calls.
