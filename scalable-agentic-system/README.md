# Scalable Agentic System

A reference architecture and working skeleton for an agent that stays accurate
and fast whether it has **50 tools or 5,000+**. Built around one core idea:

> **Never give the LLM all the tools. Retrieve the right few, every turn.**

---

## 1. The Problem

Function-calling accuracy degrades as the number of tool definitions in context
grows — the model has more surface area to confuse, more near-duplicate
descriptions to disambiguate, and a bigger prompt to reason over. A naive
design that binds all N tools into every LLM call works fine at N=10 and
quietly falls apart at N=200.

The fix isn't a smarter prompt. It's **not sending all the tools in the first
place.** Tool selection becomes a retrieval problem, solved *before* the LLM
ever reasons about which tool to call — the same way RAG solves "which
document is relevant" before an LLM answers a question.

---

## 2. Architecture Overview

```
User ─▶ Input Processor ─▶ Tool Router & Selector ─▶ Planner ─▶ Execution Layer ─▶ User
                                     │                                  │
                         Tool Registry (Postgres+pgvector)     Memory & State (Redis+Postgres)
                            [50 .. 5000+ tools]                            │
                                                              Observability (LangSmith/OTel)
```

| Stage | Responsibility | Never sees |
|---|---|---|
| **1. Input Processor** | Light intent/entity extraction, query rewriting with recent context | The tool catalog |
| **2. Tool Router & Selector** | Embed query → ANN search registry → re-rank → **Top-K (default 8)** | Anything beyond the candidate pool |
| **3. Planner** | Turn Top-K tools + intent into an ordered execution plan (single or multi-step) | Any tool outside the Top-K |
| **4. Execution Layer** | Validate args → call → retry on transient failure → log | Tool selection logic |

The **entire scaling story lives in stage 2.** Stages 3 and 4 always operate
on a small, fixed-size candidate set, so their cost and accuracy are flat with
respect to registry size.

### Why this holds up from 50 → 500 → 1000+ tools

- **Retrieval is O(log n)-ish, not O(n).** pgvector's `ivfflat` index means
  searching a registry of 5,000 tools costs roughly the same as searching 50.
- **The LLM's context never grows.** Top-K is a config value
  (`TOP_K_TOOLS`), not a function of registry size.
- **Adding a new integration is a data operation, not a code change.** A new
  500-endpoint API collection becomes 500 rows via `bulk_register()`
  (`scripts/seed_registry.py`) — no prompt rewrites, no new `if` branches in
  the agent.
- **Categories provide an optional second axis of narrowing.** If the caller
  already knows the domain (e.g. a "Billing" tab in the UI), `category_hint`
  in `tool_router.select_tools()` pre-filters before the vector search even
  runs.

---

## 3. The PayPal Scenario, Walked Through

**"Send an invoice for $50 to user_123"**

1. Input Processor rewrites the query with session context (resolves
   `user_123` if it was mentioned earlier).
2. Router embeds the query, ANN-searches the registry, re-ranks by
   `similarity (0.65) + historical_success_rate (0.20) + permission_match (0.10) − latency_penalty (0.05)`.
   Top candidates: `Create Invoice (0.96)`, `Send Payment (0.72)`,
   `Get Invoice (0.65)` — everything else is filtered out before the LLM is
   ever called.
3. Planner receives just those 3–8 tool defs, picks `paypal.invoices.create`,
   fills `{amount: 50, recipient_email: ..., currency: "USD"}`.
4. Validation service checks args against the JSON schema. Amount is under the
   $500 confirmation threshold, so it proceeds straight to execution.
5. Execution Layer calls the PayPal API wrapper (`tools/paypal_tools.py`),
   retries on transient failure, logs the result to `execution_log`.
6. Responder returns a natural-language confirmation.

**"Is there a dispute open from user_123?"** routes to
`paypal.disputes.get_status` the same way — a *read* call, so it's classified
idempotent (`services/retry.py`) and gets a more aggressive retry policy than
the invoice-creation call above, which is non-idempotent and retries
conservatively to avoid double-sends.

**Scaling to 500 APIs across many services:** nothing above changes. The
registry now has 500+ rows instead of 50; the router still returns ~8
candidates; the planner still reasons over ~8 options. The only new
operational concern is registry hygiene (see §7).

---

## 4. RAG Tool & System Search Tool

Both are treated as **exactly one tool each** in the registry — not one tool
per document, not one tool per log table. This is a deliberate application of
the same principle that keeps the *tool* catalog scalable:

- `rag.knowledge_base.search` — the router picks this ONE tool for
  "how do I..." questions; retrieval over the (possibly huge) document corpus
  happens *inside* the tool, via LlamaIndex, after tool selection is already
  done. Growing the knowledge base from 100 to 100,000 pages never grows the
  tool catalog.
- `system.capability_search` / `system.log_search` — lets the agent answer
  meta-questions ("what tools exist for invoices?", "what's the status of my
  last request?") by querying the *same* tool registry and the execution log,
  surfaced as one more selectable tool rather than a special-cased code path.

---

## 5. Framework Choices & Trade-offs

| Choice | Why | Trade-off accepted |
|---|---|---|
| **LangGraph** for orchestration | Explicit, typed state machine (`agent/state.py`); first-class conditional branches (clarification / high-risk confirmation gates) and human-in-the-loop interrupts; native LangSmith tracing per node. | Steeper learning curve and one more dependency vs. a bare LLM SDK loop or a single LangChain `AgentExecutor`. We don't use plain `AgentExecutor` because its ReAct loop buries branching logic inside free-form reasoning text rather than the explicit graph structure this system needs. |
| **Postgres + pgvector** for the Tool Registry | One system for both structured metadata (permissions, schemas, success rates) *and* vector search — no separate vector DB to keep in sync; `ivfflat` scales to millions of rows; SQL joins make permission/category filtering trivial. | A dedicated vector DB (Pinecone, Weaviate, Qdrant) would have faster ANN search at very large scale (10M+ vectors) and managed ops, at the cost of a second system to keep consistent with tool metadata. For a tool catalog (thousands, not millions, of rows) pgvector's ceiling is more than sufficient. |
| **Redis** for session/working state, Postgres for durable state | Redis gives sub-ms reads for the hot path (conversation summary on every turn); Postgres gives durability and queryability (execution log, audit trail) that a pure cache can't. | Two systems instead of one — accepted because ephemeral vs. durable state genuinely have different access patterns and failure-mode requirements. |
| **LlamaIndex** for the RAG pipeline | Purpose-built ingestion/indexing/retrieval abstractions (loaders, node parsing, pgvector store integration) — more RAG-specific ergonomics than building on raw LangChain retrievers. | Another framework in the stack; mitigated by keeping RAG fully encapsulated behind one tool interface (`tools/rag_tool.py`) so it could be swapped for a hand-rolled retriever without touching the agent. |
| **LangSmith + OpenTelemetry** for observability | LangSmith traces every LangGraph node automatically (near-zero instrumentation cost) and gives per-tool cost/latency breakdowns out of the box; OTel is layered in for infra-level metrics (API latency, error rates) that ops teams already consume via existing dashboards. | LangSmith is a paid, hosted product — the design keeps OTel as a vendor-neutral fallback so tracing isn't fully locked to one vendor. |
| **Sequential plan execution** (not a full DAG scheduler) | Most multi-tool turns in this domain are short, linearly dependent chains (2–4 steps). A sequential executor with `{{step_N.output.field}}` references covers this with far less complexity than a general DAG engine. | A genuinely parallel, branching workflow (e.g. fan-out across 10 independent API calls) would benefit from a real DAG/workflow engine (e.g. Temporal). Flagged as a natural v2 extension, not built now, to avoid over-engineering for the given scenario. |

**Frameworks considered and not chosen:**
- **CrewAI** — optimized for multi-agent role-play (a "crew" of specialized
  agents). This system is fundamentally single-agent-many-tools, not
  multi-agent, so CrewAI's abstractions (crews, roles, delegation) would add
  overhead without solving the actual bottleneck, which is tool retrieval.
- **DSPy** — excellent for *optimizing* prompts/few-shot examples against a
  metric, but it's a prompt-programming framework, not an orchestration
  framework. It could sit *inside* the Planner node later (auto-tuning the
  planning prompt against logged success/failure data) but isn't a
  replacement for LangGraph's state-machine role.
- **Plain LangChain `AgentExecutor`** — simplest option, but its single-loop
  ReAct pattern doesn't cleanly express "stop and ask for confirmation before
  a $5,000 payment" as a first-class graph edge the way LangGraph does.

---

## 6. State Management

Three tiers, matching how frequently each layer changes and how durable it
needs to be:

1. **Working state** (`agent/state.py`, in-memory during one graph run) —
   candidate tools, plan, execution results for the current turn.
2. **Session state** (Redis, `database/connection.py`) — rolling conversation
   summary, keyed by `session_id`, TTL'd. Read/written on every turn; this is
   the hot path.
3. **Durable state** (Postgres, `database/models.py`) — `execution_log`
   (every tool call, success/failure, latency — audit trail + feeds
   `historical_success_rate` back into ranking) and `sessions` (so a
   conversation can resume even on a Redis cache miss).

---

## 7. Scalability Notes

- **Registry growth is a data-plane, not code-plane, operation.** New
  integrations = new rows (`scripts/seed_registry.py`), via
  `bulk_register()` with batched embedding calls.
- **Hygiene as the catalog grows:** near-duplicate tool descriptions
  (e.g. two integrations both exposing "create invoice") will compete in
  ranking. Mitigations: category namespacing (`category_hint` pre-filter),
  periodic embedding-similarity audits across the registry to flag
  duplicates, and letting `historical_success_rate` naturally demote the
  worse-performing duplicate over time.
- **Horizontal scaling:** the FastAPI layer is stateless (all state lives in
  Redis/Postgres), so it scales horizontally behind a load balancer with no
  sticky-session requirement. pgvector read replicas absorb router query
  load independently of the write path used for registry updates.
- **Cold-start cost:** embedding the user's query is one extra network hop
  per turn. Mitigated by using a small/fast embedding model
  (`text-embedding-3-small`) and caching embeddings for frequently repeated
  queries.

## 8. Error Handling

- **Validation before execution** (`services/validation.py`) — hallucinated
  or missing parameters are caught against the tool's JSON schema before any
  external call is made.
- **Idempotency-aware retries** (`services/retry.py`) — read-style calls
  (reports, dispute status, RAG/system search) retry aggressively;
  state-changing calls (payments, invoice creation) retry conservatively and
  only on clearly transient errors, never on 4xx, to avoid double side
  effects.
- **Risk-gated confirmation** (`services/security.py`) — high-risk categories
  (payments, refunds, dispute resolution) above a configurable amount
  threshold short-circuit the graph to a confirmation prompt instead of
  auto-executing.
- **Graceful ambiguity handling** — if the router returns no tool above the
  similarity floor, or the planner can't fill required parameters, the graph
  routes to a clarifying question instead of guessing (`agent/planner.py`'s
  `needs_clarification` contract).
- **Full audit trail** — every attempt (success or failure), with retry count
  and latency, is persisted to `execution_log`, which both the System Search
  Tool and observability dashboards read from.

---

## 9. Project Layout

```
scalable-agentic-system/
├── app/            # FastAPI entrypoint, config, shared schemas
├── agent/          # LangGraph state machine, planner
├── router/         # Tool registry, embeddings, ranking, Top-K selection
├── tools/          # PayPal wrappers, RAG tool, System Search tool
├── rag/            # LlamaIndex ingestion/retrieval for the RAG tool
├── services/       # Execution layer, validation, retry policy, security
├── database/       # Postgres models (execution log, sessions), Redis cache
├── scripts/        # seed_registry.py — bulk-load tool collections
└── tests/          # Router ranking, agent branching, validation/retry unit tests
```

## 10. Running It

```bash
pip install -r requirements.txt
cp .env .env.local   # fill in real API keys / DB URLs
python -m scripts.seed_registry   # loads PayPal + RAG + System tools into pgvector
uvicorn app.main:app --reload
```

```bash
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","user_id":"u1","message":"Send an invoice for $50 to user_123"}'
```

Run tests (no live DB/API keys required — external calls are mocked):

```bash
pytest tests/
```

## 11. What I'd Build Next

- Feedback loop that writes `execution_log` outcomes back into
  `historical_success_rate` on a schedule, so ranking genuinely improves
  with usage rather than being static after seeding.
- A duplicate/overlap detector that runs periodically over the registry's
  embeddings to catch near-identical tools across integrations before they
  degrade ranking quality.
- Swap sequential plan execution for a real DAG executor once workflows need
  genuine fan-out/fan-in rather than linear chains.
- Streaming responses over the `/chat` endpoint (SSE) so multi-step plans
  surface intermediate progress instead of one blocking response.
