# Enterprise RAG Agent: Memory Strategy Evaluation

**Stakeholder:** Priya Sharma, Lead AI Engineer  
**Date:** 2026-11-09
**Status:** Configuration stub with recorded real-model execution

## 1. Executive Summary

The proposed agent routes executive questions to three evidence sources:
structured enterprise metrics through a SQL Database Chain, internal HR and
operational documents through VectorStoreQA, and public market news through
SERP API.

Conversation context uses two complementary components:

1. ConversationBufferWindowMemory exposes the five most recent exchanges.
2. VectorStoreRetrieverMemory retrieves up to three semantically relevant
   historical exchanges from persistent storage.

This approach preserves immediate conversational continuity while making older
context available without repeatedly inserting the full conversation.

Memory does not eliminate hallucination. The agent must re-query authoritative
sources, acknowledge missing evidence, and avoid treating remembered answers
as verified current facts.

## 2. Architecture

See [pipeline diagram](../architecture/pipeline_diagram.md).

Enterprise wiki knowledge and conversational memory use separate collections.
The trusted application selects the user's conversation scope and authorized
wiki scope before constructing the agent.

The agent can call several tools for one question. For example, explaining a
regional revenue change may require SQL results and public news.

Retrieved tool observations are injected into the agent context. Wiki passages
also enter the wiki QA chain's own LLM context.

## 3. Memory Comparison

| Criterion | Five-interaction buffer | Semantic retrieval |
|---|---|---|
| Selection | Most recent exchanges | Similarity to the current query |
| Prompt size | Depends on lengths of five exchanges | Depends on retrieved count and entry lengths |
| Retrieval latency | Local history lookup | Query embedding plus vector search |
| Immediate follow-ups | Usually strong because chronology is retained | May struggle with phrases such as "that region" |
| Older context | Excluded after it leaves the window | Potentially retrievable |
| Failure mode | Deterministic loss of older prompt context | Retrieval misses and irrelevant matches |
| Chronology | Preserved within the exposed window | Not inherently preserved |
| Persistence | In-process in this stub | Persistent Chroma directory in this stub |
| Operating cost | History tokens in LLM requests | History tokens plus embedding/storage/search costs |

Here, k=5 refers to five user/assistant exchanges in the conventional paired
chat pattern. It is not five tokens and does not provide a fixed token cap.

The buffer limits exposed prompt history; it should not be assumed to enforce
a storage-retention or deletion policy.

### Illustrative token estimates

Assumptions, not measured results:

- Full conversation: 40 exchanges.
- Average exchange: 600 tokens.
- Recent buffer: 5 exchanges.
- Semantic retrieval: 3 entries averaging 250 tokens each.

| Strategy | Approximate history tokens per request |
|---|---:|
| Full conversation | 40 × 600 = 24,000 |
| Recent buffer | 5 × 600 = 3,000 |
| Semantic retrieval | 3 × 250 = 750 |
| Hybrid | 3,000 + 750 = 3,750 |

The hybrid example uses approximately 84.4% fewer history tokens than replaying
the full conversation: 1 - 3,750 / 24,000.

These estimates exclude formatting, metadata, system instructions, tool
descriptions, tool observations, and generated output.

The provided stub stores complete exchanges rather than summarized entries.
If retrieved entries average 600 tokens, semantic history instead costs about
1,800 tokens and hybrid history about 4,800 tokens. The smaller estimate
therefore requires shorter exchanges or a separate bounded-memory ingestion
policy.

Semantic retrieval is not automatically cheaper than a recent buffer. Long
retrieved entries or excessive retrieval counts can reverse the comparison.

### Latency

The buffer requires no remote retrieval operation in this stub.

Semantic retrieval requires a query embedding and a vector search. A hosted
deployment adds network latency. Memory writes can also add embedding and
storage latency after a response is generated.

The stub uses synchronous memory operations, so these costs affect end-to-end
request time. The four-turn real evaluation is recorded in section 9. Its observed timings do not establish a benchmark or isolate memory latency.

Evaluate p50 and p95 latency separately for:

- Memory loading.
- SQL, wiki, and news tools.
- Model generation.
- Memory persistence.
- Complete requests.

### Context loss and factual reliability

The buffer reliably retains recent references, but older information disappears
from the prompt once it falls outside the five-exchange window.

Semantic retrieval can recover older constraints and preferences, but retrieval
is approximate. Relevant context may be missed, and outdated or incorrect
answers may be retrieved.

Memory should guide interpretation. SQL, approved wiki documents, and dated
news results should ground factual answers.

## 4. Recommended Hybrid Strategy and Token Budget

Use the recent buffer for immediate continuity, and add semantic history when
available.

If semantic retrieval returns nothing or fails, continue with recent history.
If neither source resolves an ambiguous question, ask for clarification.

The stub implements this fallback through a resilient semantic-memory wrapper.
CombinedMemory alone does not provide exception fallback behavior.

### Illustrative context budget

Assume a 16,000-token context capacity for planning purposes. This is a design
example, not a claim about the configured model.

| Allocation | Tokens |
|---|---:|
| System instructions and tool definitions | 1,500 |
| Current question | 500 |
| Recent conversation | 3,000 |
| Retrieved semantic memories | 750 |
| Tool observations and agent scratchpad | 4,000 |
| Reserved model output | 2,000 |
| Safety margin | 4,250 |
| Total | 16,000 |

Production must count actual tokens before every model call, including
intermediate agent iterations and wiki QA calls.

If the budget is exceeded:

1. Remove duplicate recent/semantic content.
2. Remove irrelevant older memories.
3. Reduce or summarize oversized tool results while preserving provenance.
4. Trim older recent exchanges while retaining essential constraints.
5. Ask the user to narrow the question if sufficient evidence cannot fit.

Neither k=5 nor vector retrieval k=3 enforces this total budget.
The configuration stub does not implement the budget manager.

## 5. Tool Integration

### SQL Database Chain

The chain generates SQL from natural-language questions and reflected table
information. It is configured with return_sql=True so generated SQL is not
executed immediately.

A wrapper:

1. Accepts a single SELECT statement.
2. Restricts referenced tables.
3. Rejects unsupported query structures.
4. Applies a result-row limit.
5. Executes through a read-only SQLite connection.

The default approved tables are:

- regional_sales: Q1/Q2 revenue and target margins.
- client_contracts: USD contract values and status.

employee_roster is excluded because employee access requires a separately
authorized scope. Its clearance_level column does not authenticate a user.

The schema has no foreign keys, reporting year, or explicit sales currency.
The agent must not invent joins, associate figures with a particular year,
or label sales revenue as USD without additional authoritative information.

Database-level read-only access remains essential. SQL validation alone is
not the security boundary.

For PostgreSQL deployment, use a dedicated SELECT-only role on approved views,
with appropriate row restrictions and statement timeouts. Do not grant schema
creation, mutation, or unnecessary function privileges.

### VectorStoreQA

The HR_Ops_Wiki tool uses RetrievalQA over a separate wiki vector collection.

A separate ingestion process must:

- Load approved documents.
- Split them into appropriately sized passages.
- Attach source, effective-date, jurisdiction, and access metadata.
- Embed passages using the same model used at query time.
- Populate the authorized collection.

The tool returns a grounded answer and document source identifiers.

Normal chat does not insert demonstration policy. The explicit --evaluate mode indexes data/demo_wiki.md, a clearly labelled synthetic test policy, in a fresh isolated evaluation collection. It must not be treated as actual HR guidance. An empty collection in normal chat produces no supporting wiki evidence.

### SERP API

The news wrapper uses SerpAPIWrapper with news-search parameters.
It preserves available article titles, snippets, URLs, and publication dates,
and records the retrieval timestamp.

Search snippets are partial evidence, not full article verification.
A missing date must remain missing rather than being invented.

Only public market terms should leave the enterprise boundary.
The stub expresses this restriction in tool instructions; production must
enforce it using an outbound-data policy.

News may suggest explanations for a business trend, but does not establish
causation by itself.

## 6. International Deployment Trade-offs

### Language and token consumption

Token counts vary by language and tokenizer. English estimates must be checked
against representative Arabic and other deployment languages.

Evaluate multilingual retrieval using actual enterprise terminology and
cross-language follow-up questions.

### Regional latency and residency

A distant embedding endpoint or vector database increases retrieval latency.
Place approved services near users where data-residency policy permits.

Local persistent Chroma is a development choice. Multi-region production needs
managed persistence, access controls, backups, and residency-aware routing.

### Time, currency, and jurisdiction

Resolve terms such as "today" using the user's timezone.

Do not interpret "last quarter" from this schema without reporting-period
metadata. Likewise, sales currency cannot be inferred from region alone.

HR retrieval must select the relevant jurisdiction and effective policy.
The stub assumes its authorized wiki scope is appropriately populated;
fine-grained jurisdiction filtering remains production work.

### Privacy, retention, and isolation

Conversation memory can contain sensitive executive information.
Use tenant/user isolation, encrypted storage, retention limits, and explicit
deletion workflows.

The stub namespaces semantic memory by tenant, user, and session. It does not
retrieve memories across sessions. Cross-session recall requires a separately
authorized user-level memory design.

Deleting a session buffer is not sufficient to delete persisted semantic memory.

## 7. Evaluation Plan

Use a fixed set of multi-turn scenarios and compare buffer-only, semantic-only,
and hybrid configurations.

Include:

- Immediate follow-up references.
- A relevant constraint introduced more than five exchanges earlier.
- A deliberately irrelevant semantic match.
- Corrected facts and stale remembered answers.
- Arabic/English cross-language follow-ups.
- Empty and unavailable semantic storage.
- Unauthorized wiki and employee-data requests.
- SQL mutation attempts.
- Missing reporting year and sales currency.
- News requests with no dated supporting results.

Measure:

- Input tokens across all model calls.
- End-to-end p50/p95 latency.
- Retrieval relevance and recall on labeled examples.
- Correct follow-up resolution.
- Unsupported factual claims.
- Source attribution quality.
- Successful denial of unauthorized access and SQL mutations.

Compare outputs with known answers from the supplied seed data and approved
documents. Do not invent benchmark results.

## 8. Conclusion

The recommended architecture combines five recent exchanges with selectively
retrieved older conversation.

This provides useful continuity with less prompt growth than replaying the
entire conversation, while retaining an explicit recent-history fallback.

Correctness still depends on authoritative retrieval, access control,
freshness checks, token-budget enforcement, and evaluation.

The submitted Python is a configuration stub, not a production-complete or
benchmark-validated system.

## 9. Recorded real-model evaluation — 9 November 2026

See [run evidence](../evidence/RUN_EVIDENCE.md),
[actual console output](../evidence/console_output.txt), and
[provider/tool event record](../evidence/live_run.json).
The agent used the Sprints gateway with response model identifier
`gemini/gemini-3.8-flash`; this is real remote inference, not an offline check.
Embeddings and direct semantic recall also succeeded.

The fresh four-turn conversation grounded SQL values (980000 and 1100000), resolved
a follow-up using the recent buffer and re-queried SQL for 12.2449% growth, then
retrieved a cited synthetic HR policy. Live SerpAPI returned articles with URLs; these are search snippets, not full-article verification.

| Turn | Tools | Duration | Model responses | Recorded chat tokens |
| --- | --- | --- | --- | --- |
| 1 | EnterpriseSQLDB | 17.94s | 3 | 2503 |
| 2 | EnterpriseSQLDB | 18.47s | 3 | 3634 |
| 3 | HR_Ops_Wiki | 10.74s | 3 | 2542 |
| 4 | LiveMarketNews | 13.65s | 2 | 4106 |

These are actual end-to-end observations for this run, not memory-only measurements.
The token arithmetic in sections 3–4 remains illustrative. No controlled comparison
of buffer-only, semantic-only and hybrid modes, eviction recall or international
latency benchmark was performed. Separate local checks verified k=5 eviction,
SQL mutation rejection, table allowlisting and the read-only SQLite connection.
