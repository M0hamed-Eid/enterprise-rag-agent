# Enterprise RAG Agent Pipeline

The agent combines recent conversation, relevant older memories, and
authoritative tool results. A question may require multiple tools.

```mermaid
flowchart TD
    USER["Executive question"]
    AUTH["Trusted application boundary<br/>Authenticate user and establish access scope"]
    BUFFER["Short-term conversation<br/>ConversationBufferWindowMemory<br/>k = 5 interactions"]
    SEMANTIC["Long-term conversation<br/>VectorStoreRetrieverMemory<br/>Retrieve top 3 similar memories"]
    MEMDB[("Private conversation vector store<br/>Scoped by tenant, user, and session")]
    CONTEXT["Conversation context<br/>Recent exchanges + relevant older memories"]
    ROUTER["LLM agent and tool router<br/>Resolve references; select one or more tools"]

    SQL["EnterpriseSQLDB<br/>SQL Database Chain generates SQL"]
    SQLCHECK["Validate one SELECT statement<br/>Restrict tables and result size"]
    SQLDB[("Enterprise database<br/>regional_sales<br/>client_contracts<br/>employee_roster only if authorized")]
    SQLRESULT["SQL result rows<br/>Query provenance"]

    WIKI["HR_Ops_Wiki<br/>VectorStoreQA using RetrievalQA"]
    WIKIDB[("Enterprise wiki vector store<br/>Separate from conversational memory")]
    WIKICONTEXT["Relevant wiki passages"]
    WIKIRESULT["Grounded wiki answer<br/>Document source identifiers"]

    NEWS["LiveMarketNews<br/>SERP API wrapper"]
    SERP["External news search"]
    NEWSRESULT["News titles, snippets,<br/>URLs and available dates"]

    ASSEMBLY["LLM context assembly<br/>Instructions + question + memory + tool evidence<br/>Production: enforce total token budget"]
    ANSWER["Answer with evidence and uncertainty<br/>Request clarification when necessary"]
    SAVE["Save completed interaction<br/>Production: retention, consent and redaction policy"]

    DOCS["Approved HR and operational documents"]
    INGEST["Separate ingestion workflow<br/>Validate, chunk, attach source metadata, embed"]

    USER --> AUTH
    AUTH --> BUFFER
    AUTH --> SEMANTIC
    SEMANTIC -->|"Similarity query"| MEMDB
    MEMDB -->|"Relevant older interactions"| SEMANTIC
    BUFFER --> CONTEXT
    SEMANTIC --> CONTEXT
    SEMANTIC -.->|"Empty result or retrieval failure:<br/>continue with recent buffer"| CONTEXT

    CONTEXT --> ROUTER
    AUTH -->|"Current question and access scope"| ROUTER

    ROUTER --> SQL
    SQL --> SQLCHECK
    SQLCHECK -->|"Read-only connection"| SQLDB
    SQLDB --> SQLRESULT

    ROUTER --> WIKI
    WIKI -->|"Embed search query"| WIKIDB
    WIKIDB --> WIKICONTEXT
    WIKICONTEXT -->|"Injected into wiki QA LLM context"| WIKI
    WIKI --> WIKIRESULT

    ROUTER --> NEWS
    NEWS --> SERP
    SERP --> NEWSRESULT

    CONTEXT --> ASSEMBLY
    ROUTER -->|"Current question and instructions"| ASSEMBLY
    SQLRESULT --> ASSEMBLY
    WIKIRESULT --> ASSEMBLY
    NEWSRESULT --> ASSEMBLY

    ASSEMBLY -->|"Tool observations enter agent context"| ROUTER
    ROUTER -->|"Final grounded response"| ANSWER
    ANSWER --> USER
    ANSWER --> SAVE
    SAVE --> BUFFER
    SAVE --> MEMDB

    DOCS --> INGEST
    INGEST --> WIKIDB
```

## Schema mapping

- `regional_sales`: region, Q1 revenue, Q2 revenue, and target margin.
- `client_contracts`: client names, contract values in USD, and contract status.
- `employee_roster`: department, role, and clearance level; restricted by default.

The schema defines no foreign keys between these tables. Do not invent joins.

The sales table has no reporting year or explicit currency. Questions requiring
those attributes need clarification or additional authoritative metadata.

## Read-only SQL boundary

The SQL chain generates SQL without executing it. A separate wrapper validates
the statement and executes it against a read-only database connection.

For the local stub, SQLite uses URI `mode=ro` and `PRAGMA query_only=ON`.
Production PostgreSQL should use a dedicated role with SELECT privileges only
on approved views, with row access policies where required.

Prompt instructions and SQL parsing supplement database permissions; they do
not replace them.

## Memory behavior

Recent conversation is always available. Semantic retrieval adds older relevant
interactions. Empty results or semantic-store failures leave the recent buffer
available as a fallback.

Memory is conversation context, not authoritative evidence for current revenue,
policies, or news. Those facts must be retrieved again when needed.

## Implementation boundary

The configuration stub demonstrates memory, routing, retrieval, and read-only
SQL execution. Production authentication, document ingestion, strict token
budget enforcement, distributed storage, and retention enforcement remain
separate implementation work.