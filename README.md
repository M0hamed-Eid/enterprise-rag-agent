# Enterprise RAG Agent

A modular LangChain agent for read-only sales/contract queries, wiki retrieval,
and market news. It combines a five-interaction conversation buffer with
persistent semantic memory in Chroma. Models run through the configured Sprints
OpenAI-compatible gateway.

## Deliverables and status

- [Architecture diagram](architecture/pipeline_diagram.md)
- [Memory strategy report](reports/memory_strategy_evaluation.md)
- [Python configuration](src/agent_config_stub.py)
- [Database schema and seed data](data/enterprise_schema.sql)

The assignment's architecture and configuration-stub deliverables are present.
SQL grounding, a follow-up calculation, semantic recall, buffer fallback, and
read-only SQL protections were verified. The explicit evaluation mode now indexes a synthetic wiki and records real model/tool evidence. See evidence/RUN_EVIDENCE.md for news status and scope; this is not a production-complete system.

## Setup

Prerequisite: install [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run all commands from the project root:

```bash
cd /path/to/enterprise-rag-agent
uv python install 3.13
uv sync --locked --python 3.13
```

Create `.env` only if it does not already exist, then edit it:

```bash
test -f .env || cp .env.example .env
nano .env
```

Fill in these settings using your actual credentials and gateway-supported model
IDs. The model IDs below worked in the verified demo; availability can change.

```dotenv
OPENAI_API_KEY=your_sprints_key
OPENAI_BASE_URL=https://management.sprints.ai/litellm/v1
OPENAI_CHAT_MODEL=gemini/gemini-3.8-flash
OPENAI_EMBEDDING_MODEL=gemini/gemini-embedding-001
SERPAPI_API_KEY=your_serpapi_key
ENTERPRISE_DB_PATH=./data/enterprise.db
VECTOR_ROOT=./runtime_vectors
```

Get the gateway key from Sprints and the news key from
[SerpApi](https://serpapi.com/manage-api-key). Keep `.env` private.
`LONG_TERM_MEMORY_EMBEDDER_MODEL` is also accepted as an embedding-model variable.
If earlier terminal exports override `.env`, clear them:

```bash
unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_API_BASE
unset OPENAI_CHAT_MODEL OPENAI_EMBEDDING_MODEL LONG_TERM_MEMORY_EMBEDDER_MODEL
unset SERPAPI_API_KEY ENTERPRISE_DB_PATH VECTOR_ROOT
```

Initialize the database once. This safely skips an existing database:

```bash
uv run python - <<'PY'
import sqlite3
from pathlib import Path

schema = Path("data/enterprise_schema.sql")
database = Path("data/enterprise.db")
if not schema.is_file():
    raise SystemExit(f"Schema not found: {schema}")
if database.exists():
    print(f"Already exists: {database}; unchanged.")
else:
    with sqlite3.connect(database) as connection:
        connection.executescript(schema.read_text())
    print(f"Created {database}")
PY
```

## Run

Verify SQL, follow-up handling, and semantic retrieval:

```bash
uv run --env-file .env python src/agent_config_stub.py --smoke-test
```

Expected: EMEA Q1 **980,000**, Q2 **1,100,000**, growth approximately **12.24%**,
and a final `PASS` message. The test sends demo schema and retrieved data to the
gateway and uses your API quota.

Start interactive chat; type `exit` to quit:

```bash
uv run --env-file .env python src/agent_config_stub.py
```

Ask a single question:

```bash
uv run --env-file .env python src/agent_config_stub.py --question "What is EMEA's Q2 revenue?"
```

Reuse a named local semantic-memory session:

```bash
uv run --env-file .env python src/agent_config_stub.py --session-id my-demo
```

Otherwise, each invocation starts a fresh session. Only semantic memory persists
across processes; the recent buffer is not restored.

Optionally try live news after supplying a valid SerpApi key:

```bash
uv run --env-file .env python src/agent_config_stub.py --question "Find recent public news about global oil markets and cite the source URLs."
```

Show available command options:

```bash
uv run python src/agent_config_stub.py --help
```

## Limits

Wiki retrieval needs approved documents indexed into its authorized collection;
normal chat does not ingest documents. --evaluate ingests only the synthetic test wiki. The actual news result is documented in evidence/RUN_EVIDENCE.md.
Changing embedding models may require re-indexing into a new vector directory.
Legacy-component deprecation warnings are expected for this assignment.
Production authentication, outbound-data filtering, retention, and strict token
budget enforcement are outside this stub.


## Resubmission: actual execution evidence

```bash
uv run python src/agent_config_stub.py --evaluate
```

The script loads the local `.env`; do not replace an existing key file.
The recorded four-turn run and limitations are in
[evidence/RUN_EVIDENCE.md](evidence/RUN_EVIDENCE.md).
Upload **submission.zip**: the requested architecture/report/script/schema, the
synthetic wiki needed for reproduction, and the execution evidence. The environment,
secrets, database binaries and generated vectors are excluded. Standalone setup
instructions for the minimal archive are included in RUN_EVIDENCE.md.
