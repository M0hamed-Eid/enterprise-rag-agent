# Real-model agent execution evidence

Completed 9 November 2026. This is a fresh four-turn agent run, not restored
checkpoint output and not an offline/fake-model simulation.

The agent and nested SQL/QA chains called the configured Sprints OpenAI-compatible
model gateway. The response metadata reports `gemini/gemini-3.8-flash`.
This is a gateway model identifier, not a claim that a direct Google endpoint or
GPT-4o-mini was tested. Embeddings use the gateway's
`gemini/gemini-embedding-001`; Chroma recall was checked directly without the
resilient-memory fallback. Model availability depends on the account.

SQL data is the supplied assignment seed. The HR wiki is an explicitly synthetic
test policy in `data/demo_wiki.md`, indexed with source ID
`synthetic-demo-hybrid-work.md` / `DEMO-HR-001`. It is not actual corporate HR policy.

## Actual results

- Turn 1: EMEA Q1=980000, Q2=1100000, retrieved via generated and validated SQL.
- Turn 2: the follow-up resolved EMEA and computed growth=12.244897959183673%
  through a fresh SQL query; the answer rounded this to 12.24%.
- Turn 3: the vector QA tool retrieved the synthetic policy, answered two remote
  days per week with manager approval and cited DEMO-HR-001.
- Turn 4: Live SerpAPI returned articles with URLs; these are search snippets, not full-article verification.

| Turn | Observed tools | End-to-end duration | Model responses | Recorded total tokens |
| --- | --- | --- | --- | --- |
| 1 | EnterpriseSQLDB | 17.94s | 3 | 2503 |
| 2 | EnterpriseSQLDB | 18.47s | 3 | 3634 |
| 3 | HR_Ops_Wiki | 10.74s | 3 | 2542 |
| 4 | LiveMarketNews | 13.65s | 2 | 4106 |

Durations include memory load/save and tools. Token sums count provider-reported
chat-model usage only, exclude embeddings/search, and are not a billing statement.
This small run does not establish p50/p95 latency or multilingual recall quality.

## Evidence map

- `console_output.txt`: actual stdout/stderr from the evaluation command, including
  questions, observed tool inputs/results, final answers and checks.
- `live_run.json`: actual model responses, response/token metadata, ordered public
  tool events, SQL observations, answers, semantic recall documents and source hash.
- `local_checks.json`: separate local SQL read-only/validation and k=5 buffer checks.
  These supplement the real run; they are not represented as model inference.

No private chain-of-thought is requested. Non-streaming preserves provider usage
metadata in the gateway's agent responses. Deprecation warnings are retained;
the named classic components are used because the assignment explicitly asks for them.

## Reproduce in the local project

```bash
uv sync --locked
uv run python src/agent_config_stub.py --evaluate
```

The script loads this project's `.env`. It creates a fresh isolated evaluation
vector directory, indexes only the synthetic wiki, and runs all four questions.
It does not seed or modify the enterprise SQL database. Missing configuration or
failed grounding checks raise errors instead of producing substitute answers.

## Reproduce from the minimal submission archive

Open a terminal in the extracted `enterprise-rag-agent` folder:

```bash
uv venv --python 3.13
uv pip install 'langchain-classic==1.0.8' 'langchain-community==0.4.2' 'langchain-experimental==0.4.2' 'langchain-openai==1.7.3' 'langchain-chroma==1.1.0' 'sqlalchemy==2.1.8' 'sqlglot==30.29.0' 'google-search-results==2.4.2' 'python-dotenv==1.2.4'
```

Create a local `.env` with your own account credentials (never submit it):

```dotenv
OPENAI_API_KEY=your_gateway_key
OPENAI_BASE_URL=https://management.sprints.ai/litellm/v1
OPENAI_CHAT_MODEL=gemini/gemini-3.8-flash
OPENAI_EMBEDDING_MODEL=gemini/gemini-embedding-001
SERPAPI_API_KEY=your_serpapi_key
ENTERPRISE_DB_PATH=./data/enterprise.db
VECTOR_ROOT=./runtime_vectors
```

These model identifiers are those used in the recorded run; use supported
identifiers for your account. A direct OpenAI account can use its endpoint,
chat model and embedding model instead. Do not send real enterprise secrets.
Execution consumes the configured accounts' API quotas.

Initialize a NEW demo database from the supplied schema (existing files unchanged):

```bash
.venv/bin/python - <<'PYDB'
import sqlite3
from pathlib import Path
path = Path('data/enterprise.db')
if path.exists():
    print('Existing database kept unchanged')
else:
    with sqlite3.connect(path) as connection:
        connection.executescript(Path('data/enterprise_schema.sql').read_text())
PYDB
.venv/bin/python src/agent_config_stub.py --evaluate
```

The existing evidence is reviewable without credentials or new API requests.
The run exercises immediate follow-up and vector recall, not recall after eviction
from a full five-turn window. The report's token budgets remain labelled estimates.
