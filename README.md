# Vanara.ai: ResumeAI Backend

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4-orange)](https://langchain-ai.github.io/langgraph/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)

> **Open-source resume optimization. Giving it back to the community.**

A multi-agent FastAPI backend that iteratively rewrites résumés against a target
job description until it hits an ATS-style score threshold. Five specialised
LLM agents cooperate under a [LangGraph](https://langchain-ai.github.io/langgraph/)
orchestrator, all backed by [Groq](https://groq.com/)-hosted models.

**The frontend lives in a sibling repo:
[resumeaiui](https://github.com/vanara-ai/vanara-serverui).**

## ✨ Why it exists

Every commercial resume optimizer gates decent results behind a subscription,
and then sends your private resume + job data through *their* LLM account.
This project flips both: you bring your own Groq key, the server never sees
it, and
the whole stack is MIT-licensed and forkable.

> See our research: [arXiv:PENDING].

## 🔑 BYOK: Bring Your Own Key

- Every request must carry an `X-Groq-Key` header.
- Keys are used **only** for the request in progress. They are never logged,
  never cached, and never written to disk.
- Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

When no key is supplied the server returns `401` with a message pointing users
to the settings modal in the frontend.

## 🏗️ Architecture

```
                               ┌────────────────────────────┐
 POST /optimize_resume/        │  FastAPI (app/main.py)     │
 X-Groq-Key: gsk_...           │                            │
 resume.pdf + jobdesc   ─────▶ │  1. Dep extracts Groq key  │
                               │  2. Per-request            │
                               │     ResumeOptimizer        │
                               └──────────────┬─────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │  LangGraph state machine    │
                               │  (app/resume_optimizer.py)  │
                               │                             │
                               │    parse ──▶ score ──▶      │
                               │       ▲        │            │
                               │       └────────┘  (< 90)    │
                               │           │                 │
                               │           ▼                 │
                               │    orchestrate ─▶ 5 agents  │
                               │      summary / skills /     │
                               │      experience / projects /│
                               │      education              │
                               └──────────────┬──────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │ xhtml2pdf → PDF download    │
                               └─────────────────────────────┘
```

> 📖 **Deeper dive:** see [ARCHITECTURE.md](./ARCHITECTURE.md) for full request lifecycle diagrams, the 5-agent state machine, and module-by-module design notes.

## ⚡ Two Modes

| | Stateless (default) | Full (Supabase) |
|---|---|---|
| **Setup** | Zero config, just run | Set `SUPABASE_URL` + `SUPABASE_KEY` in `.env` |
| **Optimize** | ✅ | ✅ |
| **Resume history** | – | ✅ Per-user, filterable |
| **Smart Library** | – | ✅ Parse once, reuse across jobs |
| **Google sign-in** | – | ✅ (configure in Supabase dashboard) |
| **Feedback** | Accepted (not stored) | ✅ Stored + optional email |
| **Request audit log** | – | ✅ |

**Stateless** is perfect for local use or self-hosting without a database.
**Full** adds persistence. Set the two env vars and the app lights up automatically.

## 🚀 Quickstart

```bash
# 1. clone
git clone https://github.com/vanara-ai/vanara-server.git
cd resumeai

# 2. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. configure (all envs optional; .env.example has details)
cp .env.example .env

# 4. run
uvicorn app.main:app --reload
```

Browse http://localhost:8000/docs for the OpenAPI playground. Hit
`/health` to confirm persistence mode (`stateless` or `enabled`).

Or with Docker:

```bash
docker compose up --build
```

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

51 unit tests covering:
- **BYOK contract**: 401 without key, blank-key rejection, stateless history → 501, `/health` open
- **PDF Unicode sanitizer**: em-dashes, smart quotes, bullets, control chars, nested payloads
- **Pydantic schemas**: resume validation, environment/tech-stack line filtering in experience bullets
- **Auth helpers**: `get_groq_key` header enforcement

With coverage:
```bash
pytest tests/ --cov=app --cov-report=term-missing
```

The suite runs in ~1s and makes no network calls. LLM interactions are
verified via manual integration tests, not mocks.

## 📁 Project layout

```
app/
├── main.py               # FastAPI entrypoint + routes
├── deps.py               # BYOK dependency (X-Groq-Key header)
├── logging_utils.py      # Allow-list header redactor (safety-critical)
├── resume_optimizer.py   # ResumeOptimizer + LangGraph state machine
├── resume_parser.py      # Standalone ResumeParser used by /parse-resume
├── optimized_pipeline.py # PDF text → optimize → render orchestration
├── pdf_utils.py          # xhtml2pdf / Jinja2 PDF rendering
├── models.py             # Pydantic schemas (Resume, ATSScoreOutput, etc.)
├── cloud_taxonomy.py     # Role-level cloud-provider detection
├── database.py           # Optional Supabase persistence layer
├── email_service.py      # Optional feedback email (SMTP)
├── logger.py             # JSON-ish structured logger
├── constants.py          # PDF_SUFFIX, TEMPLATE_DIR
└── env.py                # load .env
templates/                # 8 HTML resume templates (Jinja2)
tests/                    # pytest suite (BYOK, sanitizer, schemas, auth helpers)
```

## 🔐 Security

- **No API keys on the server.** Every LLM call is driven by a
  per-request key. There is no env-var fallback in the open-source path.
- **Header redaction.** The request-audit log in Supabase (when enabled)
  writes an **allow-list** of safe headers only (`host`, `user-agent`,
  `x-user-*`, etc.). `X-Groq-Key`, `Authorization`, `Cookie` are dropped.
  See [`app/logging_utils.py`](./app/logging_utils.py).
- **Path-traversal guard** on the `/download/{filename}` endpoint.
- **CORS** is explicit-origin (`CORS_ORIGINS` env var); no `*` + credentials.

Report issues per [SECURITY.md](./SECURITY.md).

## 🛠️ API

| Endpoint | Method | Requires Groq key | Requires Supabase |
|---|---|---|---|
| `/health` | GET | – | – |
| `/optimize_resume/` | POST | ✅ | optional (for history) |
| `/parse-resume/` | POST | ✅ | optional (for cache) |
| `/download/{filename}` | GET | – | – |
| `/feedback` | POST | – | optional |
| `/resume-history` | GET | – | ✅ (else 501) |
| `/parsed-resumes/` | GET/DELETE | – | ✅ |
| `/optimize-from-parsed/` | POST | ✅ | ✅ |
| `/generate-pdf` | POST | – | ✅ |
| `/analytics/requests` | GET | – | ✅ |

> 💡 **Interactive API docs** auto-generated by FastAPI at [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI) and [http://localhost:8000/redoc](http://localhost:8000/redoc) (ReDoc) when the server is running.

## 🤝 Contributing

Contributions of any size are welcome: typo fixes, better prompts, new
templates, agent improvements, bug reports, design critiques. You do not
need permission to open an issue or send a PR.

Start with [CONTRIBUTING.md](./CONTRIBUTING.md). It has the quick-start
setup and the full PR checklist. By contributing you agree to the
[Code of Conduct](./CODE_OF_CONDUCT.md).

## 🙏 Acknowledgments

Built by three friends who got tired of paying for resume optimizers.
See [AUTHORS](./AUTHORS) for the core team and the GitHub contributors
page for everyone who has pitched in.

## 📜 License

MIT. See [LICENSE](./LICENSE). Contributions welcomed from everyone.
