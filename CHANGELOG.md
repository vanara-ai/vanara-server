# Changelog

All notable changes to Vanara.ai backend are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-04-26

### Added
- Initial public release of Vanara.ai backend
- BYOK (Bring Your Own Key) architecture: Groq API key forwarded per-request via `X-Groq-Key` header, never persisted
- 5-agent resume optimization pipeline (planner, skills, experience, projects, scorer) built on LangGraph
- `/optimize_resume/`: end-to-end PDF upload → optimized PDF
- `/parse-resume/`: PDF → structured JSON (no LLM spend)
- `/optimize-from-parsed/`: optimize from cached parsed resume (Smart Library)
- `/resume-history`: paginated history with filters (score range, company, date)
- `/generate-pdf`: regenerate PDF from stored resume JSON
- `/feedback`: user feedback with optional SMTP delivery
- Optional Supabase persistence (history, parsed resumes, feedback, audit logs); backend runs stateless if unset
- Two resume templates: elegant (template_7) and classic (template_10)
- PDF generation via `xhtml2pdf` (pure Python, no native deps)
- Allow-list log redaction to prevent accidental PII leaks
- Dockerfile + docker-compose.yml for one-command dev
- CI workflow (GitHub Actions) with multi-version Python matrix
- Test suite (64 tests) covering PDF Unicode sanitization, pydantic schema validators, `get_groq_key` header enforcement, and `invoke_with_retry` transient-failure handling
- `requirements-dev.txt` with pinned development dependencies (pytest, pytest-asyncio, pytest-cov, ruff, pre-commit)
- `tests/conftest.py` that neutralizes `.env` loading and Supabase env vars during test collection
- `app/llm_retry.py`: reusable `invoke_with_retry` helper with exponential backoff for transient LLM failures (json_validate_failed, rate_limit, 5xx, OutputParserException)
- `max_completion_tokens=32768` on both `resume_parser.py` and `resume_optimizer.py` to prevent reasoning-token starvation on longer resumes with `gpt-oss-20b`/`gpt-oss-120b`

### Fixed
- Smart Library parse failures on resumes larger than ~6K characters caused by gpt-oss reasoning tokens consuming the default 2048 completion budget
- Non-deterministic `json_validate_failed` errors from Groq strict JSON schema mode when the LLM omits nullable fields (e.g. `actionable_steps`); the ATS score call now retries up to 2 times with exponential backoff
- `ATSScoreOutput` schema tightened: nullable fields on `SectionInstructions`, `ProfessionalExperienceInstructions`, `EducationInstructions`, `ProjectSpecificInstruction`; `extra="ignore"` to drop stray LLM fields

### Security
- No secrets in logs; allow-list redaction in `logging_utils.py`
- CORS configurable via `CORS_ORIGINS` env var; defaults to `localhost:3000`
- All external API keys sourced from env vars; `.env` in `.gitignore`

[Unreleased]: https://github.com/vanara-ai/vanara-server/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/vanara-ai/vanara-server/releases/tag/v1.0.0
