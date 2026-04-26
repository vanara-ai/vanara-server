# Contributing to Vanara.ai

This project started because three of us were tired of paying for resume
optimizers and watching our own data end up in someone else's LLM account.
We open-sourced it so anyone can self-host, fork, or just read how it works.

Contributions of any size are welcome: typo fixes, better prompts, new
templates, agent improvements, bug reports, design critiques. You do not
need permission to open an issue or send a PR.

If you're new to open source, read this whole doc once before your first
PR. If you've done this before, skip to [Pull Request Process](#pull-request-process).

## Quick Start

```bash
git clone https://github.com/vanara-ai/vanara-server.git
cd resumeai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
cp .env.example .env
uvicorn app.main:app --reload
```

Run tests:
```bash
pytest tests/ -v                           # 51 tests, ~1s
pytest tests/ --cov=app                    # with coverage
ruff check app/ tests/                     # lint
ruff format --check app/ tests/            # format check
```

## Ways to Contribute

- **Bug reports**: open a GitHub issue using the bug report template
- **Feature requests**: open a GitHub issue using the feature request template
- **Code**: see "Pull Request Process" below
- **Docs**: typo fixes, clearer explanations, more examples, all welcome
- **New agents/tools**: the pipeline is plugin-friendly. Add a new specialized agent in `app/agents/` and wire it in `optimized_pipeline.py`

## Pull Request Process

1. **Fork** the repo and create a topic branch: `git checkout -b feat/your-thing`
2. **Write tests** for new behavior. We don't have a strict coverage bar, but untested code is harder to accept.
3. **Keep PRs focused**: one concern per PR. Split mechanical refactors from behavior changes.
4. **Match existing style.** Python uses type hints; avoid bare `except`.
5. **Run the full check locally:**
   ```bash
   ruff check app/ tests/
   ruff format --check app/ tests/
   pytest tests/ -v
   python -m compileall -q app/
   ```
   Or just `pre-commit run --all-files` after installing hooks.
6. **Fill in the PR template**: what, why, how tested.
7. **Update the README** if you change public behavior (API endpoints, env vars, deps).

## Code Style

- Python 3.11+. Use modern union syntax (`X | None`), PEP 604 unions, `match` if it helps
- Type hints on public functions
- No new top-level dependencies without discussion. Open an issue first
- LLM model names stay as env-var-overridable constants, not scattered strings
- Ruff handles formatting + lint; don't hand-format

## Commit Messages

Conventional-ish: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`. One-line subject, wrap at 72. Describe *why* in the body if non-obvious.

## Design Principles (please preserve)

- **BYOK forever.** The server never stores LLM credentials. Don't add features that require a shared server-side key.
- **Stateless-capable.** Supabase is optional. Core optimization must work with zero persistence configured.
- **Allow-list logging.** Any new request metadata piped to the database goes through `safe_headers()`. Don't log auth, don't log bodies.

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). By participating, you agree to its terms.

## Questions?

Open an issue with the `question` label, or email sinduku1@depaul.edu.
