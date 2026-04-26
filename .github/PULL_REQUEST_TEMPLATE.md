## What does this PR do?

<!-- One or two sentences. What behavior changes? -->

## Why?

<!-- Link to issue if applicable, or describe the motivation. -->

Closes #

## How was this tested?

- [ ] `pytest tests/ -v` passes locally
- [ ] `python -m compileall -q app/` is clean
- [ ] Added/updated tests for new behavior
- [ ] Manually smoke-tested against a running instance (describe below)

<!-- Describe your manual testing if applicable. -->

## Design principles

- [ ] BYOK preserved — no server-side LLM keys introduced
- [ ] Supabase persistence remains optional (or new feature explicitly requires it)
- [ ] Any new header/body logged via `safe_headers()` allow-list, not raw

## Breaking changes

- [ ] None
- [ ] Yes (describe migration below)

<!-- If yes, what do existing users need to do? -->
