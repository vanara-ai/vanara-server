# Security Policy

## Reporting a Vulnerability

If you discover a security issue in Vanara.ai, please **do not** open a public GitHub issue.

Instead, email the maintainer directly:

**sinduku1@depaul.edu**

Include:
- A description of the issue and its potential impact
- Steps to reproduce (proof-of-concept code welcome)
- Your name/handle if you'd like credit

You'll receive acknowledgment within 72 hours. Fixes for confirmed issues are typically released within 14 days, coordinated with you before public disclosure.

## Scope

### In scope
- Auth/session bypass on the backend API
- Injection attacks (SQL, prompt injection with material impact, path traversal)
- Leakage of user API keys, request bodies, or parsed resume content through logs or responses
- CORS or CSRF bypass enabling credentialed requests from untrusted origins
- Docker image hardening issues

### Out of scope
- Rate limiting or DoS via the public `/optimize_resume/` endpoint (requests are priced to the caller's own Groq key — this is by design)
- Vulnerabilities in third-party services (Groq, Supabase, Sentry — report to those vendors)
- Issues requiring a compromised Groq API key supplied by the user themselves
- Social-engineering or phishing scenarios

## Supported Versions

Only the `main` branch receives security updates. Deployed instances should pull from `main` regularly.

## Design Principles

Vanara.ai is a BYOK (bring-your-own-key) service:
- User LLM keys are **never stored server-side** — they're sent per-request via the `X-Groq-Key` header and held in the caller's browser localStorage
- Request metadata logged to Supabase (if enabled) uses an **allow-list** of non-sensitive fields — no auth headers, no bodies
- Supabase and SMTP are **optional** — the core optimizer runs stateless
