"""Logging helpers — specifically, redaction of sensitive request headers.

Never log anything that could contain an API key. We use an allow-list
(only log known-safe headers) rather than a deny-list, because a deny-list
is one missed entry away from leaking a secret.
"""

from collections.abc import Mapping

_SAFE_HEADERS = frozenset(
    {
        "host",
        "user-agent",
        "accept",
        "accept-language",
        "accept-encoding",
        "content-type",
        "content-length",
        "origin",
        "referer",
        "x-forwarded-for",
        "x-forwarded-proto",
        "x-real-ip",
        "x-user-id",
        "x-user-email",
        "x-user-name",
    }
)


def safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of headers containing only allow-listed (non-sensitive) entries.

    Anything not on the allow-list is dropped entirely — including
    X-Groq-Key, Authorization, Cookie, and any custom key header we
    might add later.
    """
    return {k: v for k, v in headers.items() if k.lower() in _SAFE_HEADERS}
