"""User-facing error classification for Vanara.ai.

Maps internal exception patterns to brand-aligned, actionable messages.
Never leak raw schema, stack traces, or LLM internals to the client.

Design principles:
- Name the actor ("Groq didn't recognize your key," not "auth failed")
- Say what to do next (every message ends with an action)
- Admit flakiness honestly (AI is occasionally flaky; users appreciate it)
- Keep jargon out (no "JSON schema," "500 Internal Server Error," etc.)
- Reserve the contact address for truly unexpected errors
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class UserError:
    """A classified, user-facing error response."""

    status_code: int
    message: str
    request_id: str | None = None

    @property
    def detail(self) -> str:
        """The string we actually send to the client as ``detail``."""
        if self.request_id:
            return f"{self.message} (ref: {self.request_id})"
        return self.message


_MSG_INVALID_KEY = (
    "Groq didn't recognize your API key. Double-check it in Settings, or create a fresh one at console.groq.com/keys."
)
_MSG_RATE_LIMIT = (
    "Groq is asking us to slow down. Give it a minute and try again. "
    "If you hit this often on the free tier, consider upgrading to Groq's "
    "Dev plan at console.groq.com/settings/billing."
)
_MSG_CONTEXT_TOO_LONG = (
    "Your resume is too long for the model to read in one pass. Try trimming older or less relevant roles."
)
_MSG_TIMEOUT = "Groq is taking too long to respond. Try again in a moment."
_MSG_UNAVAILABLE = "Groq is temporarily unavailable. Try again in a moment."
_MSG_AI_CONFUSED = "The AI got confused analyzing your resume. Usually works on retry — try again in a moment."
_MSG_PDF_UNREADABLE = "We couldn't read this PDF. If it's a scanned image, try exporting a text version first."
_MSG_PDF_EMPTY = "This PDF looks empty. Check the file and re-upload when ready."
_MSG_PDF_IMAGE_ONLY = "This PDF looks empty or is an image-only scan. Export a text PDF and try again."
_MSG_GENERIC = "Something broke on our end. Try again — if it keeps happening, drop us a note at vanara.ai@yahoo.com."


# Pattern -> (status, message). Substrings matched case-insensitively
# against str(exception). First match wins.
_ERROR_PATTERNS: tuple[tuple[str, int, str], ...] = (
    ("invalid_api_key", 401, _MSG_INVALID_KEY),
    ("401", 401, _MSG_INVALID_KEY),
    ("rate_limit", 429, _MSG_RATE_LIMIT),
    ("rate limit", 429, _MSG_RATE_LIMIT),
    ("too many requests", 429, _MSG_RATE_LIMIT),
    ("context_length_exceeded", 400, _MSG_CONTEXT_TOO_LONG),
    ("maximum context length", 400, _MSG_CONTEXT_TOO_LONG),
    ("timeout", 504, _MSG_TIMEOUT),
    ("gateway_timeout", 504, _MSG_TIMEOUT),
    ("service_unavailable", 503, _MSG_UNAVAILABLE),
    ("bad_gateway", 503, _MSG_UNAVAILABLE),
    ("json_validate_failed", 500, _MSG_AI_CONFUSED),
    ("outputparserexception", 500, _MSG_AI_CONFUSED),
    ("could not extract text", 400, _MSG_PDF_UNREADABLE),
    ("empty resume", 400, _MSG_PDF_EMPTY),
    ("no text extracted", 400, _MSG_PDF_IMAGE_ONLY),
)


def classify_error(err: Exception | str, *, with_request_id: bool = True) -> UserError:
    """Map an internal error to a safe user-facing response.

    Args:
        err: The original exception (or its str form).
        with_request_id: If True, attach a short request ID to generic (500)
            errors so users can reference logs when emailing support.

    Returns:
        A ``UserError`` whose ``detail`` is safe to send to the client.
    """
    needle = str(err).lower()

    for pattern, status, message in _ERROR_PATTERNS:
        if pattern in needle:
            # Known errors are user-fixable; no request ID noise.
            return UserError(status_code=status, message=message)

    # Unknown error: include a short ID so the user can reference our logs.
    request_id = _short_id() if with_request_id else None
    return UserError(status_code=500, message=_MSG_GENERIC, request_id=request_id)


def _short_id() -> str:
    """Return an 8-char lowercase hex ID for error correlation."""
    return uuid.uuid4().hex[:8]
