"""Retry helper for LLM structured-output calls.

Groq'"'"'s json_schema mode and occasional provider blips produce transient
failures that succeed on retry. We retry on a known set of error classes
with exponential backoff. Non-retryable errors (auth, quota, code bugs,
bad input) propagate immediately.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import TypeVar

from langchain_core.exceptions import OutputParserException

from .logger import logger

T = TypeVar("T")

# Substrings matched case-insensitively against str(exception). These cover:
# - Groq strict-schema drops (json_validate_failed)
# - Rate limiting (429)
# - Transient provider 5xx
# - Network timeouts
_RETRYABLE_ERROR_PATTERNS: tuple[str, ...] = (
    "json_validate_failed",
    "rate_limit",
    "rate limit",
    "too many requests",
    "service_unavailable",
    "internal_error",
    "internal server error",
    "bad_gateway",
    "gateway_timeout",
    "timeout",
)


def _is_retryable(err: Exception, extra_patterns: Iterable[str] = ()) -> bool:
    """Return True if *err* matches a known transient-failure class."""
    if isinstance(err, OutputParserException):
        return True
    needle = str(err).lower()
    patterns = (*_RETRYABLE_ERROR_PATTERNS, *extra_patterns)
    return any(p in needle for p in patterns)


def invoke_with_retry(
    invoke: Callable[[], T],
    *,
    label: str,
    max_attempts: int = 3,
    base_delay_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    extra_retryable_patterns: Iterable[str] = (),
) -> T:
    """Call ``invoke()`` up to ``max_attempts`` times with exponential backoff.

    Args:
        invoke: Zero-argument callable that runs the LLM chain.
        label: Short name used in log messages (e.g. ``"score"``, ``"parse_resume"``).
        max_attempts: Total attempts including the first. Must be >= 1.
        base_delay_seconds: Delay before attempt 2 is ``base_delay_seconds``;
            before attempt 3 it is doubled, etc.
        sleep: Injection point for deterministic tests. Defaults to ``time.sleep``.
        extra_retryable_patterns: Additional case-insensitive substrings that
            should be treated as retryable.

    Returns:
        The return value of ``invoke()`` on first success.

    Raises:
        The last exception if all attempts fail, or immediately on a
        non-retryable error.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return invoke()
        except Exception as err:
            last_err = err
            is_last = attempt == max_attempts - 1
            if is_last or not _is_retryable(err, extra_retryable_patterns):
                raise
            delay = base_delay_seconds * (2**attempt)
            logger.warning(
                "%s retry %d/%d in %.1fs: %s",
                label,
                attempt + 1,
                max_attempts - 1,
                delay,
                str(err)[:200],
            )
            sleep(delay)

    # Unreachable: the loop either returns on success or raises. This keeps
    # the type checker happy.
    raise last_err if last_err else RuntimeError(f"{label} failed with no result")
