"""Tests for the user-facing error classifier."""

import re

import pytest
from app.errors import UserError, classify_error


def test_invalid_api_key_returns_401() -> None:
    err = Exception("Error code: 401 - invalid_api_key")
    out = classify_error(err)
    assert out.status_code == 401
    assert "Groq didn't recognize" in out.message
    assert "console.groq.com/keys" in out.message
    assert out.request_id is None


def test_bare_401_returns_401_with_auth_message() -> None:
    out = classify_error("401 Unauthorized")
    assert out.status_code == 401
    assert "Groq didn't recognize" in out.message


def test_rate_limit_returns_429() -> None:
    out = classify_error("Error: rate_limit_exceeded")
    assert out.status_code == 429
    assert "slow down" in out.message
    # Actionable upgrade hint for free-tier users.
    assert "Dev plan" in out.message
    assert "console.groq.com/settings/billing" in out.message


def test_rate_limit_spaced_phrase() -> None:
    out = classify_error("You have hit a rate limit.")
    assert out.status_code == 429


def test_too_many_requests_returns_429() -> None:
    out = classify_error("429 Too Many Requests")
    assert out.status_code == 429


def test_context_length_returns_400() -> None:
    out = classify_error("context_length_exceeded: 8192 tokens max")
    assert out.status_code == 400
    assert "too long" in out.message
    assert "trim" in out.message.lower()


def test_timeout_returns_504() -> None:
    out = classify_error("Request timed out after 30s: timeout")
    assert out.status_code == 504
    assert "too long to respond" in out.message


def test_service_unavailable_returns_503() -> None:
    out = classify_error("Error 503 service_unavailable")
    assert out.status_code == 503


def test_json_validate_failed_returns_500_ai_confused() -> None:
    out = classify_error("Error 400 - json_validate_failed: missing actionable_steps")
    assert out.status_code == 500
    assert "AI got confused" in out.message
    assert out.request_id is None


def test_output_parser_exception_returns_ai_confused() -> None:
    out = classify_error("OutputParserException: Could not parse...")
    assert out.status_code == 500
    assert "AI got confused" in out.message


def test_pdf_unreadable_returns_400() -> None:
    out = classify_error("Could not extract text from PDF")
    assert out.status_code == 400
    assert "couldn't read this PDF" in out.message


def test_empty_resume_returns_400() -> None:
    out = classify_error("Empty resume uploaded")
    assert out.status_code == 400
    assert "empty" in out.message.lower()


def test_unknown_error_returns_generic_500_with_request_id() -> None:
    out = classify_error(RuntimeError("something wildly unexpected"))
    assert out.status_code == 500
    assert "broke on our end" in out.message
    assert "vanara.ai@yahoo.com" in out.message
    assert out.request_id is not None
    assert re.match(r"^[0-9a-f]{8}$", out.request_id)


def test_unknown_error_detail_includes_ref() -> None:
    out = classify_error(RuntimeError("mystery"))
    assert f"(ref: {out.request_id})" in out.detail


def test_known_error_detail_has_no_ref() -> None:
    out = classify_error("rate_limit")
    assert "ref:" not in out.detail
    assert out.detail == out.message


def test_with_request_id_false_suppresses_id() -> None:
    out = classify_error("mystery bug", with_request_id=False)
    assert out.status_code == 500
    assert out.request_id is None
    assert "ref:" not in out.detail


def test_case_insensitive_matching() -> None:
    out = classify_error("RATE_LIMIT_EXCEEDED")
    assert out.status_code == 429


def test_user_error_is_immutable() -> None:
    out = classify_error("rate_limit")
    with pytest.raises((AttributeError, TypeError)):
        out.status_code = 999  # frozen dataclass


def test_unknown_errors_get_distinct_request_ids() -> None:
    a = classify_error("mystery a")
    b = classify_error("mystery b")
    assert a.request_id != b.request_id


def test_no_jargon_in_user_messages() -> None:
    """Every message should avoid developer jargon per brand voice."""
    banned = ("HTTPException", "JSON schema", "stack trace", "traceback", "500 Internal", "API response")
    test_errors = [
        "invalid_api_key",
        "rate_limit",
        "context_length_exceeded",
        "timeout",
        "json_validate_failed",
        "could not extract text",
        "mystery unknown bug",
    ]
    for err in test_errors:
        msg = classify_error(err).message
        for bad_word in banned:
            assert bad_word.lower() not in msg.lower(), f"'{bad_word}' leaked into: {msg}"


def test_first_pattern_wins() -> None:
    """'401' is before other patterns; 'Error code: 401 - rate_limit_exceeded'
    should match 401 (auth) first. This documents the precedence policy."""
    out = classify_error("Error code: 401 - rate_limit_exceeded")
    assert out.status_code == 401
