"""Tests for ``app.deps.get_groq_key`` — BYOK header enforcement.

The function is the single authorization gate for every endpoint that
calls an LLM. A bug here would either (a) let unauthenticated requests
through, or (b) reject valid keys. Both are critical.

These tests call the dependency function directly rather than through
FastAPI's dependency injection — see ``test_byok.py`` for end-to-end
coverage of the HTTP surface.
"""

import pytest
from app.deps import get_groq_key
from fastapi import HTTPException


class TestGetGroqKey:
    def test_none_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_groq_key(x_groq_key=None)
        assert exc_info.value.status_code == 401
        assert "X-Groq-Key" in exc_info.value.detail

    def test_empty_string_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_groq_key(x_groq_key="")
        assert exc_info.value.status_code == 401

    def test_whitespace_only_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_groq_key(x_groq_key="   ")
        assert exc_info.value.status_code == 401

    def test_tab_and_newline_only_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_groq_key(x_groq_key="\t\n")
        assert exc_info.value.status_code == 401

    def test_valid_key_returned_trimmed(self):
        assert get_groq_key(x_groq_key="gsk_abc123") == "gsk_abc123"

    def test_valid_key_with_surrounding_whitespace_trimmed(self):
        assert get_groq_key(x_groq_key="  gsk_abc123  ") == "gsk_abc123"

    def test_error_message_points_users_to_console(self):
        # The error surface should tell the user WHERE to get a key.
        with pytest.raises(HTTPException) as exc_info:
            get_groq_key(x_groq_key=None)
        detail = exc_info.value.detail
        assert "console.groq.com" in detail
        assert "browser" in detail  # reassures users the key stays client-side
