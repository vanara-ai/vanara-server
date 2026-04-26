"""Tests for the `invoke_with_retry` LLM retry helper."""

from unittest.mock import MagicMock

import pytest
from langchain_core.exceptions import OutputParserException

from app.llm_retry import invoke_with_retry


class _Counter:
    """Records sleep durations without actually sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def test_success_first_try_no_sleep() -> None:
    fn = MagicMock(return_value="ok")
    sleep = _Counter()

    out = invoke_with_retry(fn, label="score", sleep=sleep)

    assert out == "ok"
    assert fn.call_count == 1
    assert sleep.delays == []


def test_retry_on_json_validate_failed() -> None:
    err = Exception("Error code: 400 - json_validate_failed: missing actionable_steps")
    fn = MagicMock(side_effect=[err, "ok"])
    sleep = _Counter()

    out = invoke_with_retry(fn, label="score", sleep=sleep)

    assert out == "ok"
    assert fn.call_count == 2
    assert sleep.delays == [2.0]


def test_retry_on_rate_limit_429() -> None:
    err = Exception("Error code: 429 - rate_limit_exceeded")
    fn = MagicMock(side_effect=[err, "ok"])
    sleep = _Counter()

    out = invoke_with_retry(fn, label="parse", sleep=sleep)

    assert out == "ok"
    assert fn.call_count == 2
    assert sleep.delays == [2.0]


def test_retry_on_output_parser_exception() -> None:
    err = OutputParserException("Could not parse LLM output")
    fn = MagicMock(side_effect=[err, "ok"])
    sleep = _Counter()

    out = invoke_with_retry(fn, label="score", sleep=sleep)

    assert out == "ok"
    assert fn.call_count == 2


def test_exponential_backoff_doubles_delay() -> None:
    err = Exception("rate limit hit")
    fn = MagicMock(side_effect=[err, err, "ok"])
    sleep = _Counter()

    out = invoke_with_retry(fn, label="score", sleep=sleep, base_delay_seconds=1.0)

    assert out == "ok"
    assert fn.call_count == 3
    assert sleep.delays == [1.0, 2.0]


def test_raises_after_all_retries_exhausted() -> None:
    err = Exception("json_validate_failed: missing foo")
    fn = MagicMock(side_effect=[err, err, err])
    sleep = _Counter()

    with pytest.raises(Exception, match="json_validate_failed"):
        invoke_with_retry(fn, label="score", sleep=sleep)

    assert fn.call_count == 3
    # 2 retries only; final attempt does not sleep.
    assert sleep.delays == [2.0, 4.0]


def test_no_retry_on_auth_error() -> None:
    err = Exception("Error code: 401 - invalid_api_key")
    fn = MagicMock(side_effect=[err])
    sleep = _Counter()

    with pytest.raises(Exception, match="invalid_api_key"):
        invoke_with_retry(fn, label="score", sleep=sleep)

    assert fn.call_count == 1
    assert sleep.delays == []


def test_no_retry_on_code_bug() -> None:
    err = AttributeError("'NoneType' object has no attribute '.get'")
    fn = MagicMock(side_effect=[err])
    sleep = _Counter()

    with pytest.raises(AttributeError):
        invoke_with_retry(fn, label="score", sleep=sleep)

    assert fn.call_count == 1
    assert sleep.delays == []


def test_case_insensitive_pattern_match() -> None:
    err = Exception("INTERNAL SERVER ERROR at upstream")
    fn = MagicMock(side_effect=[err, "ok"])
    sleep = _Counter()

    out = invoke_with_retry(fn, label="score", sleep=sleep)

    assert out == "ok"
    assert fn.call_count == 2


def test_custom_max_attempts() -> None:
    err = Exception("timeout")
    fn = MagicMock(side_effect=[err, err, err, err, "ok"])
    sleep = _Counter()

    out = invoke_with_retry(fn, label="score", sleep=sleep, max_attempts=5)

    assert out == "ok"
    assert fn.call_count == 5
    assert sleep.delays == [2.0, 4.0, 8.0, 16.0]


def test_max_attempts_one_means_no_retries() -> None:
    err = Exception("timeout")
    fn = MagicMock(side_effect=[err])
    sleep = _Counter()

    with pytest.raises(Exception, match="timeout"):
        invoke_with_retry(fn, label="score", sleep=sleep, max_attempts=1)

    assert fn.call_count == 1
    assert sleep.delays == []


def test_invalid_max_attempts_zero() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        invoke_with_retry(lambda: None, label="score", max_attempts=0)


def test_extra_retryable_patterns() -> None:
    err = Exception("quota_exceeded on the premium tier")
    fn = MagicMock(side_effect=[err, "ok"])
    sleep = _Counter()

    out = invoke_with_retry(
        fn,
        label="score",
        sleep=sleep,
        extra_retryable_patterns=("quota_exceeded",),
    )

    assert out == "ok"
    assert fn.call_count == 2
