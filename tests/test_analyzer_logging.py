# -*- coding: utf-8 -*-
"""Regression tests for analyzer LLM logging privacy controls."""

from types import SimpleNamespace

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

import logging

import pytest

from src.analyzer import (
    AnalysisResult,
    GeminiAnalyzer,
    _should_log_llm_content_preview,
)
from src.logging_config import (
    is_sensitive_log_preview_enabled,
    set_sensitive_log_preview_enabled,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_sensitive_preview_flag():
    set_sensitive_log_preview_enabled(False)
    yield
    set_sensitive_log_preview_enabled(False)


def _make_config(*, log_level: str = "INFO", debug: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        gemini_request_delay=0,
        report_language="zh",
        litellm_model="gemini/gemini-2.5-flash",
        llm_temperature=0.2,
        report_integrity_enabled=False,
        report_integrity_retry=0,
        log_level=log_level,
        debug=debug,
    )


def _make_analyzer(
    config: SimpleNamespace,
    prompt: str,
    response_text: str,
    model_used: str = None,
) -> GeminiAnalyzer:
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._config_override = config
    analyzer._requested_skills = None
    analyzer._skill_instructions_override = None
    analyzer._default_skill_policy_override = None
    analyzer._use_legacy_default_prompt_override = None
    analyzer._resolved_prompt_state = None
    analyzer._router = None
    analyzer._litellm_available = True
    analyzer._get_analysis_system_prompt = lambda *args, **kwargs: "system prompt"
    analyzer._format_prompt = lambda *args, **kwargs: prompt
    analyzer._call_litellm = lambda *args, **kwargs: (
        response_text,
        model_used or config.litellm_model,
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    analyzer._build_market_snapshot = lambda *args, **kwargs: {"price": "123.45"}
    analyzer._parse_response = lambda *args, **kwargs: AnalysisResult(
        code="600519",
        name="贵州茅台",
        sentiment_score=66,
        trend_prediction="看多",
        operation_advice="持有",
        analysis_summary="总结",
    )
    return analyzer


def test_should_log_llm_content_preview_accepts_existing_debug_switches():
    assert not _should_log_llm_content_preview(_make_config())
    assert _should_log_llm_content_preview(_make_config(log_level="DEBUG"))
    assert _should_log_llm_content_preview(_make_config(debug=True))

    set_sensitive_log_preview_enabled(True)

    assert _should_log_llm_content_preview(_make_config())


def test_setup_logging_only_enables_sensitive_preview_for_explicit_debug_levels(tmp_path):
    setup_logging(log_dir=str(tmp_path / "notset"), console_level=logging.NOTSET)
    assert not is_sensitive_log_preview_enabled()

    setup_logging(log_dir=str(tmp_path / "debug"), console_level=logging.DEBUG)
    assert is_sensitive_log_preview_enabled()


def test_analyze_does_not_log_prompt_or_response_preview_by_default(caplog, monkeypatch):
    prompt = "Authorization: Bearer raw-secret-token\napi_key=sk-live-123456"
    response_text = "password=hunter2\nsession_id=abc123"
    analyzer = _make_analyzer(_make_config(), prompt, response_text)
    monkeypatch.setattr("src.analyzer.persist_llm_usage", lambda *args, **kwargs: None)

    with caplog.at_level(logging.DEBUG, logger="src.analyzer"):
        result = analyzer.analyze({"code": "600519", "stock_name": "贵州茅台"}, news_context="news")

    assert result.raw_response == response_text
    assert "LLM Prompt 调试预览" not in caplog.text
    assert "LLM返回 调试预览" not in caplog.text
    assert prompt not in caplog.text
    assert response_text not in caplog.text
    assert "[LLM配置] Prompt 长度:" in caplog.text
    assert "[LLM返回]" in caplog.text


def test_analyze_logs_only_redacted_single_line_preview_in_debug_mode(caplog, monkeypatch):
    long_tail = "超长调试内容" * 80
    prompt = (
        "Authorization: Bearer raw-secret-token\n"
        "api_key=sk-live-123456 password=hunter2\n"
        f"{long_tail}"
    )
    response_text = (
        "email=user@example.com\n"
        "session_id=abc123 password=open-sesame\n"
        f"{long_tail}"
    )
    analyzer = _make_analyzer(_make_config(log_level="DEBUG"), prompt, response_text)
    monkeypatch.setattr("src.analyzer.persist_llm_usage", lambda *args, **kwargs: None)

    with caplog.at_level(logging.DEBUG, logger="src.analyzer"):
        result = analyzer.analyze({"code": "600519", "stock_name": "贵州茅台"}, news_context=None)

    assert result.raw_response == response_text

    preview_messages = [record.getMessage() for record in caplog.records if "调试预览" in record.getMessage()]
    assert len(preview_messages) == 2

    prompt_preview = next(msg for msg in preview_messages if "LLM Prompt 调试预览" in msg)
    response_preview = next(msg for msg in preview_messages if "LLM返回 调试预览" in msg)

    assert "\n" not in prompt_preview
    assert "\n" not in response_preview
    assert "Authorization=Bearer [REDACTED]" in prompt_preview
    assert "api_key=[REDACTED]" in prompt_preview
    assert "password=[REDACTED]" in prompt_preview
    assert "[REDACTED_EMAIL]" in response_preview
    assert "session_id=[REDACTED]" in response_preview
    assert "password=[REDACTED]" in response_preview
    assert "raw-secret-token" not in caplog.text
    assert "sk-live-123456" not in caplog.text
    assert "hunter2" not in caplog.text
    assert "user@example.com" not in caplog.text
    assert "abc123" not in caplog.text
    assert "open-sesame" not in caplog.text
    assert "..." in prompt_preview
    assert "..." in response_preview


def test_analyze_logs_actual_model_used_in_response_metadata(caplog, monkeypatch):
    configured_model = "gemini/gemini-2.5-flash"
    actual_model = "openai/gpt-4.1-mini"
    analyzer = _make_analyzer(
        _make_config(log_level="INFO"),
        "prompt",
        "response",
        model_used=actual_model,
    )
    analyzer._config_override.litellm_model = configured_model
    monkeypatch.setattr("src.analyzer.persist_llm_usage", lambda *args, **kwargs: None)

    with caplog.at_level(logging.INFO, logger="src.analyzer"):
        result = analyzer.analyze({"code": "600519", "stock_name": "贵州茅台"}, news_context=None)

    assert result.model_used == actual_model
    assert f"[LLM配置] 模型: {configured_model}" in caplog.text
    assert f"[LLM返回] {actual_model} 响应成功" in caplog.text
    assert f"[LLM返回] {configured_model} 响应成功" not in caplog.text
