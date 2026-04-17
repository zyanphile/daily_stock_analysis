# -*- coding: utf-8 -*-
"""Regression tests for realtime quote fallback logging semantics."""

import asyncio
import importlib.util
import logging
import sys
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

try:
    json_repair_available = importlib.util.find_spec("json_repair") is not None
except ValueError:
    json_repair_available = "json_repair" in sys.modules

if not json_repair_available and "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

from data_provider.base import DataFetcherManager
from data_provider.realtime_types import RealtimeSource, UnifiedRealtimeQuote
from src.core.pipeline import StockAnalysisPipeline
from src.enums import ReportType


class _DummyFetcher:
    def __init__(self, name: str, priority: int, result=None, error: Exception | None = None):
        self.name = name
        self.priority = priority
        self._result = result
        self._error = error

    def get_realtime_quote(self, *args, **kwargs):
        if self._error is not None:
            raise self._error
        return self._result


def _make_quote(code: str = "600519", name: str = "贵州茅台") -> UnifiedRealtimeQuote:
    return UnifiedRealtimeQuote(
        code=code,
        name=name,
        source=RealtimeSource.AKSHARE_EM,
        price=1688.0,
        change_pct=1.2,
    )


def _make_pipeline(enable_realtime_quote: bool, realtime_quote=None) -> StockAnalysisPipeline:
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.config = SimpleNamespace(
        enable_realtime_quote=enable_realtime_quote,
        enable_chip_distribution=True,
        agent_mode=False,
        agent_skills=[],
        fundamental_stage_timeout_seconds=1.5,
        report_language="zh",
    )
    pipeline.fetcher_manager = MagicMock()
    pipeline.fetcher_manager.get_stock_name.return_value = "贵州茅台"
    pipeline.fetcher_manager.get_realtime_quote.return_value = realtime_quote
    pipeline.fetcher_manager.get_chip_distribution.return_value = None
    pipeline.fetcher_manager.get_fundamental_context.return_value = {
        "source_chain": [],
        "coverage": {},
    }
    pipeline.fetcher_manager.build_failed_fundamental_context.return_value = {
        "source_chain": [],
        "coverage": {},
    }
    pipeline.db = MagicMock()
    pipeline.db.save_fundamental_snapshot.return_value = None
    pipeline.db.get_data_range.return_value = []
    pipeline.db.get_analysis_context.return_value = {}
    pipeline.search_service = SimpleNamespace(is_available=False)
    pipeline.social_sentiment_service = SimpleNamespace(is_available=False)
    pipeline.trend_analyzer = MagicMock()
    pipeline.analyzer = MagicMock()
    pipeline.analyzer.analyze.return_value = None
    pipeline._attach_belong_boards_to_fundamental_context = MagicMock(side_effect=lambda code, ctx: ctx)
    pipeline._enhance_context = MagicMock(return_value={"realtime": {}})
    pipeline.save_context_snapshot = False
    return pipeline


@patch("src.config.get_config")
def test_manager_does_not_warn_when_fallback_source_succeeds(mock_get_config, caplog):
    mock_get_config.return_value = SimpleNamespace(
        enable_realtime_quote=True,
        realtime_source_priority="efinance,akshare_em",
    )
    manager = DataFetcherManager(
        fetchers=[
            _DummyFetcher("EfinanceFetcher", 0, error=RuntimeError("efinance timeout")),
            _DummyFetcher("AkshareFetcher", 1, result=_make_quote()),
        ]
    )

    with caplog.at_level(logging.INFO):
        quote = manager.get_realtime_quote("600519")

    assert quote is not None
    assert quote.name == "贵州茅台"
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert "所有数据源均不可用" not in caplog.text


def test_pipeline_warns_once_when_all_realtime_sources_fail(caplog):
    pipeline = _make_pipeline(enable_realtime_quote=True, realtime_quote=None)

    with caplog.at_level(logging.INFO):
        result = pipeline.analyze_stock("600519", ReportType.SIMPLE, "q1")

    assert result is None
    pipeline.fetcher_manager.get_stock_name.assert_called_once_with("600519", allow_realtime=False)
    pipeline.fetcher_manager.get_realtime_quote.assert_called_once_with("600519", log_final_failure=False)
    downgrade_logs = [
        record.message
        for record in caplog.records
        if "历史收盘价继续分析" in record.message
    ]
    assert downgrade_logs == ["贵州茅台(600519) 所有实时行情数据源均不可用，已降级为历史收盘价继续分析"]


@patch("src.config.get_config")
def test_event_monitor_keeps_manager_failure_summary_for_direct_quote_call(mock_get_config, caplog):
    from src.agent.events import EventMonitor, PriceAlert

    mock_get_config.return_value = SimpleNamespace(
        enable_realtime_quote=True,
        realtime_source_priority="efinance",
    )
    manager = DataFetcherManager(
        fetchers=[
            _DummyFetcher("EfinanceFetcher", 0, error=RuntimeError("efinance timeout")),
        ]
    )
    monitor = EventMonitor()
    rule = PriceAlert(stock_code="600519", direction="above", price=1800.0)

    async def _run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("data_provider.DataFetcherManager", return_value=manager), patch(
        "src.agent.events.asyncio.to_thread", new=_run_inline
    ), caplog.at_level(logging.INFO):
        result = asyncio.run(monitor._check_price(rule))

    assert result is None
    assert "[实时行情] 600519 所有数据源均失败: [efinance] 失败: efinance timeout" in caplog.text


def test_pipeline_logs_disabled_realtime_once_without_fetching_quote(caplog):
    pipeline = _make_pipeline(enable_realtime_quote=False, realtime_quote=_make_quote())

    with caplog.at_level(logging.INFO):
        result = pipeline.analyze_stock("600519", ReportType.SIMPLE, "q1")

    assert result is None
    pipeline.fetcher_manager.get_stock_name.assert_called_once_with("600519", allow_realtime=False)
    pipeline.fetcher_manager.get_realtime_quote.assert_not_called()
    downgrade_logs = [
        record.message
        for record in caplog.records
        if "历史收盘价继续分析" in record.message
    ]
    assert downgrade_logs == ["贵州茅台(600519) 实时行情已禁用，使用历史收盘价继续分析"]


def test_pipeline_trend_lookup_normalizes_prefixed_stock_code():
    pipeline = _make_pipeline(enable_realtime_quote=False, realtime_quote=None)
    pipeline.config.agent_mode = True
    pipeline.db.get_data_range.side_effect = [[], []]
    pipeline._analyze_with_agent = MagicMock(return_value=None)

    pipeline.analyze_stock("SH600519", ReportType.SIMPLE, "q1")

    first_call = pipeline.db.get_data_range.call_args_list[0]
    assert first_call.args[0] == "600519"


def test_pipeline_trend_lookup_prefers_more_complete_original_bucket():
    pipeline = _make_pipeline(enable_realtime_quote=False, realtime_quote=None)
    pipeline.config.agent_mode = True
    pipeline._analyze_with_agent = MagicMock(return_value=None)
    pipeline.trend_analyzer.analyze.return_value = SimpleNamespace(
        trend_status=SimpleNamespace(value="震荡"),
        buy_signal=SimpleNamespace(value="观望"),
        signal_score=60,
    )

    def _make_bars(code: str, count: int, *, end_date: date):
        bars = []
        for idx in range(count):
            current_date = end_date - timedelta(days=count - idx - 1)
            close = 100 + idx
            bar = MagicMock()
            bar.date = current_date
            bar.to_dict.return_value = {
                "date": current_date,
                "open": close - 1,
                "high": close + 1,
                "low": close - 2,
                "close": close,
                "volume": 1000.0,
            }
            bars.append(bar)
        return bars

    pipeline.db.get_data_range.side_effect = [
        _make_bars("600519", 10, end_date=date(2026, 4, 16)),
        _make_bars("SH600519", 60, end_date=date(2026, 4, 16)),
    ]

    pipeline.analyze_stock("SH600519", ReportType.SIMPLE, "q1")

    analyze_call = pipeline.trend_analyzer.analyze.call_args
    assert len(analyze_call.args[0]) == 60
    assert pipeline.db.get_data_range.call_args_list[0].args[0] == "600519"
    assert pipeline.db.get_data_range.call_args_list[1].args[0] == "SH600519"
