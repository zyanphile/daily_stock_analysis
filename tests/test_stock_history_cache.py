# -*- coding: utf-8 -*-
"""Tests for shared stock history cache behavior."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict

import pandas as pd

from src.services.stock_history_cache import (
    AGENT_HISTORY_BASELINE_DAYS,
    ensure_min_history_cached,
    load_recent_history_df,
    reset_shared_history_runtime,
)


@dataclass
class _Bar:
    code: str
    date: date
    close: float
    data_source: str = "seed"

    def to_dict(self):
        return {
            "code": self.code,
            "date": self.date,
            "open": self.close - 1,
            "high": self.close + 1,
            "low": self.close - 2,
            "close": self.close,
            "volume": 1000.0,
            "amount": 10000.0,
            "pct_chg": 1.0,
            "ma5": self.close,
            "ma10": self.close,
            "ma20": self.close,
            "volume_ratio": 1.0,
            "data_source": self.data_source,
        }


class _DummyDB:
    def __init__(self):
        self._rows: Dict[str, Dict[date, _Bar]] = {}

    def seed(self, code: str, count: int, *, end_date: date, source: str = "seed") -> None:
        self._rows.setdefault(code, {})
        for idx in range(count):
            current_date = end_date - timedelta(days=count - idx - 1)
            self._rows[code][current_date] = _Bar(
                code=code,
                date=current_date,
                close=100 + idx,
                data_source=source,
            )

    def get_latest_data(self, code: str, days: int = 2):
        rows = sorted(self._rows.get(code, {}).values(), key=lambda bar: bar.date, reverse=True)
        return rows[:days]

    def get_data_range(self, code: str, start_date: date, end_date: date):
        rows = [
            bar
            for bar in self._rows.get(code, {}).values()
            if start_date <= bar.date <= end_date
        ]
        return sorted(rows, key=lambda bar: bar.date)

    def save_daily_data(self, df: pd.DataFrame, code: str, data_source: str = "Unknown") -> int:
        bucket = self._rows.setdefault(code, {})
        new_count = 0
        for row in df.to_dict(orient="records"):
            row_date = row["date"]
            if isinstance(row_date, pd.Timestamp):
                row_date = row_date.date()
            if row_date not in bucket:
                new_count += 1
            bucket[row_date] = _Bar(
                code=code,
                date=row_date,
                close=float(row.get("close", 0) or 0),
                data_source=data_source,
            )
        return new_count


def _make_history_df(code: str, count: int, *, end_date: date) -> pd.DataFrame:
    rows = []
    for idx in range(count):
        current_date = end_date - timedelta(days=count - idx - 1)
        close = 200 + idx
        rows.append(
            {
                "date": current_date,
                "open": close - 1,
                "high": close + 1,
                "low": close - 2,
                "close": close,
                "volume": 1000.0,
                "amount": 10000.0,
                "pct_chg": 1.0,
                "ma5": close,
                "ma10": close,
                "ma20": close,
                "volume_ratio": 1.0,
            }
        )
    return pd.DataFrame(rows)


class StockHistoryCacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        reset_shared_history_runtime()

    def tearDown(self) -> None:
        reset_shared_history_runtime()

    def test_load_recent_history_uses_db_when_sufficient(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 240, end_date=target_date, source="seed")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        with patch("src.services.stock_history_cache.get_db", return_value=db):
            df, source = load_recent_history_df(
                "600519",
                days=120,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertEqual(len(df), 120)
        self.assertEqual(source, "seed")
        manager.get_daily_data.assert_not_called()

    def test_ensure_min_history_cached_backfills_once_and_saves(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 30, end_date=target_date, source="seed")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.return_value = (
            _make_history_df("600519", AGENT_HISTORY_BASELINE_DAYS, end_date=target_date),
            "Fetcher",
        )

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            ok, source = ensure_min_history_cached(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )
            df, df_source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertTrue(ok)
        self.assertEqual(source, "Fetcher")
        self.assertEqual(df_source, "Fetcher")
        self.assertEqual(len(df), 60)
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)

    def test_short_history_is_not_refetched_twice_in_same_process(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.return_value = (
            _make_history_df("600519", 20, end_date=target_date),
            "Fetcher",
        )

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            first_df, first_source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )
            second_df, second_source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertEqual(len(first_df), 20)
        self.assertEqual(len(second_df), 20)
        self.assertEqual(first_source, "Fetcher")
        self.assertEqual(second_source, "Fetcher")
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)

    def test_stale_history_without_target_date_triggers_refresh(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 240, end_date=target_date - timedelta(days=1), source="seed")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.return_value = (
            _make_history_df("600519", AGENT_HISTORY_BASELINE_DAYS, end_date=target_date),
            "Fetcher",
        )

        with patch("src.services.stock_history_cache.get_db", return_value=db), patch(
            "src.core.trading_calendar.get_market_for_stock",
            return_value="cn",
        ), patch(
            "src.core.trading_calendar.get_effective_trading_date",
            return_value=target_date,
        ):
            df, source = load_recent_history_df(
                "600519",
                days=120,
                fetcher_manager=manager,
            )

        self.assertEqual(len(df), 120)
        self.assertEqual(source, "Fetcher")
        self.assertEqual(df.iloc[-1]["date"], target_date)
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)

    def test_normalized_code_lookup_hits_existing_db_cache(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 240, end_date=target_date, source="seed")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        with patch("src.services.stock_history_cache.get_db", return_value=db):
            df, source = load_recent_history_df(
                "SH600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertEqual(len(df), 60)
        self.assertEqual(source, "seed")
        manager.get_daily_data.assert_not_called()

    def test_backfill_saves_under_normalized_code(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.return_value = (
            _make_history_df("600519", AGENT_HISTORY_BASELINE_DAYS, end_date=target_date),
            "Fetcher",
        )

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            ok, source = ensure_min_history_cached(
                "600519.SH",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertTrue(ok)
        self.assertEqual(source, "Fetcher")
        self.assertIn("600519", db._rows)
        self.assertNotIn("600519.SH", db._rows)
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)

    def test_failed_fetch_is_not_retried_twice_in_same_process(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.side_effect = RuntimeError("boom")

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            first_df, first_source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )
            second_df, second_source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertTrue(first_df.empty)
        self.assertTrue(second_df.empty)
        self.assertIn("boom", first_source)
        self.assertIn("boom", second_source)
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)


if __name__ == "__main__":
    unittest.main()
