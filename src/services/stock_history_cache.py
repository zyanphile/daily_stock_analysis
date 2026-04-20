# -*- coding: utf-8 -*-
"""Shared DB-first stock history cache for agent-facing K-line access."""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from threading import Lock, RLock
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_provider.base import canonical_stock_code, normalize_stock_code
from src.storage import get_db

logger = logging.getLogger(__name__)

AGENT_HISTORY_BASELINE_DAYS = 240

_shared_fetcher_manager = None
_shared_fetcher_manager_lock = Lock()

# Agent 路径下 pipeline 冻结的目标交易日。由 `_analyze_with_agent` 在进入时 set、
# 退出时 reset，tool handler 通过 `get_agent_frozen_target_date()` 读取后
# 透传给 `load_recent_history_df(..., target_date=...)`，确保同一次 Agent 分析
# 在跨收盘边界场景下始终以 pipeline 冻结的 T 日作为预期交易日，而不是 tool 调用
# 时的墙钟。
_agent_frozen_target_date: contextvars.ContextVar[Optional[date]] = contextvars.ContextVar(
    "agent_frozen_target_date", default=None
)

# 同一次 Agent 分析内多个 tool 反复枚举 `_candidate_codes` + 4-candidate rank
# 的结果缓存。作用域与 `_agent_frozen_target_date` 对齐，由同一个 try/finally 负责
# set 空 dict / reset，跨 pipeline 运行通过 ContextVar 天然隔离，异常退出也能
# 自动释放。
_CandidatePickCacheType = Dict[Tuple[str, date], Tuple[List[object], str, str]]
_candidate_pick_cache: contextvars.ContextVar[Optional[_CandidatePickCacheType]] = (
    contextvars.ContextVar("candidate_pick_cache", default=None)
)


def set_agent_frozen_target_date(target_date: Optional[date]) -> contextvars.Token:
    """Set pipeline-frozen target date for the current (agent) context."""
    return _agent_frozen_target_date.set(target_date)


def reset_agent_frozen_target_date(token: contextvars.Token) -> None:
    """Reset the frozen target date ContextVar to its previous state."""
    _agent_frozen_target_date.reset(token)


def get_agent_frozen_target_date() -> Optional[date]:
    """Return the pipeline-frozen target date for the current context, or None."""
    return _agent_frozen_target_date.get()


def set_candidate_pick_cache(
    cache: Optional[_CandidatePickCacheType],
) -> contextvars.Token:
    """Install a per-analysis candidate-pick cache in the current context."""
    return _candidate_pick_cache.set(cache)


def reset_candidate_pick_cache(token: contextvars.Token) -> None:
    """Reset the candidate-pick cache ContextVar to its previous state."""
    _candidate_pick_cache.reset(token)


def get_candidate_pick_cache() -> Optional[_CandidatePickCacheType]:
    """Return the per-analysis candidate-pick cache, or None if not installed."""
    return _candidate_pick_cache.get()


@dataclass
class _FetchAttemptState:
    attempted_days: int = 0
    last_error: Optional[str] = None


_history_fetch_attempts: Dict[Tuple[str, date], _FetchAttemptState] = {}
_history_fetch_attempts_lock = RLock()
_history_fetch_locks: Dict[Tuple[str, date], Lock] = {}
_history_fetch_locks_guard = RLock()


def _normalize_days(days: int) -> int:
    try:
        value = int(days)
    except (TypeError, ValueError):
        value = 1
    return max(1, value)


def _resolve_target_date(target_date: Optional[date]) -> date:
    return target_date or date.today()


def _normalize_cache_code(stock_code: str) -> str:
    raw_code = (stock_code or "").strip()
    if not raw_code:
        return ""
    return canonical_stock_code(normalize_stock_code(raw_code))


def _candidate_codes(stock_code: str) -> List[str]:
    raw_code = (stock_code or "").strip()
    normalized_code = normalize_stock_code(raw_code) if raw_code else ""
    canonical_normalized_code = canonical_stock_code(normalized_code)
    canonical_raw_code = canonical_stock_code(raw_code)
    candidates: List[str] = []
    for candidate in (
        canonical_normalized_code,
        canonical_raw_code,
        normalized_code,
        raw_code,
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _bars_to_dataframe(bars: List[object]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame([bar.to_dict() for bar in bars])
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def _infer_source(bars: List[object]) -> str:
    for bar in reversed(bars):
        source = getattr(bar, "data_source", None)
        if source:
            return str(source)
    return "Database"


def _coerce_to_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _get_latest_bar_date(bars: List[object]) -> Optional[date]:
    if not bars:
        return None
    return _coerce_to_date(getattr(bars[-1], "date", None))


def rank_history_bars(bars: List[object]) -> Tuple[date, int]:
    """Rank history buckets by freshest bar first, then by available depth."""
    latest_bar_date = _get_latest_bar_date(bars) or date.min
    return latest_bar_date, len(bars)


def _resolve_expected_target_date(stock_code: str, target_date: Optional[date] = None) -> date:
    if target_date is not None:
        return target_date

    frozen = get_agent_frozen_target_date()
    if frozen is not None:
        return frozen

    try:
        from src.core.trading_calendar import get_effective_trading_date, get_market_for_stock

        market = get_market_for_stock(_normalize_cache_code(stock_code))
        return get_effective_trading_date(market)
    except Exception as exc:
        logger.debug(
            "resolve_expected_target_date(%s): falling back to today due to: %s",
            stock_code,
            exc,
        )
        return date.today()


def _has_sufficient_history(bars: List[object], requested_days: int, expected_target_date: date) -> bool:
    if len(bars) < requested_days:
        return False
    return _is_history_fresh(bars, expected_target_date)


def _is_history_fresh(bars: List[object], expected_target_date: date) -> bool:
    latest_bar_date = _get_latest_bar_date(bars)
    if latest_bar_date is None:
        return False
    return latest_bar_date >= expected_target_date


def _get_attempt_state(attempt_key: Tuple[str, date]) -> _FetchAttemptState:
    with _history_fetch_attempts_lock:
        state = _history_fetch_attempts.get(attempt_key)
        if state is None:
            return _FetchAttemptState()
        return _FetchAttemptState(
            attempted_days=int(state.attempted_days),
            last_error=state.last_error,
        )


def _record_attempt(
    attempt_key: Tuple[str, date],
    *,
    attempted_days: int,
    last_error: Optional[str] = None,
) -> None:
    with _history_fetch_attempts_lock:
        previous = _history_fetch_attempts.get(attempt_key, _FetchAttemptState())
        _history_fetch_attempts[attempt_key] = _FetchAttemptState(
            attempted_days=max(int(previous.attempted_days), int(attempted_days)),
            last_error=last_error,
        )


def _get_attempt_lock(attempt_key: Tuple[str, date]) -> Lock:
    with _history_fetch_locks_guard:
        attempt_lock = _history_fetch_locks.get(attempt_key)
        if attempt_lock is None:
            attempt_lock = Lock()
            _history_fetch_locks[attempt_key] = attempt_lock
        return attempt_lock


def _load_recent_bars_from_db(
    stock_code: str,
    days: int,
    target_date: Optional[date] = None,
) -> Tuple[List[object], str, str]:
    db = get_db()
    requested_days = _normalize_days(days)
    resolved_target = _resolve_target_date(target_date)

    best_bars: List[object] = []
    best_code = _normalize_cache_code(stock_code)

    for code in _candidate_codes(stock_code):
        try:
            if target_date is None:
                bars = list(reversed(db.get_latest_data(code, days=requested_days)))
            else:
                start_date = resolved_target - timedelta(
                    days=max(requested_days * 3, AGENT_HISTORY_BASELINE_DAYS + 30)
                )
                bars = db.get_data_range(code, start_date, resolved_target)
                if len(bars) > requested_days:
                    bars = bars[-requested_days:]
        except Exception as exc:
            logger.debug("load_recent_bars_from_db(%s): DB lookup failed for %s: %s", stock_code, code, exc)
            continue

        if rank_history_bars(bars) > rank_history_bars(best_bars):
            best_bars = list(bars)
            best_code = code

    return best_bars, _infer_source(best_bars), best_code


def resolve_history_storage_code(
    stock_code: str,
    *,
    days: int = 2,
    target_date: Optional[date] = None,
) -> str:
    """Resolve the best storage code bucket for DB-backed historical reads."""
    _bars, _source, storage_code = _load_recent_bars_from_db(
        stock_code,
        days=days,
        target_date=target_date,
    )
    return storage_code or _normalize_cache_code(stock_code) or (stock_code or "").strip()


def get_shared_fetcher_manager():
    """Return a process-wide DataFetcherManager for agent history access."""
    from data_provider import DataFetcherManager

    global _shared_fetcher_manager
    if _shared_fetcher_manager is None:
        with _shared_fetcher_manager_lock:
            if _shared_fetcher_manager is None:
                _shared_fetcher_manager = DataFetcherManager()
    return _shared_fetcher_manager


def reset_shared_history_runtime() -> None:
    """Clear shared history runtime state after config reloads."""
    global _shared_fetcher_manager

    with _shared_fetcher_manager_lock:
        current_manager = _shared_fetcher_manager
        _shared_fetcher_manager = None

    if current_manager is not None and hasattr(current_manager, "close"):
        try:
            current_manager.close()
        except Exception as exc:
            logger.debug("reset_shared_history_runtime(): closing manager failed: %s", exc)

    with _history_fetch_attempts_lock:
        _history_fetch_attempts.clear()
    with _history_fetch_locks_guard:
        _history_fetch_locks.clear()


def ensure_min_history_cached(
    stock_code: str,
    days: int = AGENT_HISTORY_BASELINE_DAYS,
    *,
    target_date: Optional[date] = None,
    fetcher_manager=None,
    force_refresh: bool = False,
) -> Tuple[bool, str]:
    """Ensure the requested history depth is cached in ``stock_daily``."""
    requested_days = _normalize_days(days)
    fetch_days = max(requested_days, AGENT_HISTORY_BASELINE_DAYS)
    canonical_code = _normalize_cache_code(stock_code)
    expected_target_date = _resolve_expected_target_date(stock_code, target_date)
    attempt_key = (canonical_code, expected_target_date)

    existing_bars, existing_source, storage_code = _load_recent_bars_from_db(
        stock_code,
        requested_days,
        target_date=target_date,
    )
    attempt_state = _get_attempt_state(attempt_key)
    if not force_refresh:
        if _has_sufficient_history(existing_bars, requested_days, expected_target_date):
            return True, existing_source
        if attempt_state.attempted_days >= requested_days and attempt_state.last_error:
            return False, attempt_state.last_error

    attempt_lock = _get_attempt_lock(attempt_key)
    with attempt_lock:
        existing_bars, existing_source, storage_code = _load_recent_bars_from_db(
            stock_code,
            requested_days,
            target_date=target_date,
        )
        attempt_state = _get_attempt_state(attempt_key)
        if not force_refresh:
            if _has_sufficient_history(existing_bars, requested_days, expected_target_date):
                return True, existing_source
            if attempt_state.attempted_days >= requested_days and attempt_state.last_error:
                return False, attempt_state.last_error

        manager = fetcher_manager if fetcher_manager is not None else get_shared_fetcher_manager()
        fetch_code = canonical_code or (stock_code or "").strip()

        try:
            df, source_name = manager.get_daily_data(fetch_code, days=fetch_days)
        except Exception as exc:
            error_message = str(exc)
            _record_attempt(
                attempt_key,
                attempted_days=fetch_days,
                last_error=error_message,
            )
            logger.warning(
                "ensure_min_history_cached(%s): fetch failed for %s days: %s",
                stock_code,
                fetch_days,
                exc,
            )
            return False, error_message

        if df is None or df.empty:
            error_message = f"No historical data available for {stock_code}"
            _record_attempt(
                attempt_key,
                attempted_days=fetch_days,
                last_error=error_message,
            )
            return False, error_message

        save_code = canonical_code or storage_code or fetch_code
        try:
            get_db().save_daily_data(df, save_code, source_name)
        except Exception as exc:
            error_message = str(exc)
            _record_attempt(
                attempt_key,
                attempted_days=fetch_days,
                last_error=error_message,
            )
            logger.warning(
                "ensure_min_history_cached(%s): save failed for %s days: %s",
                stock_code,
                fetch_days,
                exc,
            )
            return False, error_message

        persisted_bars, persisted_source, _persisted_storage_code = _load_recent_bars_from_db(
            stock_code,
            requested_days,
            target_date=target_date,
        )
        persisted_days = len(persisted_bars)
        if _has_sufficient_history(persisted_bars, requested_days, expected_target_date):
            _record_attempt(
                attempt_key,
                attempted_days=fetch_days,
                last_error=None,
            )
            logger.info(
                "ensure_min_history_cached(%s): cached %d rows via %s (requested=%d, storage_code=%s)",
                stock_code,
                len(df),
                source_name,
                requested_days,
                save_code,
            )
            return True, persisted_source or source_name

        latest_bar_date = _get_latest_bar_date(persisted_bars)
        if latest_bar_date is None:
            error_message = f"No historical data available for {stock_code}"
        elif not _is_history_fresh(persisted_bars, expected_target_date):
            error_message = (
                f"Stale historical data for {stock_code} "
                f"(latest={latest_bar_date}, expected>={expected_target_date})"
            )
        else:
            error_message = (
                f"Insufficient historical data for {stock_code} "
                f"(got={persisted_days}, need>={requested_days})"
            )

        _record_attempt(
            attempt_key,
            attempted_days=fetch_days,
            last_error=None,
        )
        logger.warning(
            "ensure_min_history_cached(%s): cached history is not ready after refresh "
            "(got=%d, latest=%s, expected>=%s, requested=%d, source=%s, storage_code=%s)",
            stock_code,
            persisted_days,
            latest_bar_date,
            expected_target_date,
            requested_days,
            source_name,
            save_code,
        )
        return False, error_message


def load_recent_history_df(
    stock_code: str,
    days: int,
    *,
    target_date: Optional[date] = None,
    fetcher_manager=None,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Load recent history from DB, backfilling from the shared fetcher when needed."""
    requested_days = _normalize_days(days)
    expected_target_date = _resolve_expected_target_date(stock_code, target_date)
    cached, source_name = ensure_min_history_cached(
        stock_code,
        requested_days,
        target_date=target_date,
        fetcher_manager=fetcher_manager,
        force_refresh=force_refresh,
    )

    bars, db_source, _storage_code = _load_recent_bars_from_db(
        stock_code,
        requested_days,
        target_date=target_date,
    )
    if not bars:
        return pd.DataFrame(), db_source if cached else source_name

    if not _is_history_fresh(bars, expected_target_date):
        latest_bar_date = _get_latest_bar_date(bars)
        stale_message = (
            source_name
            if not cached
            else (
                f"Stale historical data for {stock_code} "
                f"(latest={latest_bar_date}, expected>={expected_target_date})"
            )
        )
        logger.warning(
            "load_recent_history_df(%s): discarding stale history ending at %s "
            "(expected>=%s, cached=%s, source=%s)",
            stock_code,
            latest_bar_date,
            expected_target_date,
            cached,
            db_source or source_name,
        )
        return pd.DataFrame(), stale_message

    return _bars_to_dataframe(bars), db_source or source_name
