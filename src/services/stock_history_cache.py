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


def load_recent_bars_from_db(
    stock_code: str,
    days: int,
    target_date: Optional[date] = None,
) -> Tuple[List[object], str, str]:
    """Return the best-ranking recent history bars across all candidate codes.

    This is the single source of truth for "pick the freshest / deepest
    DB-backed bars for ``stock_code``". Both agent-facing readers (via
    :func:`load_recent_history_df`) and pipeline's trend-analysis Step 3
    call into this function so their candidate-picking semantics stay in
    lockstep (4-candidate rank via :func:`rank_history_bars`).

    The same per-analysis candidate-pick cache (installed by
    :func:`set_candidate_pick_cache` during :func:`_analyze_with_agent`) is
    consulted here to avoid re-ranking the same ``(canonical_code, target_date,
    days)`` tuple multiple times within a single agent analysis.
    """
    db = get_db()
    requested_days = _normalize_days(days)
    # 与 `_ensure_min_history_cached_with_bars` 里 save-后失效用的 key 保持
    # 对称：两边都走 `_resolve_expected_target_date`（explicit > ContextVar >
    # trading_calendar fallback），避免未来 target_date=None + Agent
    # ContextVar 场景下缓存 key 和失效 key 错位（见 #1066 follow-up P2）。
    resolved_target = _resolve_expected_target_date(stock_code, target_date)

    # Per-analysis candidate cache lookup. Key on canonical code + resolved
    # target date so different days within the same target date can reuse
    # the "which candidate code won" decision (but the bars list itself still
    # reflects the request's ``requested_days`` trim on the wins).
    canonical_code = _normalize_cache_code(stock_code)
    cache = get_candidate_pick_cache()
    cache_key: Optional[Tuple[str, date]] = None
    if cache is not None:
        cache_key = (canonical_code, resolved_target)
        cached_entry = cache.get(cache_key)
        if cached_entry is not None:
            cached_bars, cached_source, cached_code = cached_entry
            # Honour the current call's requested_days trim on a shared bars
            # snapshot; callers with smaller days do not need a re-rank.
            if len(cached_bars) > requested_days:
                trimmed = list(cached_bars[-requested_days:])
            else:
                trimmed = list(cached_bars)
            return trimmed, cached_source, cached_code

    best_bars: List[object] = []
    best_code = canonical_code

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

    result_source = _infer_source(best_bars)
    if cache is not None and cache_key is not None:
        cache[cache_key] = (list(best_bars), result_source, best_code)
    return best_bars, result_source, best_code


# Legacy alias: keep the private-looking name so callers that imported it
# (tests, older internal references) keep working without churn.
_load_recent_bars_from_db = load_recent_bars_from_db


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


def _ensure_min_history_cached_with_bars(
    stock_code: str,
    days: int = AGENT_HISTORY_BASELINE_DAYS,
    *,
    target_date: Optional[date] = None,
    fetcher_manager=None,
    force_refresh: bool = False,
) -> Tuple[bool, str, List[object]]:
    """Internal variant of :func:`ensure_min_history_cached` that also returns
    the persisted bars list on the hot success path.

    The public :func:`ensure_min_history_cached` keeps its historical
    ``Tuple[bool, str]`` signature (so external callers and tests do not change),
    while :func:`load_recent_history_df` and other internal hot paths consume
    this 3-tuple to save a redundant ``_load_recent_bars_from_db`` round-trip.
    """
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
            return True, existing_source, existing_bars
        if attempt_state.attempted_days >= requested_days and attempt_state.last_error:
            return False, attempt_state.last_error, existing_bars

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
                return True, existing_source, existing_bars
            if attempt_state.attempted_days >= requested_days and attempt_state.last_error:
                return False, attempt_state.last_error, existing_bars

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
            return False, error_message, existing_bars

        if df is None or df.empty:
            error_message = f"No historical data available for {stock_code}"
            _record_attempt(
                attempt_key,
                attempted_days=fetch_days,
                last_error=error_message,
            )
            return False, error_message, existing_bars

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
            return False, error_message, existing_bars

        # Fresh data was just persisted; drop any cached candidate-pick entries
        # for the current target date so the post-save re-read reflects the new
        # bars instead of the pre-save snapshot.
        pick_cache = get_candidate_pick_cache()
        if pick_cache is not None:
            pick_cache.pop((canonical_code, expected_target_date), None)

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
            return True, persisted_source or source_name, persisted_bars

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

        # 与 "fetch 失败" / "save 失败" 分支保持一致的语义：持久化校验失败时
        # `last_error` 必须记录实际错误信息，否则后续同一 attempt_key 的请求会
        # 因 `attempt_state.last_error is None` 而无限次重放补抓（即 #1066 里
        # "同一 attempt 45 次 HTTP" 的表面现象）。
        _record_attempt(
            attempt_key,
            attempted_days=fetch_days,
            last_error=error_message,
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
        return False, error_message, persisted_bars


def ensure_min_history_cached(
    stock_code: str,
    days: int = AGENT_HISTORY_BASELINE_DAYS,
    *,
    target_date: Optional[date] = None,
    fetcher_manager=None,
    force_refresh: bool = False,
) -> Tuple[bool, str]:
    """Ensure the requested history depth is cached in ``stock_daily``.

    Thin wrapper over :func:`_ensure_min_history_cached_with_bars` that drops
    the bars payload so the public ``Tuple[bool, str]`` signature remains
    backward compatible.
    """
    ok, source_or_error, _bars = _ensure_min_history_cached_with_bars(
        stock_code,
        days,
        target_date=target_date,
        fetcher_manager=fetcher_manager,
        force_refresh=force_refresh,
    )
    return ok, source_or_error


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
    cached, source_name, bars = _ensure_min_history_cached_with_bars(
        stock_code,
        requested_days,
        target_date=target_date,
        fetcher_manager=fetcher_manager,
        force_refresh=force_refresh,
    )
    db_source = _infer_source(bars) if bars else source_name

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
