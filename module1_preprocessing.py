"""
Module 1: Data Acquisition & Preprocessing
============================================
Input : data/raw/{SYMBOL}.csv  (published_date, open, high, low, close,
                                 per_change, traded_quantity, traded_amount, status)
Output: data/processed/{SYMBOL}.parquet  (cleaned, adjusted, per-symbol)
        data/processed/_universe_report.csv (one row per symbol: kept/dropped + why)

Design notes:
- The raw source has NO explicit corporate-action flag, so splits/bonus/rights
  are detected heuristically: a session-over-session close jump that is wildly
  inconsistent with the source's own reported `per_change` implies a change in
  share count (not a real price move) and gets back-adjusted, Yahoo-Finance style
  (all historical prices before the event are scaled down by the jump ratio).
- Outlier filtering uses a rolling z-score on log returns, not a global z-score,
  since NEPSE volatility regimes shift a lot over a 10+ year window.
- Illiquidity filter: drop symbols with too few total sessions or too high a
  fraction of zero-volume days to produce a stable feature set later.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- tunable thresholds -----------------------------------------------------
MIN_SESSIONS = 250          # need at least ~1 trading year of history
MAX_ZERO_VOLUME_FRAC = 0.35 # drop symbols traded on <65% of their own sessions
CORP_ACTION_GAP_THRESHOLD = 0.15  # 15% unexplained gap triggers adjustment
ROLL_WINDOW = 60            # rolling window for z-score outlier detection
Z_THRESHOLD = 6.0           # sessions beyond this many rolling std-devs are flagged


def load_raw(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{symbol}.csv")
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    df = df.dropna(subset=["published_date"])
    df = df.sort_values("published_date").drop_duplicates(subset="published_date", keep="last")
    df = df.reset_index(drop=True)
    return df


def detect_and_adjust_corporate_actions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Back-adjust close/open/high/low for large price gaps not explained by
    the source's own per_change field (classic split/bonus/rights signature)."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    implied_change = (df["close"] - prev_close) / prev_close
    reported_change = df["per_change"] / 100.0

    # A corporate action is where the *implied* move from raw close prices
    # diverges sharply from what the exchange *reported* as the day's % change.
    gap = (implied_change - reported_change).abs()
    action_mask = (gap > CORP_ACTION_GAP_THRESHOLD) & prev_close.notna()

    n_actions = int(action_mask.sum())
    if n_actions == 0:
        return df, 0

    # Back-adjust: everything BEFORE each action date gets scaled by the ratio
    # actual_close / (prev_close * (1 + reported_change)), applied cumulatively
    # from most recent action to oldest (standard back-adjustment order).
    adj_factor = np.ones(len(df))
    action_idx = np.where(action_mask.values)[0]
    for idx in sorted(action_idx, reverse=True):
        expected_close = prev_close.iloc[idx] * (1 + reported_change.iloc[idx])
        actual_close = df["close"].iloc[idx]
        if expected_close <= 0 or pd.isna(expected_close):
            continue
        ratio = actual_close / expected_close
        if ratio <= 0 or not np.isfinite(ratio):
            continue
        adj_factor[:idx] *= ratio  # scale everything strictly before this session

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] * adj_factor

    df["corp_action_adjusted"] = action_mask.values
    return df, n_actions


def handle_missing_and_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    if "corp_action_adjusted" not in df.columns:
        df["corp_action_adjusted"] = False

    # log returns for outlier detection (adjusted close by now)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    roll_mean = log_ret.rolling(ROLL_WINDOW, min_periods=20).mean()
    roll_std = log_ret.rolling(ROLL_WINDOW, min_periods=20).std()
    z = (log_ret - roll_mean) / roll_std.replace(0, np.nan)

    # a flagged bad tick is extreme AND not already explained by a detected
    # corporate action (those are legitimate, just already adjusted for)
    outlier_mask = (z.abs() > Z_THRESHOLD) & (~df["corp_action_adjusted"])
    n_outliers = int(outlier_mask.sum())

    df.loc[outlier_mask, ["open", "high", "low", "close"]] = np.nan
    df["is_bad_tick"] = outlier_mask.values

    # forward-fill OHLC for missing trading days (holidays already absent from
    # source; this only fills genuinely missing/nulled prices), volume -> 0
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
    df["traded_quantity"] = df["traded_quantity"].fillna(0)
    df["traded_amount"] = df["traded_amount"].fillna(0)

    return df, n_outliers


def evaluate_liquidity(df: pd.DataFrame) -> dict:
    n_sessions = len(df)
    zero_vol_frac = (df["traded_quantity"] <= 0).mean() if n_sessions else 1.0
    keep = n_sessions >= MIN_SESSIONS and zero_vol_frac <= MAX_ZERO_VOLUME_FRAC
    reason = []
    if n_sessions < MIN_SESSIONS:
        reason.append(f"only {n_sessions} sessions (< {MIN_SESSIONS})")
    if zero_vol_frac > MAX_ZERO_VOLUME_FRAC:
        reason.append(f"{zero_vol_frac:.0%} zero-volume sessions")
    return {"n_sessions": n_sessions, "zero_vol_frac": round(zero_vol_frac, 3),
            "kept": keep, "drop_reason": "; ".join(reason)}


def process_symbol(symbol: str) -> dict:
    df = load_raw(symbol)
    if df.empty:
        return {"symbol": symbol, "kept": False, "drop_reason": "empty/unreadable file",
                "n_sessions": 0, "zero_vol_frac": None, "n_corp_actions": 0, "n_outliers": 0}

    df, n_actions = detect_and_adjust_corporate_actions(df)
    df, n_outliers = handle_missing_and_outliers(df)
    liq = evaluate_liquidity(df)

    if liq["kept"]:
        df.to_parquet(OUT_DIR / f"{symbol}.parquet", index=False)

    return {"symbol": symbol, "n_corp_actions": n_actions, "n_outliers": n_outliers, **liq}


def run(limit: int | None = None):
    symbols = sorted(p.stem for p in RAW_DIR.glob("*.csv"))
    if limit:
        symbols = symbols[:limit]

    report_rows = [process_symbol(sym) for sym in symbols]
    report = pd.DataFrame(report_rows)
    report.to_csv(OUT_DIR / "_universe_report.csv", index=False)

    kept = report["kept"].sum()
    print(f"Processed {len(report)} symbols -> kept {kept}, dropped {len(report) - kept}")
    print(f"Total corporate-action adjustments detected: {report['n_corp_actions'].sum()}")
    print(f"Total bad ticks nulled+filled: {report['n_outliers'].sum()}")
    return report


if __name__ == "__main__":
    run()
