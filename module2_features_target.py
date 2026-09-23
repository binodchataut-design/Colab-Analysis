"""
Module 2: Feature Engineering & Target Labelling
==================================================
Input : data/processed/{SYMBOL}.parquet   (Module 1 output: cleaned, adjusted OHLCV)
Output: data/processed/features/{SYMBOL}.parquet   (feature matrix + target label Y)
        data/processed/_feature_report.csv          (per-symbol label balance / coverage)

Design notes:
- Every feature at row t uses ONLY data available up to and including day t
  (no lookahead). The target label, by contrast, deliberately looks FORWARD
  30-35 trading days — that's the whole point of it being a target.
- The last ~35 rows of each symbol get a NaN target (their forward window runs
  past the end of available history) and are dropped before saving, since they
  can't be used for supervised training — but they'll matter later for live
  inference (Module 5 / production use), not training.
- Sector relative performance needs a sector index proxy. This script computes
  a NEPSE-wide equal-weight return as a stand-in for "the market" since we
  don't have official sub-indices wired up yet — flagged clearly below so this
  can be swapped for real sector indices later without changing anything else.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
FEATURE_DIR = PROCESSED_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# --- target definition (from the pipeline spec) -----------------------------
HORIZON_DAYS = 35          # forward window: ~1.5 months of trading days
PROFIT_TARGET = 0.10       # +10% triggers a positive label...
STOP_LOSS = -0.05          # ...unless -5% is hit first


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral when undefined (e.g. no losses yet)


def compute_atr(high, low, close, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_macd_hist(close, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def compute_bollinger_width(close, period=20, n_std=2) -> pd.Series:
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper, lower = mid + n_std * std, mid - n_std * std
    return (upper - lower) / mid.replace(0, np.nan)


def compute_obv(close, volume) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def build_features(df: pd.DataFrame, market_return: pd.Series) -> pd.DataFrame:
    df = df.sort_values("published_date").reset_index(drop=True)
    close, high, low, vol = df["close"], df["high"], df["low"], df["traded_quantity"]

    feat = pd.DataFrame(index=df.index)
    feat["published_date"] = df["published_date"]

    # --- volatility ---
    atr14 = compute_atr(high, low, close)
    feat["atr_14"] = atr14
    feat["atr_pct_of_price"] = atr14 / close.replace(0, np.nan)
    feat["bb_width_20"] = compute_bollinger_width(close)

    # --- momentum & trend ---
    feat["rsi_14"] = compute_rsi(close)
    feat["macd_hist"] = compute_macd_hist(close)
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    feat["dist_from_sma50"] = (close - sma50) / sma50.replace(0, np.nan)
    feat["dist_from_sma200"] = (close - sma200) / sma200.replace(0, np.nan)

    # --- volume ---
    obv = compute_obv(close, vol)
    feat["obv"] = obv
    feat["obv_20d_slope"] = obv.diff(20)
    vol_sma20 = vol.rolling(20, min_periods=20).mean()
    feat["volume_ratio_20d"] = vol / vol_sma20.replace(0, np.nan)

    # --- NEPSE-specifics: relative performance vs market proxy ---
    # NOTE: market_return is an equal-weight proxy across all kept symbols,
    # standing in for a real sector index until one is wired up (see module docstring).
    stock_return_20d = close.pct_change(20)
    feat["rel_perf_vs_market_20d"] = stock_return_20d - market_return.reindex(df["published_date"]).values

    return feat


def build_target(df: pd.DataFrame) -> pd.Series:
    """Y=1 if the +PROFIT_TARGET threshold is hit within HORIZON_DAYS
    *before* the -STOP_LOSS threshold is hit; Y=0 otherwise (including the
    'nothing happens' and 'stop-loss hits first' cases). NaN where the forward
    window runs past the end of history (can't be labeled yet)."""
    close = df["close"].values
    n = len(close)
    y = np.full(n, np.nan)

    for i in range(n):
        end = i + HORIZON_DAYS
        if end >= n:
            continue  # not enough forward history yet
        entry = close[i]
        if not np.isfinite(entry) or entry <= 0:
            continue  # bad/zero entry price, can't compute a relative return
        window = close[i + 1: end + 1]
        rel_change = (window - entry) / entry

        hit_profit = np.where(rel_change >= PROFIT_TARGET)[0]
        hit_stop = np.where(rel_change <= STOP_LOSS)[0]

        first_profit = hit_profit[0] if len(hit_profit) else np.inf
        first_stop = hit_stop[0] if len(hit_stop) else np.inf

        if first_profit == np.inf and first_stop == np.inf:
            y[i] = 0  # neither threshold touched in window -> no trade signal
        else:
            y[i] = 1.0 if first_profit < first_stop else 0.0

    return pd.Series(y, index=df.index)


def compute_market_proxy(all_symbol_dfs: dict) -> pd.Series:
    """Equal-weight daily return across all kept symbols, indexed by date."""
    frames = []
    for sym, df in all_symbol_dfs.items():
        s = df.set_index("published_date")["close"].pct_change()
        frames.append(s.rename(sym))
    combined = pd.concat(frames, axis=1)
    return combined.mean(axis=1)  # equal-weight average daily return


def run():
    symbol_files = sorted(PROCESSED_DIR.glob("*.parquet"))
    all_dfs = {p.stem: pd.read_parquet(p) for p in symbol_files}
    print(f"Loaded {len(all_dfs)} cleaned symbols from Module 1")

    market_return = compute_market_proxy(all_dfs)

    report_rows = []
    for sym, df in all_dfs.items():
        feat = build_features(df, market_return)
        feat["target"] = build_target(df).values

        # drop rows with no target (tail of history) or NaN in any feature
        # from the warm-up period (e.g. first 200 rows before SMA200 exists)
        usable = feat.dropna(subset=["target"]).copy()
        feature_cols = [c for c in usable.columns if c not in ("published_date", "target")]
        fully_warmed = usable.dropna(subset=feature_cols)

        if len(fully_warmed) > 0:
            fully_warmed.to_parquet(FEATURE_DIR / f"{sym}.parquet", index=False)

        n_total = len(usable)
        n_warmed = len(fully_warmed)
        pos_rate = fully_warmed["target"].mean() if n_warmed else np.nan
        report_rows.append({
            "symbol": sym, "n_labeled_rows": n_total, "n_usable_rows": n_warmed,
            "positive_rate": round(pos_rate, 3) if pd.notna(pos_rate) else np.nan,
        })

    report = pd.DataFrame(report_rows)
    report.to_csv(PROCESSED_DIR / "_feature_report.csv", index=False)

    total_rows = report["n_usable_rows"].sum()
    overall_pos_rate = None
    all_feat_files = list(FEATURE_DIR.glob("*.parquet"))
    if all_feat_files:
        combined = pd.concat([pd.read_parquet(p) for p in all_feat_files], ignore_index=True)
        overall_pos_rate = combined["target"].mean()

    print(f"Symbols with usable feature rows: {(report['n_usable_rows'] > 0).sum()} / {len(report)}")
    print(f"Total usable labeled rows (pooled, non-independent): {total_rows}")
    if overall_pos_rate is not None:
        print(f"Overall positive-label rate (Y=1): {overall_pos_rate:.1%}")
    return report


if __name__ == "__main__":
    run()
