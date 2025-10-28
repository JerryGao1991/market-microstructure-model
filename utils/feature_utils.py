"""
Feature engineering utilities for market microstructure analysis
Implements OFI, Microprice, Imbalance, Cancel Rate, and other features
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_microprice(
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    method: str = "weighted"
) -> np.ndarray:
    """
    Calculate microprice using bid-ask weighted average

    Microprice = (bid_px_1 * ask_sz_1 + ask_px_1 * bid_sz_1) / (bid_sz_1 + ask_sz_1)

    Args:
        bid_px: Best bid prices
        bid_sz: Best bid sizes
        ask_px: Best ask prices
        ask_sz: Best ask sizes
        method: Calculation method ("weighted" or "simple")

    Returns:
        Array of microprices
    """
    if method == "weighted":
        total_size = bid_sz + ask_sz
        microprice = np.where(
            total_size > 0,
            (bid_px * ask_sz + ask_px * bid_sz) / total_size,
            (bid_px + ask_px) / 2
        )
    else:  # simple mid-price
        microprice = (bid_px + ask_px) / 2

    return microprice


def calculate_spread(bid_px: np.ndarray, ask_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spread in absolute and basis points

    Args:
        bid_px: Best bid prices
        ask_px: Best ask prices

    Returns:
        Tuple of (spread_absolute, spread_bps)
    """
    spread_abs = ask_px - bid_px
    mid_px = (bid_px + ask_px) / 2
    spread_bps = np.where(mid_px > 0, (spread_abs / mid_px) * 10000, 0)
    return spread_abs, spread_bps


def calculate_queue_imbalance(
    bid_sz: np.ndarray,
    ask_sz: np.ndarray,
    levels: int = 1
) -> np.ndarray:
    """
    Calculate queue imbalance at specified levels

    Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

    Args:
        bid_sz: Bid sizes (can be 2D array for multiple levels)
        ask_sz: Ask sizes (can be 2D array for multiple levels)
        levels: Number of levels to aggregate

    Returns:
        Array of imbalance values
    """
    if bid_sz.ndim == 1:
        bid_volume = bid_sz
        ask_volume = ask_sz
    else:
        bid_volume = np.sum(bid_sz[:, :levels], axis=1)
        ask_volume = np.sum(ask_sz[:, :levels], axis=1)

    total_volume = bid_volume + ask_volume
    imbalance = np.where(
        total_volume > 0,
        (bid_volume - ask_volume) / total_volume,
        0
    )
    return imbalance


def calculate_ofi(
    bid_px: pd.Series,
    bid_sz: pd.Series,
    ask_px: pd.Series,
    ask_sz: pd.Series,
    window: Optional[int] = None
) -> pd.Series:
    """
    Calculate Order Flow Imbalance (OFI)

    OFI = sum(delta_bid_volume) - sum(delta_ask_volume)

    where delta_bid_volume accounts for price changes and size changes at bid

    Args:
        bid_px: Best bid prices
        bid_sz: Best bid sizes
        ask_px: Best ask prices
        ask_sz: Best ask sizes
        window: Rolling window size (None for cumulative)

    Returns:
        Series of OFI values
    """
    # Calculate volume changes
    bid_px_prev = bid_px.shift(1)
    bid_sz_prev = bid_sz.shift(1)
    ask_px_prev = ask_px.shift(1)
    ask_sz_prev = ask_sz.shift(1)

    # Bid side contribution
    bid_contribution = pd.Series(0.0, index=bid_px.index)

    # Case 1: Bid price increases - all volume at new price is added
    bid_contribution += np.where(bid_px > bid_px_prev, bid_sz, 0)

    # Case 2: Bid price unchanged - volume change
    bid_contribution += np.where(
        bid_px == bid_px_prev,
        np.maximum(bid_sz - bid_sz_prev, 0),
        0
    )

    # Case 3: Bid price decreases - volume at old price is removed
    bid_contribution -= np.where(bid_px < bid_px_prev, bid_sz_prev, 0)

    # Ask side contribution (similar logic)
    ask_contribution = pd.Series(0.0, index=ask_px.index)

    ask_contribution += np.where(ask_px < ask_px_prev, ask_sz, 0)
    ask_contribution += np.where(
        ask_px == ask_px_prev,
        np.maximum(ask_sz - ask_sz_prev, 0),
        0
    )
    ask_contribution -= np.where(ask_px > ask_px_prev, ask_sz_prev, 0)

    # OFI is bid contribution minus ask contribution
    ofi = bid_contribution - ask_contribution

    if window is not None:
        ofi = ofi.rolling(window=window).sum()

    return ofi


def calculate_depth_imbalance(
    df: pd.DataFrame,
    n_levels: int = 10
) -> pd.DataFrame:
    """
    Calculate depth imbalance at multiple levels

    Args:
        df: DataFrame with bid_sz_N and ask_sz_N columns
        n_levels: Number of levels

    Returns:
        DataFrame with imbalance features
    """
    result = pd.DataFrame(index=df.index)

    for level in [1, 3, 5, n_levels]:
        if level <= n_levels:
            bid_cols = [f"bid_sz_{i}" for i in range(1, level + 1)]
            ask_cols = [f"ask_sz_{i}" for i in range(1, level + 1)]

            bid_volume = df[bid_cols].sum(axis=1)
            ask_volume = df[ask_cols].sum(axis=1)

            total_volume = bid_volume + ask_volume
            result[f"imbalance_L{level}"] = np.where(
                total_volume > 0,
                (bid_volume - ask_volume) / total_volume,
                0
            )

    return result


def calculate_weighted_mid_price(
    df: pd.DataFrame,
    n_levels: int = 10
) -> pd.Series:
    """
    Calculate volume-weighted mid price across multiple levels

    Args:
        df: DataFrame with bid/ask prices and sizes
        n_levels: Number of levels to consider

    Returns:
        Series of weighted mid prices
    """
    bid_value = 0
    ask_value = 0
    total_volume = 0

    for i in range(1, n_levels + 1):
        bid_vol = df[f"bid_sz_{i}"]
        ask_vol = df[f"ask_sz_{i}"]

        bid_value += df[f"bid_px_{i}"] * bid_vol
        ask_value += df[f"ask_px_{i}"] * ask_vol
        total_volume += bid_vol + ask_vol

    weighted_mid = np.where(
        total_volume > 0,
        (bid_value + ask_value) / total_volume,
        (df["bid_px_1"] + df["ask_px_1"]) / 2
    )

    return pd.Series(weighted_mid, index=df.index)


def calculate_price_levels_occupied(
    df: pd.DataFrame,
    n_levels: int = 10,
    min_size: float = 0.01
) -> Tuple[pd.Series, pd.Series]:
    """
    Count number of price levels with meaningful size

    Args:
        df: DataFrame with bid/ask sizes
        n_levels: Total number of levels
        min_size: Minimum size to count as occupied

    Returns:
        Tuple of (bid_levels_occupied, ask_levels_occupied)
    """
    bid_levels = 0
    ask_levels = 0

    for i in range(1, n_levels + 1):
        bid_levels += (df[f"bid_sz_{i}"] >= min_size).astype(int)
        ask_levels += (df[f"ask_sz_{i}"] >= min_size).astype(int)

    return pd.Series(bid_levels, index=df.index), pd.Series(ask_levels, index=df.index)


def calculate_vwap(prices: pd.Series, volumes: pd.Series, window: int) -> pd.Series:
    """
    Calculate Volume Weighted Average Price

    Args:
        prices: Price series
        volumes: Volume series
        window: Rolling window size

    Returns:
        VWAP series
    """
    pv = (prices * volumes).rolling(window=window).sum()
    v = volumes.rolling(window=window).sum()
    vwap = np.where(v > 0, pv / v, prices)
    return pd.Series(vwap, index=prices.index)


def calculate_trade_flow_imbalance(
    df: pd.DataFrame,
    window: int,
    price_col: str = "price",
    size_col: str = "quantity",
    side_col: str = "aggressor_side"
) -> pd.Series:
    """
    Calculate trade flow imbalance from trade data

    Args:
        df: DataFrame with trade data
        window: Rolling window size
        price_col: Column name for price
        size_col: Column name for size
        side_col: Column name for aggressor side

    Returns:
        Trade flow imbalance series
    """
    buy_volume = np.where(df[side_col] == "B", df[size_col], 0)
    sell_volume = np.where(df[side_col] == "S", df[size_col], 0)

    buy_sum = pd.Series(buy_volume, index=df.index).rolling(window=window).sum()
    sell_sum = pd.Series(sell_volume, index=df.index).rolling(window=window).sum()

    total = buy_sum + sell_sum
    tfi = np.where(total > 0, (buy_sum - sell_sum) / total, 0)

    return pd.Series(tfi, index=df.index)


def calculate_quote_age(ts_event: pd.Series) -> pd.Series:
    """
    Calculate time since last quote update (in milliseconds)

    Args:
        ts_event: Timestamp series

    Returns:
        Quote age in milliseconds
    """
    time_diff = ts_event.diff()
    quote_age = time_diff.dt.total_seconds() * 1000  # Convert to milliseconds
    return quote_age.fillna(0)


def calculate_rolling_volatility(
    prices: pd.Series,
    window: int,
    method: str = "std"
) -> pd.Series:
    """
    Calculate rolling volatility

    Args:
        prices: Price series
        window: Rolling window size
        method: Volatility measure ("std", "parkinson", "garman_klass")

    Returns:
        Volatility series
    """
    if method == "std":
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std()
    else:
        # Placeholder for more sophisticated volatility measures
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std()

    return volatility


def create_lagged_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    lags: List[int]
) -> pd.DataFrame:
    """
    Create lagged versions of features

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        lags: List of lag values

    Returns:
        DataFrame with original and lagged features
    """
    result = df.copy()

    for col in feature_cols:
        for lag in lags:
            result[f"{col}_lag{lag}"] = df[col].shift(lag)

    return result


def normalize_features(
    df: pd.DataFrame,
    method: str = "zscore",
    window: Optional[int] = None
) -> pd.DataFrame:
    """
    Normalize features using specified method

    Args:
        df: DataFrame with features
        method: Normalization method ("zscore", "minmax", "robust")
        window: Rolling window for normalization (None for global)

    Returns:
        Normalized DataFrame
    """
    result = df.copy()

    if method == "zscore":
        if window is not None:
            mean = df.rolling(window=window).mean()
            std = df.rolling(window=window).std()
            result = (df - mean) / std.replace(0, 1)
        else:
            result = (df - df.mean()) / df.std().replace(0, 1)

    elif method == "minmax":
        if window is not None:
            min_val = df.rolling(window=window).min()
            max_val = df.rolling(window=window).max()
            result = (df - min_val) / (max_val - min_val).replace(0, 1)
        else:
            result = (df - df.min()) / (df.max() - df.min()).replace(0, 1)

    return result
