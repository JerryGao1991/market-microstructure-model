"""
Data Preprocessing Module
Cleans, validates, and resamples raw L2 order book and trade data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, List
from datetime import datetime

from config import *
from utils.io_utils import (
    read_orderbook_snapshot, read_trades, read_instrument_info,
    read_trading_calendar, write_parquet
)

# logging.basicConfig(**LOGGING_CONFIG)
configure_logging()
logger = logging.getLogger(__name__)


class OrderBookValidator:
    """Validates order book data according to schema requirements"""

    def __init__(self, instrument_info: pd.DataFrame):
        self.instrument_info = instrument_info.set_index("instrument_id")

    def validate_snapshot(
        self,
        df: pd.DataFrame,
        instrument_id: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate order book snapshot

        Args:
            df: Order book DataFrame
            instrument_id: Instrument identifier

        Returns:
            Tuple of (cleaned_df, error_messages)
        """
        errors = []
        df_clean = df.copy()

        if instrument_id not in self.instrument_info.index:
            errors.append(f"Instrument {instrument_id} not in instrument_info")
            return df_clean, errors

        info = self.instrument_info.loc[instrument_id]
        tick_size = info["tick_size"]
        lot_size = info["lot_size"]

        # Check timestamps are monotonic
        if not df_clean["ts_event"].is_monotonic_increasing:
            errors.append("Timestamps not monotonic")
            df_clean = df_clean.sort_values("ts_event")

        # Validate bid/ask relationship
        invalid_spread = df_clean["ask_px_1"] < df_clean["bid_px_1"]
        if invalid_spread.any():
            errors.append(f"Found {invalid_spread.sum()} crossed quotes")
            df_clean = df_clean[~invalid_spread]

        # Check spread is reasonable
        df_clean["spread_bps"] = (
            (df_clean["ask_px_1"] - df_clean["bid_px_1"]) /
            ((df_clean["ask_px_1"] + df_clean["bid_px_1"]) / 2) * 10000
        )
        invalid_spread = df_clean["spread_bps"] > MAX_SPREAD_BPS
        if invalid_spread.any():
            errors.append(f"Found {invalid_spread.sum()} excessive spreads")
            df_clean = df_clean[~invalid_spread]

        # Check minimum sizes
        for i in range(1, N_LEVELS + 1):
            invalid_bid = df_clean[f"bid_sz_{i}"] < 0
            invalid_ask = df_clean[f"ask_sz_{i}"] < 0
            if invalid_bid.any() or invalid_ask.any():
                errors.append(f"Found negative sizes at level {i}")
                df_clean = df_clean[~(invalid_bid | invalid_ask)]

        # Check price ordering (bid non-increasing, ask non-decreasing)
        for i in range(1, N_LEVELS):
            bid_ordering_error = df_clean[f"bid_px_{i}"] < df_clean[f"bid_px_{i+1}"]
            ask_ordering_error = df_clean[f"ask_px_{i}"] > df_clean[f"ask_px_{i+1}"]

            if bid_ordering_error.any():
                errors.append(f"Bid price ordering error at level {i}")
                df_clean = df_clean[~bid_ordering_error]

            if ask_ordering_error.any():
                errors.append(f"Ask price ordering error at level {i}")
                df_clean = df_clean[~ask_ordering_error]

        # Drop spread_bps temporary column
        df_clean = df_clean.drop(columns=["spread_bps"])

        logger.info(
            f"Validation complete for {instrument_id}: "
            f"{len(df)} -> {len(df_clean)} rows, {len(errors)} error types"
        )

        return df_clean, errors


class DataCleaner:
    """Cleans and preprocesses order book data"""

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        price_col: str = "mid_px",
        max_change_pct: float = MAX_PRICE_CHANGE_PCT
    ) -> pd.DataFrame:
        """
        Remove price outliers based on maximum change threshold

        Args:
            df: DataFrame with price data
            price_col: Column name for price
            max_change_pct: Maximum allowed price change percentage

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        df[price_col] = (df["bid_px_1"] + df["ask_px_1"]) / 2

        price_changes = df[price_col].pct_change().abs() * 100
        outliers = price_changes > max_change_pct

        logger.info(f"Removing {outliers.sum()} price outliers")
        return df[~outliers]

    @staticmethod
    def fill_missing_levels(df: pd.DataFrame, n_levels: int = N_LEVELS) -> pd.DataFrame:
        """
        Fill missing or zero price levels with appropriate values

        Args:
            df: Order book DataFrame
            n_levels: Number of price levels

        Returns:
            DataFrame with filled levels
        """
        df = df.copy()

        for i in range(1, n_levels + 1):
            # Fill zero bid prices with previous level (if available)
            if i > 1:
                df[f"bid_px_{i}"] = df[f"bid_px_{i}"].replace(0, np.nan)
                df[f"bid_px_{i}"] = df[f"bid_px_{i}"].fillna(df[f"bid_px_{i-1}"])

            # Fill zero ask prices with previous level (if available)
            if i > 1:
                df[f"ask_px_{i}"] = df[f"ask_px_{i}"].replace(0, np.nan)
                df[f"ask_px_{i}"] = df[f"ask_px_{i}"].fillna(df[f"ask_px_{i-1}"])

            # Fill zero sizes with 0
            df[f"bid_sz_{i}"] = df[f"bid_sz_{i}"].fillna(0)
            df[f"ask_sz_{i}"] = df[f"ask_sz_{i}"].fillna(0)

        return df

    @staticmethod
    def handle_trading_halts(
        df: pd.DataFrame,
        calendar: pd.DataFrame,
        date: str,
        venue: str
    ) -> pd.DataFrame:
        """
        Filter data to trading sessions only

        Args:
            df: Order book DataFrame
            calendar: Trading calendar
            date: Trading date
            venue: Venue code

        Returns:
            Filtered DataFrame
        """
        session_info = calendar[
            (calendar["venue"] == venue) &
            (calendar["date"] == pd.to_datetime(date))
        ]

        if session_info.empty or not session_info.iloc[0]["is_trading_day"]:
            logger.warning(f"No trading session for {venue} on {date}")
            return pd.DataFrame()

        session = session_info.iloc[0]
        session_start = session["session_open_utc"]
        session_end = session["session_close_utc"]

        # Filter to session times
        mask = (df["ts_event"] >= session_start) & (df["ts_event"] <= session_end)
        df_session = df[mask].copy()

        logger.info(
            f"Filtered to trading session: {len(df)} -> {len(df_session)} rows"
        )

        return df_session


class DataResampler:
    """Resamples order book data to uniform time intervals"""

    @staticmethod
    def resample_orderbook(
        df: pd.DataFrame,
        freq_ms: int = RESAMPLE_FREQ_MS,
        method: str = "last"
    ) -> pd.DataFrame:
        """
        Resample order book to uniform time grid

        Args:
            df: Order book DataFrame with ts_event index
            freq_ms: Resampling frequency in milliseconds
            method: Resampling method ("last", "ffill")

        Returns:
            Resampled DataFrame
        """
        df = df.copy()

        if "ts_event" not in df.columns:
            df = df.reset_index()

        df = df.set_index("ts_event")

        freq_str = f"{freq_ms}ms"

        if method == "last":
            df_resampled = df.resample(freq_str).last()
        elif method == "ffill":
            df_resampled = df.resample(freq_str).ffill()
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        # Forward fill any remaining NaNs
        df_resampled = df_resampled.ffill()

        # Drop rows with any remaining NaNs
        df_resampled = df_resampled.dropna()

        logger.info(
            f"Resampled from {len(df)} to {len(df_resampled)} rows at {freq_ms}ms"
        )

        return df_resampled.reset_index()


def preprocess_orderbook(
    date: str,
    instrument_id: str,
    instrument_info: pd.DataFrame,
    calendar: pd.DataFrame,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Main preprocessing pipeline for order book data

    Args:
        date: Date in YYYY-MM-DD format
        instrument_id: Instrument identifier
        instrument_info: Instrument metadata
        calendar: Trading calendar
        output_path: Optional output path for processed data

    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Preprocessing {instrument_id} for {date}")

    # Read raw data
    df = read_orderbook_snapshot(
        date=date,
        instrument_id=instrument_id,
        root_path=ORDERBOOK_SNAPSHOTS_PATH,
        n_levels=N_LEVELS
    )

    if df.empty:
        logger.warning(f"No data found for {instrument_id} on {date}")
        return df

    # Get venue from instrument info
    venue = instrument_info[
        instrument_info["instrument_id"] == instrument_id
    ]["venue_primary"].iloc[0]

    # Validate
    validator = OrderBookValidator(instrument_info)
    df, errors = validator.validate_snapshot(df, instrument_id)

    if errors:
        logger.warning(f"Validation errors: {errors}")

    # Clean
    cleaner = DataCleaner()
    df = cleaner.remove_outliers(df)
    df = cleaner.fill_missing_levels(df)
    df = cleaner.handle_trading_halts(df, calendar, date, venue)

    if df.empty:
        logger.warning(f"No data remaining after cleaning")
        return df

    # Resample
    resampler = DataResampler()
    df = resampler.resample_orderbook(df, RESAMPLE_FREQ_MS)

    # Add derived fields
    df["mid_px"] = (df["bid_px_1"] + df["ask_px_1"]) / 2
    df["spread_px"] = df["ask_px_1"] - df["bid_px_1"]
    df["spread_bps"] = (df["spread_px"] / df["mid_px"]) * 10000

    # Save if output path provided
    if output_path is not None:
        output_file = output_path / f"date={date}" / f"{instrument_id}.parquet"
        write_parquet(df, output_file)

    logger.info(f"Preprocessing complete: {len(df)} rows")

    return df


def preprocess_trades(
    date: str,
    instrument_id: str,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Preprocess trade data

    Args:
        date: Date in YYYY-MM-DD format
        instrument_id: Instrument identifier
        output_path: Optional output path for processed data

    Returns:
        Preprocessed trade DataFrame
    """
    logger.info(f"Preprocessing trades for {instrument_id} on {date}")

    df = read_trades(
        date=date,
        instrument_id=instrument_id,
        root_path=TRADES_PATH
    )

    if df.empty:
        logger.warning(f"No trade data found")
        return df

    # Sort by timestamp
    df = df.sort_values("ts_event")

    # Remove duplicates
    df = df.drop_duplicates(subset=["trade_id"], keep="first")

    # Filter out invalid trades
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]

    # Save if output path provided
    if output_path is not None:
        output_file = output_path / f"date={date}" / f"{instrument_id}_trades.parquet"
        write_parquet(df, output_file)

    logger.info(f"Trade preprocessing complete: {len(df)} trades")

    return df


def main():
    """Main execution function"""
    logger.info("Starting data preprocessing pipeline")

    # Load metadata
    instrument_info = read_instrument_info(INSTRUMENT_INFO_PATH)
    calendar = read_trading_calendar(TRADING_CALENDAR_PATH)

    logger.info(f"Loaded {len(instrument_info)} instruments")

    # Example: Process single instrument and date
    example_date = "2025-09-15"
    example_instrument = "AAPL.P.XNAS"

    # Check if example files exist
    if (ORDERBOOK_SNAPSHOTS_PATH / f"date={example_date}").exists():
        df_orderbook = preprocess_orderbook(
            date=example_date,
            instrument_id=example_instrument,
            instrument_info=instrument_info,
            calendar=calendar,
            output_path=INTERIM_DATA_PATH
        )

        logger.info(f"Processed order book shape: {df_orderbook.shape}")

        df_trades = preprocess_trades(
            date=example_date,
            instrument_id=example_instrument,
            output_path=INTERIM_DATA_PATH
        )

        logger.info(f"Processed trades shape: {df_trades.shape}")
    else:
        logger.info(
            f"Example data not found at {ORDERBOOK_SNAPSHOTS_PATH / f'date={example_date}'}"
        )
        logger.info("Please place raw data files in data/raw/ directory")

    logger.info("Data preprocessing pipeline complete")


if __name__ == "__main__":
    main()
