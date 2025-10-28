"""
Feature Engineering Module
Extracts microstructure features: OFI, Microprice, Imbalance, Cancel Rate, etc.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

from config import *
from utils.io_utils import read_parquet, write_parquet
from utils.feature_utils import (
    calculate_microprice, calculate_spread, calculate_queue_imbalance,
    calculate_ofi, calculate_depth_imbalance, calculate_weighted_mid_price,
    calculate_price_levels_occupied, calculate_quote_age,
    calculate_rolling_volatility, create_lagged_features, normalize_features
)

# logging.basicConfig(**LOGGING_CONFIG)
configure_logging()
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts microstructure features from order book data"""

    def __init__(self, n_levels: int = N_LEVELS):
        self.n_levels = n_levels

    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic price and spread features

        Args:
            df: Order book DataFrame

        Returns:
            DataFrame with basic features
        """
        logger.info("Extracting basic features")

        result = df.copy()

        # Microprice
        result["microprice"] = calculate_microprice(
            df["bid_px_1"].values,
            df["bid_sz_1"].values,
            df["ask_px_1"].values,
            df["ask_sz_1"].values,
            method=MICROPRICE_METHOD
        )

        # Spread features
        spread_abs, spread_bps = calculate_spread(
            df["bid_px_1"].values,
            df["ask_px_1"].values
        )
        result["spread_abs"] = spread_abs
        result["spread_bps"] = spread_bps

        # Weighted mid price across levels
        result["weighted_mid"] = calculate_weighted_mid_price(df, self.n_levels)

        # Quote age
        result["quote_age_ms"] = calculate_quote_age(df["ts_event"])

        return result

    def extract_imbalance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract queue imbalance features at multiple levels

        Args:
            df: Order book DataFrame

        Returns:
            DataFrame with imbalance features
        """
        logger.info("Extracting imbalance features")

        result = df.copy()

        # Imbalance at level 1
        result["imbalance_L1"] = calculate_queue_imbalance(
            df["bid_sz_1"].values,
            df["ask_sz_1"].values
        )

        # Depth imbalance at multiple levels
        depth_imbalance = calculate_depth_imbalance(df, self.n_levels)
        result = pd.concat([result, depth_imbalance], axis=1)

        # Price levels occupied
        bid_levels, ask_levels = calculate_price_levels_occupied(df, self.n_levels)
        result["bid_levels_occupied"] = bid_levels
        result["ask_levels_occupied"] = ask_levels

        return result

    def extract_ofi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Order Flow Imbalance features at multiple windows

        Args:
            df: Order book DataFrame

        Returns:
            DataFrame with OFI features
        """
        logger.info("Extracting OFI features")

        result = df.copy()

        for window in OFI_WINDOWS:
            ofi = calculate_ofi(
                df["bid_px_1"],
                df["bid_sz_1"],
                df["ask_px_1"],
                df["ask_sz_1"],
                window=window
            )
            result[f"ofi_{window}"] = ofi

        return result

    def extract_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract volatility and price change features

        Args:
            df: Order book DataFrame

        Returns:
            DataFrame with volatility features
        """
        logger.info("Extracting volatility features")

        result = df.copy()

        # Price returns
        result["mid_return"] = df["mid_px"].pct_change()
        result["microprice_return"] = result["microprice"].pct_change()

        # Rolling volatility at multiple windows
        for window in [10, 50, 100]:
            result[f"volatility_{window}"] = calculate_rolling_volatility(
                df["mid_px"], window
            )

        # Price momentum
        for lag in [5, 10, 50]:
            result[f"price_momentum_{lag}"] = (
                df["mid_px"] - df["mid_px"].shift(lag)
            ) / df["mid_px"].shift(lag)

        return result

    def extract_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract volume-related features

        Args:
            df: Order book DataFrame

        Returns:
            DataFrame with volume features
        """
        logger.info("Extracting volume features")

        result = df.copy()

        # Total volume at each side
        bid_volume_cols = [f"bid_sz_{i}" for i in range(1, self.n_levels + 1)]
        ask_volume_cols = [f"ask_sz_{i}" for i in range(1, self.n_levels + 1)]

        result["total_bid_volume"] = df[bid_volume_cols].sum(axis=1)
        result["total_ask_volume"] = df[ask_volume_cols].sum(axis=1)
        result["total_volume"] = result["total_bid_volume"] + result["total_ask_volume"]

        # Volume ratios
        result["bid_ask_volume_ratio"] = np.where(
            result["total_ask_volume"] > 0,
            result["total_bid_volume"] / result["total_ask_volume"],
            0
        )

        # Volume changes
        result["bid_volume_change"] = result["total_bid_volume"].pct_change()
        result["ask_volume_change"] = result["total_ask_volume"].pct_change()

        # Rolling volume statistics
        for window in [10, 50]:
            result[f"avg_volume_{window}"] = result["total_volume"].rolling(window).mean()
            result[f"volume_std_{window}"] = result["total_volume"].rolling(window).std()

        return result

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all microstructure features

        Args:
            df: Order book DataFrame

        Returns:
            DataFrame with all features
        """
        logger.info("Extracting all microstructure features")

        result = df.copy()

        # Extract feature groups
        result = self.extract_basic_features(result)
        result = self.extract_imbalance_features(result)
        result = self.extract_ofi_features(result)
        result = self.extract_volatility_features(result)
        result = self.extract_volume_features(result)

        logger.info(f"Total features extracted: {len(result.columns)}")

        return result


def create_labels(
    df: pd.DataFrame,
    horizon_ms: int = PREDICTION_HORIZON_MS,
    threshold_bps: float = PRICE_CHANGE_THRESHOLD_BPS
) -> pd.DataFrame:
    """
    Create price direction labels for supervised learning

    Args:
        df: DataFrame with price data
        horizon_ms: Prediction horizon in milliseconds
        threshold_bps: Threshold for up/down classification in basis points

    Returns:
        DataFrame with labels
    """
    logger.info(f"Creating labels with {horizon_ms}ms horizon")

    result = df.copy()

    # Calculate future mid price at horizon
    # For time-based: approximate by number of rows
    rows_ahead = max(1, int(horizon_ms / RESAMPLE_FREQ_MS))

    result["future_mid_px"] = result["mid_px"].shift(-rows_ahead)

    # Calculate price change
    result["price_change"] = (
        (result["future_mid_px"] - result["mid_px"]) / result["mid_px"] * 10000
    )

    # Create labels: 0 = Down, 1 = Stationary, 2 = Up
    result["label"] = 1  # Default: Stationary

    result.loc[result["price_change"] > threshold_bps, "label"] = 2  # Up
    result.loc[result["price_change"] < -threshold_bps, "label"] = 0  # Down

    # Drop future price and change (to prevent leakage)
    result = result.drop(columns=["future_mid_px", "price_change"])

    # Remove rows without labels (at the end)
    result = result.dropna(subset=["label"])

    label_dist = result["label"].value_counts()
    logger.info(f"Label distribution:\n{label_dist}")

    return result


def process_features(
    date: str,
    instrument_id: str,
    input_path: Path = INTERIM_DATA_PATH,
    output_path: Path = FEATURES_PATH,
    create_labels_flag: bool = True
) -> pd.DataFrame:
    """
    Main feature engineering pipeline

    Args:
        date: Date in YYYY-MM-DD format
        instrument_id: Instrument identifier
        input_path: Path to preprocessed data
        output_path: Path to save features
        create_labels_flag: Whether to create labels

    Returns:
        DataFrame with features and labels
    """
    logger.info(f"Processing features for {instrument_id} on {date}")

    # Read preprocessed data
    input_file = input_path / f"date={date}" / f"{instrument_id}.parquet"

    try:
        df = read_parquet(input_file)
    except Exception as e:
        logger.error(f"Could not read input file: {e}")
        return pd.DataFrame()

    # Extract features
    extractor = FeatureExtractor(n_levels=N_LEVELS)
    df_features = extractor.extract_all_features(df)

    # Create labels if requested
    if create_labels_flag:
        df_features = create_labels(df_features)

    # Drop rows with NaN values (from rolling calculations)
    df_features = df_features.dropna()

    # Save features
    output_file = output_path / f"date={date}" / f"{instrument_id}.parquet"
    write_parquet(df_features, output_file)

    logger.info(
        f"Feature engineering complete: {len(df_features)} rows, "
        f"{len(df_features.columns)} columns"
    )

    return df_features


def main():
    """Main execution function"""
    logger.info("Starting feature engineering pipeline")

    # Example: Process single instrument and date
    example_date = "2025-09-15"
    example_instrument = "AAPL.P.XNAS"

    # Check if preprocessed data exists
    input_file = INTERIM_DATA_PATH / f"date={example_date}" / f"{example_instrument}.parquet"

    if input_file.exists():
        df_features = process_features(
            date=example_date,
            instrument_id=example_instrument,
            create_labels_flag=True
        )

        logger.info(f"Features shape: {df_features.shape}")
        logger.info(f"Feature columns: {list(df_features.columns[:20])}...")
    else:
        logger.info(f"Preprocessed data not found at {input_file}")
        logger.info("Please run 0_data_preprocessing.py first")

    logger.info("Feature engineering pipeline complete")


if __name__ == "__main__":
    main()
