"""
I/O utilities for reading and writing data files
Supports CSV and Parquet formats with proper timestamp handling
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Union, List, Optional
import logging

logger = logging.getLogger(__name__)


def read_csv(
    file_path: Union[str, Path],
    parse_dates: Optional[List[str]] = None,
    dtype: Optional[dict] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read CSV file with proper timestamp parsing

    Args:
        file_path: Path to CSV file
        parse_dates: List of columns to parse as dates
        dtype: Dictionary of column dtypes
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame with parsed data
    """
    try:
        df = pd.read_csv(
            file_path,
            parse_dates=parse_dates,
            dtype=dtype,
            **kwargs
        )
        logger.info(f"Successfully read CSV: {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV {file_path}: {str(e)}")
        raise


def write_csv(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    index: bool = False,
    **kwargs
) -> None:
    """
    Write DataFrame to CSV file

    Args:
        df: DataFrame to write
        file_path: Output path
        index: Whether to write index
        **kwargs: Additional arguments for pd.to_csv
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=index, **kwargs)
        logger.info(f"Successfully wrote CSV: {file_path}, shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error writing CSV {file_path}: {str(e)}")
        raise


def read_parquet(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    filters: Optional[List] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read Parquet file with optional column selection and filtering

    Args:
        file_path: Path to Parquet file
        columns: List of columns to read
        filters: Filters to apply (PyArrow format)
        **kwargs: Additional arguments for pd.read_parquet

    Returns:
        DataFrame with parsed data
    """
    try:
        df = pd.read_parquet(
            file_path,
            columns=columns,
            filters=filters,
            **kwargs
        )
        logger.info(f"Successfully read Parquet: {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading Parquet {file_path}: {str(e)}")
        raise


def write_parquet(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    compression: str = "snappy",
    **kwargs
) -> None:
    """
    Write DataFrame to Parquet file

    Args:
        df: DataFrame to write
        file_path: Output path
        compression: Compression algorithm
        **kwargs: Additional arguments for pd.to_parquet
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path, compression=compression, **kwargs)
        logger.info(f"Successfully wrote Parquet: {file_path}, shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error writing Parquet {file_path}: {str(e)}")
        raise


def read_orderbook_snapshot(
    date: str,
    instrument_id: str,
    root_path: Union[str, Path],
    n_levels: int = 10
) -> pd.DataFrame:
    """
    Read order book snapshot for specific date and instrument

    Args:
        date: Date in YYYY-MM-DD format
        instrument_id: Instrument identifier
        root_path: Root path to orderbook_snapshots directory
        n_levels: Number of price levels

    Returns:
        DataFrame with order book snapshots
    """
    file_path = Path(root_path) / f"date={date}" / f"instrument_id={instrument_id}.csv"

    # Define columns to read
    columns = ["instrument_id", "venue", "ts_event", "book_seq"]
    for i in range(1, n_levels + 1):
        columns.extend([
            f"bid_px_{i}", f"bid_sz_{i}",
            f"ask_px_{i}", f"ask_sz_{i}"
        ])

    try:
        df = read_csv(
            file_path,
            parse_dates=["ts_event"],
            dtype={"instrument_id": str, "venue": str}
        )
        return df
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return pd.DataFrame()


def read_trades(
    date: str,
    instrument_id: str,
    root_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Read trades for specific date and instrument

    Args:
        date: Date in YYYY-MM-DD format
        instrument_id: Instrument identifier
        root_path: Root path to trades directory

    Returns:
        DataFrame with trades
    """
    file_path = Path(root_path) / f"date={date}" / f"instrument_id={instrument_id}.csv"

    try:
        df = read_csv(
            file_path,
            parse_dates=["ts_event"],
            dtype={
                "instrument_id": str,
                "venue": str,
                "trade_id": str,
                "aggressor_side": str
            }
        )
        return df
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return pd.DataFrame()


def read_instrument_info(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read instrument information

    Args:
        file_path: Path to instrument_info.csv

    Returns:
        DataFrame with instrument metadata
    """
    df = read_csv(
        file_path,
        dtype={
            "instrument_id": str,
            "symbol": str,
            "venue_primary": str,
            "asset_class": str,
            "currency": str
        }
    )
    return df


def read_trading_calendar(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read trading calendar

    Args:
        file_path: Path to trading_calendar.csv

    Returns:
        DataFrame with trading calendar
    """
    df = read_csv(
        file_path,
        parse_dates=[
            "date", "session_open_utc", "session_close_utc",
            "auction_open_utc", "auction_close_utc"
        ],
        dtype={"venue": str}
    )
    return df


def list_available_dates(root_path: Union[str, Path]) -> List[str]:
    """
    List all available dates in data directory

    Args:
        root_path: Root path to data directory

    Returns:
        List of date strings
    """
    dates = []
    for path in Path(root_path).glob("date=*"):
        if path.is_dir():
            date = path.name.replace("date=", "")
            dates.append(date)
    return sorted(dates)


def list_available_instruments(root_path: Union[str, Path], date: str) -> List[str]:
    """
    List all available instruments for a specific date

    Args:
        root_path: Root path to data directory
        date: Date in YYYY-MM-DD format

    Returns:
        List of instrument IDs
    """
    instruments = []
    date_path = Path(root_path) / f"date={date}"
    if date_path.exists():
        for file_path in date_path.glob("instrument_id=*.csv"):
            instrument_id = file_path.stem.replace("instrument_id=", "")
            instruments.append(instrument_id)
    return sorted(instruments)


def save_features(
    df: pd.DataFrame,
    feature_name: str,
    date: str,
    instrument_id: str,
    output_path: Union[str, Path]
) -> None:
    """
    Save extracted features to disk

    Args:
        df: DataFrame with features
        feature_name: Name of feature set
        date: Date string
        instrument_id: Instrument identifier
        output_path: Root output path
    """
    file_path = Path(output_path) / feature_name / f"date={date}" / f"{instrument_id}.parquet"
    write_parquet(df, file_path)


def load_features(
    feature_name: str,
    date: str,
    instrument_id: str,
    input_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Load extracted features from disk

    Args:
        feature_name: Name of feature set
        date: Date string
        instrument_id: Instrument identifier
        input_path: Root input path

    Returns:
        DataFrame with features
    """
    file_path = Path(input_path) / feature_name / f"date={date}" / f"{instrument_id}.parquet"
    return read_parquet(file_path)
