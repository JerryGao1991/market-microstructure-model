"""
Global configuration for Market Microstructure Modeling Platform
Defines paths, constants, and parameters for all modules
"""

import logging
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Raw data paths
RAW_DATA_ROOT = DATA_ROOT / "raw"
ORDERBOOK_SNAPSHOTS_PATH = RAW_DATA_ROOT / "orderbook_snapshots"
TRADES_PATH = RAW_DATA_ROOT / "trades"
META_PATH = RAW_DATA_ROOT / "meta"
INSTRUMENT_INFO_PATH = META_PATH / "instrument_info.csv"
TRADING_CALENDAR_PATH = META_PATH / "trading_calendar.csv"

# Processed data paths
INTERIM_DATA_PATH = DATA_ROOT / "interim"
FEATURES_PATH = DATA_ROOT / "features"
MODELS_PATH = DATA_ROOT / "models"
SIMULATION_PATH = DATA_ROOT / "simulation"
REPORTS_PATH = DATA_ROOT / "reports"

# Output paths
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CHARTS_PATH = OUTPUTS_ROOT / "charts"
LOGS_PATH = OUTPUTS_ROOT / "logs"

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================

# Timestamp precision
TIMESTAMP_UNIT = "ns"  # Options: "ns", "us", "ms"
TIMEZONE = "UTC"

# Order book parameters
N_LEVELS = 10  # Number of price levels (5, 10, or 20)
MIN_LEVELS = 5  # Minimum levels required for valid snapshot

# Resampling parameters
RESAMPLE_FREQ_MS = 100  # Resample frequency in milliseconds
EVENT_TIME_WINDOW = 1000  # Event time window for features (number of events)

# Data cleaning thresholds
MAX_SPREAD_BPS = 500  # Maximum spread in basis points (5%)
MAX_PRICE_CHANGE_PCT = 10.0  # Maximum price change between snapshots (%)
MIN_BID_ASK_SIZE = 1  # Minimum bid/ask size to be valid

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# OFI (Order Flow Imbalance) parameters
OFI_WINDOWS = [10, 50, 100, 500]  # Windows in milliseconds
OFI_DECAY = 0.95  # Exponential decay factor

# Microprice calculation
MICROPRICE_METHOD = "weighted"  # Options: "weighted", "simple"

# Queue imbalance parameters
IMBALANCE_LEVELS = [1, 3, 5, 10]  # Levels to calculate imbalance

# Cancel rate parameters
CANCEL_RATE_WINDOW_MS = 1000  # Window for cancel rate calculation

# Quote age parameters
QUOTE_AGE_THRESHOLD_MS = 500  # Threshold for stale quotes

# Feature lag windows (in milliseconds)
FEATURE_LAGS = [100, 500, 1000, 5000]

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# DeepLOB parameters
DEEPLOB_CONFIG = {
    "n_levels": N_LEVELS,
    "sequence_length": 100,  # Number of snapshots in input sequence
    "hidden_dims": [64, 128, 128],
    "kernel_sizes": [3, 3, 3],
    "dropout": 0.2,
    "n_classes": 3,  # Up, Down, Stationary
}

# Transformer parameters
TRANSFORMER_CONFIG = {
    "d_model": 128,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "sequence_length": 100,
    "n_classes": 3,
}

# Training parameters
TRAINING_CONFIG = {
    "batch_size": 128,
    "learning_rate": 0.001,
    "epochs": 50,
    "early_stopping_patience": 10,
    "validation_split": 0.2,
    "optimizer": "adam",
    "loss": "cross_entropy",
    "device": "cuda",  # "cuda" or "cpu"
}

# Label parameters
PREDICTION_HORIZON_MS = 5  # Prediction horizon (5ms as mentioned)
PRICE_CHANGE_THRESHOLD_BPS = 1  # Threshold for labeling as up/down (1 bp)

# ============================================================================
# BASELINE MODEL PARAMETERS
# ============================================================================

# Avellaneda-Stoikov parameters
AS_CONFIG = {
    "gamma": 0.1,  # Risk aversion parameter
    "k": 1.5,  # Order book liquidity parameter
    "T": 1.0,  # Time horizon (in seconds)
    "max_inventory": 100,  # Maximum inventory position
}

# Almgren-Chriss parameters
AC_CONFIG = {
    "lambda_": 0.1,  # Permanent impact coefficient
    "eta": 0.01,  # Temporary impact coefficient
    "sigma": 0.02,  # Volatility
    "gamma": 0.1,  # Risk aversion
    "T": 60.0,  # Execution horizon (seconds)
}

# Hawkes process parameters
HAWKES_CONFIG = {
    "kernel": "exponential",  # Options: "exponential", "power_law"
    "baseline_intensity": 1.0,
    "decay": 0.5,
    "max_iter": 100,
}

# ============================================================================
# EXCHANGE SIMULATOR PARAMETERS
# ============================================================================

# Matching engine configuration
MATCHING_CONFIG = {
    "priority_rule": "price_time",  # Price-time priority
    "tick_size": 0.01,  # Will be overridden by instrument_info
    "lot_size": 1,  # Will be overridden by instrument_info
    "enable_self_trade_prevention": True,
    "enable_auction": True,
}

# ABM simulation parameters
ABM_CONFIG = {
    "n_informed_traders": 10,
    "n_noise_traders": 50,
    "n_market_makers": 5,
    "informed_trade_intensity": 0.1,
    "noise_trade_intensity": 0.5,
    "mm_update_frequency_ms": 100,
    "simulation_duration_seconds": 3600,
}

# Queue position parameters
QUEUE_CONFIG = {
    "enable_queue_tracking": True,
    "partial_fill_enabled": True,
}

# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================

# Signal thresholds
SIGNAL_CONFIG = {
    "long_threshold": 0.6,  # Probability threshold for long signal
    "short_threshold": 0.4,  # Probability threshold for short signal
    "min_signal_strength": 0.05,  # Minimum signal strength to trade
}

# Position management
POSITION_CONFIG = {
    "max_position_size": 1000,  # Maximum position in shares
    "max_daily_trades": 500,
    "max_order_size": 100,  # Maximum single order size
    "inventory_target": 0,  # Target inventory level
}

# Risk controls
RISK_CONFIG = {
    "enable_kill_switch": True,
    "max_daily_loss": 10000,  # Maximum daily loss in currency units
    "max_drawdown_pct": 5.0,  # Maximum drawdown percentage
    "enable_limit_price_protection": True,
    "price_collar_pct": 1.0,  # Price collar as % of mid
    "max_notional_per_second": 100000,
    "circuit_breaker_volume_multiplier": 5.0,  # Halt if volume > 5x average
}

# Execution parameters
EXECUTION_CONFIG = {
    "order_type": "limit",  # Options: "limit", "market"
    "time_in_force": "GTC",  # Good-til-cancelled
    "min_time_between_orders_ms": 10,
    "max_order_retry": 3,
}

# ============================================================================
# VALIDATION PARAMETERS
# ============================================================================

# Purged K-Fold CV parameters
CV_CONFIG = {
    "n_splits": 5,
    "purge_pct": 0.1,  # Purge 10% of data around splits
    "embargo_pct": 0.05,  # Embargo 5% after test set
    "shuffle": False,  # Time-series data should not be shuffled
}

# Calibration parameters
CALIBRATION_CONFIG = {
    "method": "isotonic",  # Options: "isotonic", "platt"
    "n_bins": 10,  # Number of bins for calibration curve
}

# Walk-forward validation
WALK_FORWARD_CONFIG = {
    "train_period_days": 60,
    "test_period_days": 15,
    "min_train_samples": 10000,
}

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Transaction costs
TRANSACTION_COSTS = {
    "maker_fee_bps": 0.3,  # Maker fee in basis points
    "taker_fee_bps": 0.8,  # Taker fee in basis points
    "slippage_model": "linear",  # Options: "linear", "sqrt", "custom"
}

# TCA parameters
TCA_CONFIG = {
    "benchmark": "arrival_price",  # Options: "arrival_price", "vwap", "twap"
    "include_opportunity_cost": True,
    "adverse_selection_window_ms": 1000,
}

# Capacity analysis
CAPACITY_CONFIG = {
    "adv_percentages": [0.1, 0.5, 1.0, 2.0, 5.0],  # % of ADV to test
    "min_daily_volume": 1000000,  # Minimum daily volume for inclusion
}

# Performance metrics
METRICS_CONFIG = {
    "risk_free_rate": 0.02,  # Annual risk-free rate
    "trading_days_per_year": 252,
    "confidence_level": 0.95,  # For VaR calculation
}

# ============================================================================
# REPORTING PARAMETERS
# ============================================================================

# Visualization settings
VIZ_CONFIG = {
    "figure_dpi": 100,
    "figure_size": (12, 8),
    "style": "seaborn-v0_8-darkgrid",
    "color_palette": "husl",
}

# Report generation
REPORT_CONFIG = {
    "format": "html",  # Options: "html", "pdf", "markdown"
    "include_diagnostics": True,
    "include_calibration_curves": True,
    "include_feature_importance": True,
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_to_file": True,
    "log_file": LOGS_PATH / "platform.log",
    "log_to_console": True,
}

def configure_logging(force: bool = False) -> None:
    """Configure logging handlers based on :data:`LOGGING_CONFIG`."""

    logging_config = LOGGING_CONFIG.copy()
    log_to_file = logging_config.pop("log_to_file", False)
    log_file = logging_config.pop("log_file", None)
    log_to_console = logging_config.pop("log_to_console", True)

    handlers = []

    if log_to_console:
        handlers.append(logging.StreamHandler())

    if log_to_file and log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path))

    basic_config_kwargs = {
        "force": force,
        **logging_config,
    }

    if handlers:
        basic_config_kwargs["handlers"] = handlers

    logging.basicConfig(**basic_config_kwargs)
    
# ============================================================================
# ENSURE DIRECTORIES EXIST
# ============================================================================

def create_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        RAW_DATA_ROOT,
        ORDERBOOK_SNAPSHOTS_PATH,
        TRADES_PATH,
        META_PATH,
        INTERIM_DATA_PATH,
        FEATURES_PATH,
        MODELS_PATH,
        SIMULATION_PATH,
        REPORTS_PATH,
        CHARTS_PATH,
        LOGS_PATH,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root: {DATA_ROOT}")
    print("All directories created successfully")
