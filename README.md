# Market Microstructure Modeling Platform

A systematic research platform for high-frequency trading mechanisms, price impact analysis, and liquidity dynamics. This platform combines L2 order book modeling, deep learning, statistical baselines, and exchange-level simulation to optimize market making and execution strategies.

## Overview

This platform implements a complete pipeline from raw order book data to strategy execution and evaluation:

1. **Data Layer**: L2 order book snapshots, tick-by-tick trades, instrument metadata
2. **Feature Layer**: OFI, Microprice, Depth/Queue Imbalance, Cancel Rate, Quote Age
3. **Model Layer**: DeepLOB/Transformer (deep learning) + Avellaneda-Stoikov/Almgren-Chriss/Hawkes (baselines)
4. **Strategy Layer**: Signal generation, execution, risk controls, TCA evaluation

## Project Structure

```
Market Microstructure Modeling/
├── data/
│   ├── raw/                      # Raw L2 order book & trade data (provided by user)
│   ├── interim/                  # Cleaned & resampled data
│   ├── features/                 # Extracted features
│   ├── models/                   # Trained model weights
│   ├── simulation/               # Exchange replay & ABM logs
│   └── reports/                  # Performance reports
├── src/
│   ├── config.py                 # Global configuration
│   ├── 0_data_preprocessing.py   # Data cleaning & validation
│   ├── 1_feature_engineering.py  # Feature extraction
│   ├── 2_model_deeplob.py        # Deep learning models
│   ├── 3_model_baselines.py      # Baseline models
│   ├── 4_exchange_simulator.py   # Matching engine + ABM
│   ├── 5_strategy_engine.py      # Signal-to-execution pipeline
│   ├── 6_validation.py           # Purged CV & calibration
│   ├── 7_evaluation_metrics.py   # P&L, TCA, capacity
│   └── 8_reporting.py            # Visualization & reports
├── utils/
│   ├── io_utils.py               # Data I/O
│   ├── feature_utils.py          # Feature calculation functions
│   ├── metrics_utils.py          # Performance metrics
│   ├── plotting_utils.py         # Visualization templates
│   └── simulation_utils.py       # Order generation & queue tracking
├── tests/
│   └── test_exchange_simulator.py
├── notebooks/
│   └── (exploratory analysis notebooks)
├── outputs/
│   ├── charts/
│   └── logs/
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-capable GPU for deep learning training

### Setup

1. Clone or download the repository:
```bash
cd "Market Microstructure Modeling"
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create directories:
```bash
python src/config.py
```

## Data Requirements

Place your raw data in `data/raw/` following this structure:

### Order Book Snapshots
```
data/raw/orderbook_snapshots/
└── date=2025-09-15/
    └── instrument_id=AAPL.P.XNAS.csv
```

Required columns: `instrument_id`, `venue`, `ts_event`, `bid_px_1` through `bid_px_10`, `bid_sz_1` through `bid_sz_10`, `ask_px_1` through `ask_px_10`, `ask_sz_1` through `ask_sz_10`

### Trades
```
data/raw/trades/
└── date=2025-09-15/
    └── instrument_id=AAPL.P.XNAS.csv
```

Required columns: `instrument_id`, `venue`, `ts_event`, `trade_id`, `price`, `quantity`, `aggressor_side`

### Metadata
```
data/raw/meta/
├── instrument_info.csv
└── trading_calendar.csv
```

See `data-schema.md` for detailed schema specifications.

## Usage

### Basic Pipeline

Run the modules in sequence:

```bash
# 1. Preprocess raw data
python src/0_data_preprocessing.py

# 2. Extract features
python src/1_feature_engineering.py

# 3. Train deep learning model
python src/2_model_deeplob.py

# 4. Run exchange simulation (optional)
python src/4_exchange_simulator.py

# 5. Backtest strategy
python src/5_strategy_engine.py

# 6. Evaluate performance
python src/7_evaluation_metrics.py

# 7. Generate reports
python src/8_reporting.py
```

### Configuration

Edit `src/config.py` to customize:

- Data paths
- Resampling frequency
- Feature windows (OFI, imbalance, etc.)
- Model hyperparameters
- Risk controls
- Transaction costs

### Running Tests

```bash
cd tests
python -m pytest
```

## Key Features

### Microstructure Features

- **OFI (Order Flow Imbalance)**: Measures net order flow at bid/ask
- **Microprice**: Volume-weighted mid price
- **Queue Imbalance**: Bid/ask volume ratios at multiple levels
- **Cancel Rate**: Order cancellation intensity
- **Quote Age**: Time since last price update

### Models

**Deep Learning:**
- DeepLOB: CNN architecture for LOB sequences
- Transformer: Attention-based temporal modeling

**Baselines:**
- Avellaneda-Stoikov: Optimal market making with inventory management
- Almgren-Chriss: Optimal execution with impact minimization
- Hawkes Process: Self-exciting point process for order arrivals

### Exchange Simulator

- Price-time priority matching
- Queue position tracking
- Agent-based modeling (ABM):
  - Informed traders
  - Noise traders
  - Market makers

### Risk Controls

- Kill-switch (daily loss limits)
- Position limits
- Price collar protection
- Self-trade prevention
- Volume circuit breakers

### Evaluation Metrics

- Net P&L (cash flow basis)
- TCA decomposition (impact cost, opportunity cost, fees)
- Capacity curves (strategy scalability)
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown
- Win rate, profit factor

## Development Workflow

1. Place raw data in `data/raw/`
2. Run preprocessing to validate and clean data
3. Extract features and create labels
4. Train models or use baselines
5. Backtest strategies with realistic simulation
6. Evaluate with buy-side metrics (net P&L, TCA)
7. Generate reports and visualizations

## Important Notes

- **No Lookahead Bias**: All features use strictly past information
- **Purged Cross-Validation**: Prevents temporal leakage in time-series
- **Realistic Costs**: Includes maker/taker fees and slippage
- **Capacity Analysis**: Tests performance at different scale levels (% of ADV)

## References

This platform implements methods from:

- DeepLOB: Zhang et al. (2019) "Deep Convolutional Neural Networks for Limit Order Books"
- Avellaneda-Stoikov: "High-frequency Trading in a Limit Order Book" (2008)
- Almgren-Chriss: "Optimal Execution of Portfolio Transactions" (2000)
- Hawkes Processes: Hawkes (1971) "Spectra of some self-exciting and mutually exciting point processes"

## License

Internal use for market microstructure research.

## Contributing

This is a research platform. For questions or improvements, contact the project maintainer.

## Support

For issues, see:
- Project documentation in `/docs`
- `CLAUDE.md` for development guidance
- `data-schema.md` for data requirements
