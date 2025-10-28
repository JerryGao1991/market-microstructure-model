"""
Evaluation Metrics: P&L, TCA, Capacity, and Performance Analysis
Industry-standard buy-side evaluation metrics
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from typing import Dict, List

from config import *
from utils.metrics_utils import (
    calculate_performance_summary, calculate_tca_metrics,
    calculate_implementation_shortfall, calculate_capacity_metrics
)

# logging.basicConfig(**LOGGING_CONFIG)
configure_logging()
logger = logging.getLogger(__name__)


class PnLCalculator:
    """Calculate net P&L including fees and slippage"""

    def __init__(
        self,
        maker_fee_bps: float = TRANSACTION_COSTS["maker_fee_bps"],
        taker_fee_bps: float = TRANSACTION_COSTS["taker_fee_bps"]
    ):
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps

    def calculate_trade_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str,
        is_maker: bool = True
    ) -> Dict[str, float]:
        """
        Calculate P&L for a single round-trip trade

        Args:
            entry_price: Entry execution price
            exit_price: Exit execution price
            quantity: Trade quantity
            side: Trade side ("long" or "short")
            is_maker: Whether orders were maker (vs taker)

        Returns:
            Dictionary with P&L components
        """
        # Gross P&L
        if side.lower() == "long":
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        # Fees
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        entry_fee = (entry_price * quantity) * (fee_bps / 10000)
        exit_fee = (exit_price * quantity) * (fee_bps / 10000)
        total_fees = entry_fee + exit_fee

        # Net P&L
        net_pnl = gross_pnl - total_fees

        return {
            "gross_pnl": gross_pnl,
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "total_fees": total_fees,
            "net_pnl": net_pnl,
            "pnl_per_share": net_pnl / quantity if quantity > 0 else 0
        }

    def calculate_portfolio_pnl(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative P&L for portfolio

        Args:
            trades_df: DataFrame with trade execution data

        Returns:
            DataFrame with P&L time series
        """
        trades_df = trades_df.copy()

        # Calculate individual trade P&L
        trades_df["gross_pnl"] = 0.0
        trades_df["fees"] = 0.0

        for idx, row in trades_df.iterrows():
            if "entry_price" in row and "exit_price" in row:
                pnl = self.calculate_trade_pnl(
                    row["entry_price"],
                    row["exit_price"],
                    row["quantity"],
                    row.get("side", "long"),
                    row.get("is_maker", True)
                )
                trades_df.loc[idx, "gross_pnl"] = pnl["gross_pnl"]
                trades_df.loc[idx, "fees"] = pnl["total_fees"]

        trades_df["net_pnl"] = trades_df["gross_pnl"] - trades_df["fees"]
        trades_df["cumulative_pnl"] = trades_df["net_pnl"].cumsum()

        return trades_df


class TCAAnalyzer:
    """Transaction Cost Analysis"""

    @staticmethod
    def decompose_costs(
        trades_df: pd.DataFrame,
        benchmark: str = TCA_CONFIG["benchmark"]
    ) -> pd.DataFrame:
        """
        Decompose implementation shortfall into components

        Args:
            trades_df: DataFrame with trade data
            benchmark: Benchmark price type

        Returns:
            DataFrame with TCA decomposition
        """
        tca_df = calculate_tca_metrics(trades_df)

        # Aggregate statistics
        summary = {
            "avg_slippage_bps": tca_df["slippage_bps"].mean(),
            "avg_total_cost_bps": tca_df["total_cost_bps"].mean(),
            "total_cost": tca_df["total_cost"].sum(),
            "benchmark": benchmark
        }

        logger.info(f"TCA Summary: {summary}")

        return tca_df

    @staticmethod
    def calculate_is_components(
        trades_df: pd.DataFrame,
        decision_price: float,
        total_quantity: float
    ) -> Dict[str, float]:
        """
        Calculate implementation shortfall components

        Args:
            trades_df: Executed trades
            decision_price: Decision/benchmark price
            total_quantity: Total order quantity

        Returns:
            Dictionary with IS components
        """
        # Assume all trades are same side
        side = trades_df.iloc[0]["side"]

        is_result = calculate_implementation_shortfall(
            trades_df,
            decision_price,
            side,
            total_quantity
        )

        logger.info(f"Implementation Shortfall: {is_result}")
        return is_result


class CapacityAnalyzer:
    """Strategy capacity analysis"""

    @staticmethod
    def analyze_capacity(
        strategy_results: pd.DataFrame,
        market_volumes: pd.Series,
        test_percentages: List[float] = CAPACITY_CONFIG["adv_percentages"]
    ) -> pd.DataFrame:
        """
        Analyze strategy capacity at different scale levels

        Args:
            strategy_results: DataFrame with strategy P&L and volumes
            market_volumes: Series of market daily volumes
            test_percentages: List of ADV percentages to test

        Returns:
            DataFrame with capacity analysis
        """
        capacity_df = calculate_capacity_metrics(
            market_volumes,
            strategy_results["volume"],
            strategy_results["pnl"]
        )

        logger.info(f"Capacity Analysis:\n{capacity_df.describe()}")
        return capacity_df


def evaluate_strategy(
    trades_df: pd.DataFrame,
    returns_series: pd.Series
) -> Dict:
    """
    Comprehensive strategy evaluation

    Args:
        trades_df: DataFrame with trade execution data
        returns_series: Series of strategy returns

    Returns:
        Dictionary with all evaluation metrics
    """
    logger.info("Evaluating strategy performance")

    # P&L metrics
    pnl_calc = PnLCalculator()
    pnl_df = pnl_calc.calculate_portfolio_pnl(trades_df)

    # Performance summary
    perf_summary = calculate_performance_summary(
        returns_series,
        trades_df,
        risk_free_rate=METRICS_CONFIG["risk_free_rate"]
    )

    # TCA
    tca_analyzer = TCAAnalyzer()
    tca_df = tca_analyzer.decompose_costs(trades_df)

    results = {
        "pnl_summary": {
            "total_gross_pnl": pnl_df["gross_pnl"].sum(),
            "total_fees": pnl_df["fees"].sum(),
            "total_net_pnl": pnl_df["net_pnl"].sum(),
            "num_trades": len(pnl_df)
        },
        "performance": perf_summary,
        "tca": {
            "avg_slippage_bps": tca_df["slippage_bps"].mean(),
            "avg_total_cost_bps": tca_df["total_cost_bps"].mean()
        }
    }

    logger.info(f"Evaluation complete: Net P&L = {results['pnl_summary']['total_net_pnl']:.2f}")

    return results


def main():
    """Main execution function"""
    logger.info("Evaluation metrics example")

    # Generate synthetic trade data
    n_trades = 100
    trades_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n_trades, freq="min"),
        "entry_price": 100 + np.random.randn(n_trades) * 0.5,
        "exit_price": 100 + np.random.randn(n_trades) * 0.5,
        "quantity": np.random.randint(10, 100, n_trades),
        "side": ["long"] * n_trades,
        "is_maker": [True] * n_trades,
        "arrival_price": 100 + np.random.randn(n_trades) * 0.5,
        "execution_price": 100 + np.random.randn(n_trades) * 0.5,
        "fee": np.random.rand(n_trades) * 5
    })

    # Generate returns
    returns = pd.Series(
        np.random.randn(n_trades) * 0.01,
        index=trades_df["timestamp"]
    )

    # Evaluate
    results = evaluate_strategy(trades_df, returns)

    logger.info("\n=== Evaluation Results ===")
    logger.info(f"P&L Summary: {results['pnl_summary']}")
    logger.info(f"Performance: {results['performance']}")
    logger.info(f"TCA: {results['tca']}")

    logger.info("Evaluation metrics complete")


if __name__ == "__main__":
    main()
