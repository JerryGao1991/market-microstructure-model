"""
Reporting Module: Generate visualizations and performance reports
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Optional

from config import *
from utils.plotting_utils import (
    plot_equity_curve, plot_drawdown, plot_returns_distribution,
    plot_calibration_curve, plot_capacity_curve, plot_pnl_attribution,
    save_figure
)

# logging.basicConfig(**LOGGING_CONFIG)
configure_logging()
logger = logging.getLogger(__name__)


class PerformanceReporter:
    """Generate performance reports and visualizations"""

    def __init__(self, output_path: Path = CHARTS_PATH):
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate_pnl_report(
        self,
        returns: pd.Series,
        trades_df: pd.DataFrame,
        title: str = "Strategy Performance"
    ):
        """Generate P&L visualizations"""
        logger.info("Generating P&L report")

        # Equity curve
        fig = plot_equity_curve(returns, title=f"{title} - Equity Curve")
        save_figure(fig, self.output_path / "equity_curve.png")

        # Drawdown
        fig = plot_drawdown(returns, title=f"{title} - Drawdown")
        save_figure(fig, self.output_path / "drawdown.png")

        # Returns distribution
        fig = plot_returns_distribution(returns, title=f"{title} - Returns Distribution")
        save_figure(fig, self.output_path / "returns_distribution.png")

        logger.info(f"P&L charts saved to {self.output_path}")

    def generate_calibration_report(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Model Calibration"
    ):
        """Generate calibration visualizations"""
        logger.info("Generating calibration report")

        fig = plot_calibration_curve(
            y_true, y_prob,
            title=title
        )
        save_figure(fig, self.output_path / "calibration_curve.png")

    def generate_capacity_report(
        self,
        capacity_df: pd.DataFrame,
        title: str = "Strategy Capacity"
    ):
        """Generate capacity analysis"""
        logger.info("Generating capacity report")

        fig = plot_capacity_curve(
            capacity_df["pct_of_adv"].values,
            capacity_df["pnl_per_volume"].values,
            title=title
        )
        save_figure(fig, self.output_path / "capacity_curve.png")

    def generate_summary_report(
        self,
        results: Dict,
        output_file: Optional[Path] = None
    ):
        """Generate text summary report"""
        if output_file is None:
            output_file = REPORTS_PATH / "summary_report.md"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            f.write("# Strategy Performance Summary\n\n")

            # P&L Summary
            f.write("## P&L Summary\n\n")
            if "pnl_summary" in results:
                for key, value in results["pnl_summary"].items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")

            # Performance Metrics
            f.write("## Performance Metrics\n\n")
            if "performance" in results:
                for key, value in results["performance"].items():
                    if isinstance(value, float):
                        f.write(f"- **{key}**: {value:.4f}\n")
                    else:
                        f.write(f"- **{key}**: {value}\n")
                f.write("\n")

            # TCA Summary
            f.write("## Transaction Cost Analysis\n\n")
            if "tca" in results:
                for key, value in results["tca"].items():
                    if isinstance(value, float):
                        f.write(f"- **{key}**: {value:.4f} bps\n")
                    else:
                        f.write(f"- **{key}**: {value}\n")

        logger.info(f"Summary report saved to {output_file}")


def main():
    """Main execution function"""
    logger.info("Reporting module example")

    # Generate synthetic data for demonstration
    n_periods = 252
    returns = pd.Series(
        np.random.randn(n_periods) * 0.01,
        index=pd.date_range("2025-01-01", periods=n_periods, freq="D")
    )

    trades_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="D"),
        "pnl": np.random.randn(100) * 100
    })

    # Initialize reporter
    reporter = PerformanceReporter()

    # Generate reports
    reporter.generate_pnl_report(returns, trades_df)

    # Generate calibration report (synthetic data)
    y_true = np.random.randint(0, 2, 1000)
    y_prob = np.random.rand(1000)
    reporter.generate_calibration_report(y_true, y_prob)

    # Generate summary report
    results = {
        "pnl_summary": {
            "total_net_pnl": 12500.0,
            "num_trades": 100,
            "win_rate": 0.58
        },
        "performance": {
            "sharpe_ratio": 1.85,
            "max_drawdown": -8.5,
            "annual_return": 22.3
        },
        "tca": {
            "avg_slippage_bps": 1.2,
            "avg_total_cost_bps": 2.8
        }
    }

    reporter.generate_summary_report(results)

    logger.info("Reporting complete")


if __name__ == "__main__":
    main()
