"""
Plotting utilities for visualization
Provides templates for common market microstructure visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Set default style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_orderbook_snapshot(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    title: str = "Order Book Snapshot",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot order book snapshot with bid/ask ladder

    Args:
        bid_prices: Array of bid prices
        bid_sizes: Array of bid sizes
        ask_prices: Array of ask prices
        ask_sizes: Array of ask sizes
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bids (green bars, left side)
    ax.barh(bid_prices, bid_sizes, height=np.diff(bid_prices, prepend=bid_prices[0] * 0.999),
            color="green", alpha=0.6, label="Bid")

    # Plot asks (red bars, right side)
    ax.barh(ask_prices, -ask_sizes, height=np.diff(ask_prices, prepend=ask_prices[0] * 1.001),
            color="red", alpha=0.6, label="Ask")

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Volume")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_price_and_volume(
    df: pd.DataFrame,
    price_col: str = "mid_px",
    volume_col: Optional[str] = None,
    title: str = "Price and Volume",
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot price time series with optional volume

    Args:
        df: DataFrame with price and volume data
        price_col: Column name for price
        volume_col: Column name for volume (optional)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if volume_col is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    # Plot price
    ax1.plot(df.index, df[price_col], label=price_col, linewidth=1)
    ax1.set_ylabel("Price")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot volume if provided
    if volume_col is not None:
        ax2.bar(df.index, df[volume_col], alpha=0.6, color="blue")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Volume")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spread_analysis(
    df: pd.DataFrame,
    spread_col: str = "spread_bps",
    title: str = "Spread Analysis",
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot spread time series and distribution

    Args:
        df: DataFrame with spread data
        spread_col: Column name for spread
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Time series
    ax1.plot(df.index, df[spread_col], linewidth=1, alpha=0.7)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Spread (bps)")
    ax1.set_title(f"{title} - Time Series")
    ax1.grid(True, alpha=0.3)

    # Distribution
    ax2.hist(df[spread_col].dropna(), bins=50, alpha=0.7, edgecolor="black")
    ax2.axvline(df[spread_col].median(), color="red", linestyle="--",
                label=f"Median: {df[spread_col].median():.2f}")
    ax2.set_xlabel("Spread (bps)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"{title} - Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_imbalance_heatmap(
    df: pd.DataFrame,
    imbalance_cols: List[str],
    title: str = "Queue Imbalance Heatmap",
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot heatmap of imbalance features over time

    Args:
        df: DataFrame with imbalance data
        imbalance_cols: List of imbalance column names
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    imbalance_data = df[imbalance_cols].T
    sns.heatmap(imbalance_data, cmap="RdYlGn", center=0, ax=ax,
                cbar_kws={"label": "Imbalance"})

    ax.set_xlabel("Time")
    ax.set_ylabel("Feature")
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_equity_curve(
    returns: pd.Series,
    title: str = "Equity Curve",
    benchmark_returns: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot cumulative equity curve

    Args:
        returns: Return series
        title: Plot title
        benchmark_returns: Optional benchmark returns
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    equity = (1 + returns).cumprod()
    ax.plot(equity.index, equity.values, label="Strategy", linewidth=2)

    if benchmark_returns is not None:
        benchmark_equity = (1 + benchmark_returns).cumprod()
        ax.plot(benchmark_equity.index, benchmark_equity.values,
                label="Benchmark", linewidth=2, alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = "Drawdown",
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot drawdown series

    Args:
        returns: Return series
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    equity = (1 + returns).cumprod()
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax

    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="red")
    ax.plot(drawdown.index, drawdown.values, color="darkred", linewidth=1)

    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot returns distribution with statistics

    Args:
        returns: Return series
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(returns.dropna(), bins=50, alpha=0.7, edgecolor="black", density=True)
    ax1.axvline(returns.mean(), color="red", linestyle="--",
                label=f"Mean: {returns.mean():.4f}")
    ax1.axvline(returns.median(), color="green", linestyle="--",
                label=f"Median: {returns.median():.4f}")
    ax1.set_xlabel("Returns")
    ax1.set_ylabel("Density")
    ax1.set_title("Histogram")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(returns.dropna(), dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance bar chart

    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        title: Plot title
        top_n: Number of top features to display
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by importance
    indices = np.argsort(importance_values)[-top_n:]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = importance_values[indices]

    # Plot
    ax.barh(range(len(sorted_features)), sorted_importance, alpha=0.7)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot calibration curve for probability predictions

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.calibration import calibration_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    ax1.plot(mean_predicted_value, fraction_of_positives, marker="o",
             linewidth=2, label="Model")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of probabilities
    ax2.hist(y_prob, bins=n_bins, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Predictions")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_capacity_curve(
    adv_percentages: np.ndarray,
    pnl_per_share: np.ndarray,
    title: str = "Capacity Curve",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot strategy capacity curve

    Args:
        adv_percentages: Array of ADV percentages
        pnl_per_share: Array of P&L per share at each capacity level
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(adv_percentages, pnl_per_share, marker="o", linewidth=2, markersize=8)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("% of ADV")
    ax.set_ylabel("P&L per Share")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    fig, ax = plt.subplots(figsize=figsize)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_pnl_attribution(
    pnl_components: Dict[str, float],
    title: str = "P&L Attribution",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot P&L attribution waterfall chart

    Args:
        pnl_components: Dictionary of P&L components
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    components = list(pnl_components.keys())
    values = list(pnl_components.values())

    colors = ["green" if v > 0 else "red" for v in values]
    ax.bar(components, values, color=colors, alpha=0.7, edgecolor="black")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("P&L")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 100) -> None:
    """
    Save figure to file

    Args:
        fig: Matplotlib figure
        filepath: Output file path
        dpi: Resolution in dots per inch
    """
    from pathlib import Path
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    logger.info(f"Figure saved to {filepath}")
    plt.close(fig)
