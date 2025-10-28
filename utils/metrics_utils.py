"""
Performance metrics utilities for strategy evaluation
Includes P&L, TCA, Sharpe ratio, drawdown, and other metrics
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns"""
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns"""
    return np.log(prices / prices.shift(1))


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio (downside deviation)

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Annualized Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
    return sortino


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and its period

    Args:
        equity_curve: Cumulative equity series

    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax

    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()

    # Find peak before trough
    peak_date = equity_curve.loc[:trough_date].idxmax()

    return max_dd, peak_date, trough_date


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown)

    Args:
        returns: Return series
        periods_per_year: Number of periods in a year

    Returns:
        Calmar ratio
    """
    equity_curve = (1 + returns).cumprod()
    annual_return = returns.mean() * periods_per_year

    max_dd, _, _ = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return 0.0

    return annual_return / abs(max_dd)


def calculate_value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR) using historical method

    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        VaR value
    """
    return np.percentile(returns.dropna(), (1 - confidence_level) * 100)


def calculate_conditional_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall)

    Args:
        returns: Return series
        confidence_level: Confidence level

    Returns:
        CVaR value
    """
    var = calculate_value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_win_rate(trades: pd.DataFrame, pnl_col: str = "pnl") -> float:
    """
    Calculate win rate from trades

    Args:
        trades: DataFrame with trade information
        pnl_col: Column name for P&L

    Returns:
        Win rate (0 to 1)
    """
    if len(trades) == 0:
        return 0.0

    winning_trades = (trades[pnl_col] > 0).sum()
    return winning_trades / len(trades)


def calculate_profit_factor(trades: pd.DataFrame, pnl_col: str = "pnl") -> float:
    """
    Calculate profit factor (gross profit / gross loss)

    Args:
        trades: DataFrame with trade information
        pnl_col: Column name for P&L

    Returns:
        Profit factor
    """
    gross_profit = trades[trades[pnl_col] > 0][pnl_col].sum()
    gross_loss = abs(trades[trades[pnl_col] < 0][pnl_col].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_average_trade_pnl(
    trades: pd.DataFrame,
    pnl_col: str = "pnl"
) -> Dict[str, float]:
    """
    Calculate average trade statistics

    Args:
        trades: DataFrame with trade information
        pnl_col: Column name for P&L

    Returns:
        Dictionary with average trade metrics
    """
    winning_trades = trades[trades[pnl_col] > 0]
    losing_trades = trades[trades[pnl_col] < 0]

    return {
        "avg_trade": trades[pnl_col].mean(),
        "avg_win": winning_trades[pnl_col].mean() if len(winning_trades) > 0 else 0,
        "avg_loss": losing_trades[pnl_col].mean() if len(losing_trades) > 0 else 0,
        "win_loss_ratio": (
            abs(winning_trades[pnl_col].mean() / losing_trades[pnl_col].mean())
            if len(losing_trades) > 0 and len(winning_trades) > 0
            else 0
        )
    }


def calculate_tca_metrics(
    trades: pd.DataFrame,
    arrival_price_col: str = "arrival_price",
    execution_price_col: str = "execution_price",
    side_col: str = "side",
    quantity_col: str = "quantity",
    fee_col: str = "fee"
) -> pd.DataFrame:
    """
    Calculate Transaction Cost Analysis metrics

    Args:
        trades: DataFrame with trade execution data
        arrival_price_col: Arrival price column
        execution_price_col: Execution price column
        side_col: Side column ("B" for buy, "S" for sell)
        quantity_col: Quantity column
        fee_col: Fee column

    Returns:
        DataFrame with TCA metrics
    """
    result = trades.copy()

    # Calculate slippage
    result["slippage"] = np.where(
        result[side_col] == "B",
        result[execution_price_col] - result[arrival_price_col],
        result[arrival_price_col] - result[execution_price_col]
    )

    # Slippage in basis points
    result["slippage_bps"] = (
        result["slippage"] / result[arrival_price_col] * 10000
    )

    # Total cost (slippage + fees)
    result["total_cost"] = result["slippage"] * result[quantity_col] + result[fee_col]

    # Total cost in basis points
    result["total_cost_bps"] = (
        result["total_cost"] / (result[arrival_price_col] * result[quantity_col]) * 10000
    )

    return result


def calculate_implementation_shortfall(
    trades: pd.DataFrame,
    benchmark_price: float,
    side: str,
    total_quantity: float
) -> Dict[str, float]:
    """
    Calculate implementation shortfall components

    Args:
        trades: DataFrame with executed trades
        benchmark_price: Benchmark price (e.g., decision price)
        side: Trade side ("B" or "S")
        total_quantity: Total order quantity

    Returns:
        Dictionary with IS components
    """
    executed_qty = trades["quantity"].sum()
    executed_value = (trades["execution_price"] * trades["quantity"]).sum()
    avg_execution_price = executed_value / executed_qty if executed_qty > 0 else 0

    # Calculate slippage
    if side == "B":
        slippage = avg_execution_price - benchmark_price
    else:
        slippage = benchmark_price - avg_execution_price

    # Impact cost (deviation from benchmark)
    impact_cost = slippage * executed_qty

    # Opportunity cost (unfilled quantity)
    unfilled_qty = total_quantity - executed_qty

    # Total cost
    total_fees = trades["fee"].sum() if "fee" in trades.columns else 0

    return {
        "executed_qty": executed_qty,
        "avg_execution_price": avg_execution_price,
        "slippage_bps": (slippage / benchmark_price) * 10000 if benchmark_price > 0 else 0,
        "impact_cost": impact_cost,
        "opportunity_cost": unfilled_qty,
        "fees": total_fees,
        "total_cost": impact_cost + total_fees,
    }


def calculate_turnover(
    positions: pd.Series,
    values: pd.Series
) -> pd.Series:
    """
    Calculate portfolio turnover

    Args:
        positions: Position series
        values: Portfolio value series

    Returns:
        Turnover series
    """
    position_changes = positions.diff().abs()
    turnover = position_changes / values.shift(1)
    return turnover.fillna(0)


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate information ratio

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods in a year

    Returns:
        Annualized information ratio
    """
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std()

    if tracking_error == 0:
        return 0.0

    ir = np.sqrt(periods_per_year) * active_returns.mean() / tracking_error
    return ir


def calculate_capacity_metrics(
    daily_volume: pd.Series,
    strategy_volume: pd.Series,
    pnl: pd.Series
) -> pd.DataFrame:
    """
    Calculate capacity analysis metrics

    Args:
        daily_volume: Daily market volume
        strategy_volume: Strategy trading volume
        pnl: Strategy P&L

    Returns:
        DataFrame with capacity metrics
    """
    result = pd.DataFrame({
        "date": daily_volume.index,
        "market_volume": daily_volume.values,
        "strategy_volume": strategy_volume.values,
        "pnl": pnl.values
    })

    # Calculate percentage of daily volume
    result["pct_of_adv"] = (
        result["strategy_volume"] / result["market_volume"] * 100
    )

    # Calculate P&L per unit of volume
    result["pnl_per_volume"] = np.where(
        result["strategy_volume"] > 0,
        result["pnl"] / result["strategy_volume"],
        0
    )

    return result


def calculate_performance_summary(
    returns: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive performance summary

    Args:
        returns: Return series
        trades: Optional DataFrame with trade information
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Dictionary with performance metrics
    """
    equity_curve = (1 + returns).cumprod()

    metrics = {
        "total_return": (equity_curve.iloc[-1] - 1) * 100,
        "annual_return": returns.mean() * periods_per_year * 100,
        "annual_volatility": returns.std() * np.sqrt(periods_per_year) * 100,
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar_ratio": calculate_calmar_ratio(returns, periods_per_year),
        "max_drawdown": calculate_max_drawdown(equity_curve)[0] * 100,
        "var_95": calculate_value_at_risk(returns, 0.95) * 100,
        "cvar_95": calculate_conditional_var(returns, 0.95) * 100,
    }

    if trades is not None and len(trades) > 0:
        metrics["num_trades"] = len(trades)
        metrics["win_rate"] = calculate_win_rate(trades) * 100
        metrics["profit_factor"] = calculate_profit_factor(trades)
        metrics.update(calculate_average_trade_pnl(trades))

    return metrics
