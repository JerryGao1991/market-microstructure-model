"""
Strategy Engine: Signal-to-Execution Pipeline
Converts model predictions to trading actions with risk controls
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from config import *
from utils.simulation_utils import Order, OrderSide, OrderType, generate_order_id

# logging.basicConfig(**LOGGING_CONFIG)
configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal"""
    timestamp: pd.Timestamp
    instrument_id: str
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    probability: float
    strength: float


@dataclass
class Position:
    """Current position state"""
    instrument_id: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class RiskManager:
    """Risk management and circuit breakers"""

    def __init__(self):
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_pnl = 0.0
        self.kill_switch_active = False
        self.total_notional_today = 0.0

    def check_kill_switch(self) -> bool:
        """Check if kill switch should be triggered"""
        if not RISK_CONFIG["enable_kill_switch"]:
            return False

        # Max daily loss
        if self.daily_pnl < -RISK_CONFIG["max_daily_loss"]:
            logger.warning(f"Kill switch: Daily loss limit breached ({self.daily_pnl})")
            self.kill_switch_active = True
            return True

        # Max drawdown
        if self.daily_pnl > self.peak_pnl:
            self.peak_pnl = self.daily_pnl

        current_drawdown = (self.peak_pnl - self.daily_pnl) / max(abs(self.peak_pnl), 1)

        if current_drawdown > RISK_CONFIG["max_drawdown_pct"] / 100:
            logger.warning(f"Kill switch: Drawdown limit breached ({current_drawdown:.2%})")
            self.kill_switch_active = True
            return True

        return False

    def check_position_limits(
        self,
        current_position: float,
        order_quantity: float
    ) -> Tuple[bool, str]:
        """Check if order would breach position limits"""
        new_position = abs(current_position + order_quantity)

        if new_position > POSITION_CONFIG["max_position_size"]:
            return False, f"Position limit breached: {new_position}"

        return True, ""

    def check_order_size(self, order_quantity: float) -> Tuple[bool, str]:
        """Check if order size is within limits"""
        if abs(order_quantity) > POSITION_CONFIG["max_order_size"]:
            return False, f"Order size exceeds limit: {abs(order_quantity)}"

        return True, ""

    def check_price_collar(
        self,
        order_price: float,
        mid_price: float
    ) -> Tuple[bool, str]:
        """Check if order price is within collar"""
        if not RISK_CONFIG["enable_limit_price_protection"]:
            return True, ""

        collar_pct = RISK_CONFIG["price_collar_pct"] / 100
        max_deviation = mid_price * collar_pct

        if abs(order_price - mid_price) > max_deviation:
            return False, f"Price outside collar: {abs(order_price - mid_price):.4f}"

        return True, ""

    def update_pnl(self, pnl_change: float):
        """Update P&L tracking"""
        self.daily_pnl += pnl_change


class StrategyEngine:
    """
    Main strategy engine
    Converts signals to orders with risk management
    """

    def __init__(self, instrument_id: str):
        self.instrument_id = instrument_id
        self.risk_manager = RiskManager()
        self.position = Position(
            instrument_id=instrument_id,
            quantity=0,
            avg_price=0.0
        )
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.trade_count_today = 0

    def generate_signal(
        self,
        timestamp: pd.Timestamp,
        prediction_proba: np.ndarray
    ) -> Optional[Signal]:
        """
        Generate trading signal from model prediction

        Args:
            timestamp: Current timestamp
            prediction_proba: Array of class probabilities [down, neutral, up]

        Returns:
            Signal or None
        """
        # Extract probabilities
        prob_down = prediction_proba[0]
        prob_neutral = prediction_proba[1]
        prob_up = prediction_proba[2]

        # Determine direction
        if prob_up > SIGNAL_CONFIG["long_threshold"]:
            direction = "LONG"
            probability = prob_up
            strength = prob_up - prob_down
        elif prob_down > (1 - SIGNAL_CONFIG["short_threshold"]):
            direction = "SHORT"
            probability = prob_down
            strength = prob_down - prob_up
        else:
            direction = "NEUTRAL"
            probability = prob_neutral
            strength = 0.0

        # Check minimum strength
        if abs(strength) < SIGNAL_CONFIG["min_signal_strength"]:
            return None

        return Signal(
            timestamp=timestamp,
            instrument_id=self.instrument_id,
            direction=direction,
            probability=probability,
            strength=strength
        )

    def signal_to_order(
        self,
        signal: Signal,
        current_bid: float,
        current_ask: float,
        mid_price: float
    ) -> Optional[Order]:
        """
        Convert signal to executable order

        Args:
            signal: Trading signal
            current_bid: Current best bid
            current_ask: Current best ask
            mid_price: Current mid price

        Returns:
            Order or None
        """
        # Check kill switch
        if self.risk_manager.check_kill_switch():
            logger.warning("Kill switch active, no new orders")
            return None

        # Check daily trade limit
        if self.trade_count_today >= POSITION_CONFIG["max_daily_trades"]:
            logger.warning("Daily trade limit reached")
            return None

        # Determine order side and quantity
        if signal.direction == "LONG":
            side = OrderSide.BUY
            # Scale quantity by signal strength
            base_qty = POSITION_CONFIG["max_order_size"]
            quantity = base_qty * signal.strength
        elif signal.direction == "SHORT":
            side = OrderSide.SELL
            base_qty = POSITION_CONFIG["max_order_size"]
            quantity = base_qty * signal.strength
        else:
            return None

        # Check order size
        valid, msg = self.risk_manager.check_order_size(quantity)
        if not valid:
            logger.warning(f"Order rejected: {msg}")
            return None

        # Check position limits
        new_qty = quantity if side == OrderSide.BUY else -quantity
        valid, msg = self.risk_manager.check_position_limits(
            self.position.quantity, new_qty
        )
        if not valid:
            logger.warning(f"Order rejected: {msg}")
            return None

        # Determine order price (limit order)
        if side == OrderSide.BUY:
            # Buy at bid or slightly below
            order_price = current_bid
        else:
            # Sell at ask or slightly above
            order_price = current_ask

        # Check price collar
        valid, msg = self.risk_manager.check_price_collar(order_price, mid_price)
        if not valid:
            logger.warning(f"Order rejected: {msg}")
            return None

        # Create order
        order = Order(
            order_id=generate_order_id("STRAT"),
            instrument_id=self.instrument_id,
            side=side,
            order_type=OrderType.LIMIT,
            price=order_price,
            quantity=quantity,
            timestamp=signal.timestamp
        )

        return order

    def process_fill(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float
    ):
        """
        Process order fill and update position

        Args:
            order: Filled order
            fill_price: Execution price
            fill_quantity: Filled quantity
        """
        if order.side == OrderSide.BUY:
            # Update position
            total_cost = self.position.quantity * self.position.avg_price
            new_cost = fill_quantity * fill_price
            new_quantity = self.position.quantity + fill_quantity

            if new_quantity > 0:
                self.position.avg_price = (total_cost + new_cost) / new_quantity
            self.position.quantity = new_quantity

        else:  # SELL
            # Calculate realized P&L
            realized_pnl = (fill_price - self.position.avg_price) * fill_quantity
            self.position.realized_pnl += realized_pnl
            self.risk_manager.update_pnl(realized_pnl)

            self.position.quantity -= fill_quantity

        self.trade_count_today += 1
        self.filled_orders.append(order)

        logger.info(
            f"Fill processed: {order.side.value} {fill_quantity} @ {fill_price:.4f}, "
            f"Position: {self.position.quantity}, PnL: {self.position.realized_pnl:.2f}"
        )

    def calculate_unrealized_pnl(self, current_price: float):
        """Calculate unrealized P&L"""
        if self.position.quantity == 0:
            self.position.unrealized_pnl = 0.0
        else:
            self.position.unrealized_pnl = (
                (current_price - self.position.avg_price) * self.position.quantity
            )

    def get_position_summary(self) -> Dict:
        """Get current position summary"""
        return {
            "instrument_id": self.position.instrument_id,
            "quantity": self.position.quantity,
            "avg_price": self.position.avg_price,
            "unrealized_pnl": self.position.unrealized_pnl,
            "realized_pnl": self.position.realized_pnl,
            "total_pnl": self.position.unrealized_pnl + self.position.realized_pnl,
            "trades_today": self.trade_count_today
        }


def main():
    """Main execution function"""
    logger.info("Strategy engine example")

    # Initialize strategy
    strategy = StrategyEngine("AAPL.P.XNAS")

    # Simulate some predictions and trading
    mid_price = 150.0

    for i in range(10):
        timestamp = pd.Timestamp.now() + pd.Timedelta(seconds=i)

        # Simulate model prediction (random for demo)
        prediction_proba = np.random.dirichlet([1, 1, 1])

        # Generate signal
        signal = strategy.generate_signal(timestamp, prediction_proba)

        if signal is not None:
            logger.info(f"Signal: {signal.direction}, Strength: {signal.strength:.3f}")

            # Convert to order
            order = strategy.signal_to_order(
                signal,
                current_bid=mid_price - 0.01,
                current_ask=mid_price + 0.01,
                mid_price=mid_price
            )

            if order is not None:
                logger.info(f"Order: {order.side.value} {order.quantity:.0f} @ {order.price:.4f}")

                # Simulate fill
                strategy.process_fill(order, order.price, order.quantity)

        # Update unrealized P&L
        strategy.calculate_unrealized_pnl(mid_price)

        # Update mid price randomly
        mid_price += np.random.randn() * 0.1

    # Final summary
    summary = strategy.get_position_summary()
    logger.info(f"Final position: {summary}")

    logger.info("Strategy engine complete")


if __name__ == "__main__":
    main()
