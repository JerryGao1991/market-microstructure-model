"""
Exchange Simulator: Matching Engine + ABM Simulation
Implements price-time priority matching, queue tracking, and agent-based modeling
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

from config import *
from utils.simulation_utils import (
    Order, Fill, OrderSide, OrderType, TraderType,
    generate_order_id, generate_fill_id, round_to_tick, round_to_lot,
    generate_poisson_arrivals, generate_informed_order_flow,
    generate_noise_order, generate_market_maker_quotes,
    validate_order, check_self_trade
)

# logging.basicConfig(**LOGGING_CONFIG)
configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Single price level in order book"""
    price: float
    orders: deque = field(default_factory=deque)  # FIFO queue

    @property
    def total_quantity(self) -> float:
        return sum(order.quantity for order in self.orders)


class MatchingEngine:
    """
    Price-time priority matching engine
    Simulates exchange-level order matching and queue dynamics
    """

    def __init__(
        self,
        instrument_id: str,
        tick_size: float = 0.01,
        lot_size: float = 1.0,
        enable_self_trade_prevention: bool = True
    ):
        self.instrument_id = instrument_id
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.enable_self_trade_prevention = enable_self_trade_prevention

        # Order books: price -> OrderBookLevel
        self.bid_book: Dict[float, OrderBookLevel] = {}
        self.ask_book: Dict[float, OrderBookLevel] = {}

        # Order tracking
        self.active_orders: Dict[str, Order] = {}

        # Execution history
        self.fills: List[Fill] = []
        self.order_history: List[Order] = []

        # Statistics
        self.total_volume = 0
        self.total_trades = 0

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        if not self.bid_book:
            return None
        return max(self.bid_book.keys())

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        if not self.ask_book:
            return None
        return min(self.ask_book.keys())

    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()

        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None

    def add_order_to_book(self, order: Order) -> None:
        """Add limit order to book"""
        if order.side == OrderSide.BUY:
            if order.price not in self.bid_book:
                self.bid_book[order.price] = OrderBookLevel(order.price)
            self.bid_book[order.price].orders.append(order)
        else:
            if order.price not in self.ask_book:
                self.ask_book[order.price] = OrderBookLevel(order.price)
            self.ask_book[order.price].orders.append(order)

        self.active_orders[order.order_id] = order

    def remove_order(self, order_id: str) -> bool:
        """Remove order from book"""
        if order_id not in self.active_orders:
            return False

        order = self.active_orders[order_id]

        if order.side == OrderSide.BUY:
            if order.price in self.bid_book:
                level = self.bid_book[order.price]
                level.orders = deque([o for o in level.orders if o.order_id != order_id])
                if len(level.orders) == 0:
                    del self.bid_book[order.price]
        else:
            if order.price in self.ask_book:
                level = self.ask_book[order.price]
                level.orders = deque([o for o in level.orders if o.order_id != order_id])
                if len(level.orders) == 0:
                    del self.ask_book[order.price]

        del self.active_orders[order_id]
        return True

    def match_order(self, order: Order) -> List[Fill]:
        """
        Match incoming order against book

        Args:
            order: Incoming order

        Returns:
            List of fills generated
        """
        fills = []
        remaining_qty = order.quantity

        if order.side == OrderSide.BUY:
            # Match against ask book
            while remaining_qty > 0 and self.ask_book:
                best_ask = self.get_best_ask()

                # Check if price crosses
                if order.order_type == OrderType.MARKET or order.price >= best_ask:
                    level = self.ask_book[best_ask]

                    while remaining_qty > 0 and len(level.orders) > 0:
                        resting_order = level.orders[0]

                        # Check self-trade
                        if self.enable_self_trade_prevention:
                            if (order.trader_id is not None and
                                order.trader_id == resting_order.trader_id):
                                level.orders.popleft()
                                continue

                        # Calculate fill quantity
                        fill_qty = min(remaining_qty, resting_order.quantity)

                        # Create fills
                        fill = Fill(
                            fill_id=generate_fill_id(),
                            order_id=order.order_id,
                            instrument_id=self.instrument_id,
                            side=order.side,
                            price=best_ask,  # Trade at resting order price
                            quantity=fill_qty,
                            timestamp=order.timestamp,
                            is_aggressor=True
                        )
                        fills.append(fill)

                        # Update quantities
                        remaining_qty -= fill_qty
                        resting_order.quantity -= fill_qty

                        # Remove if fully filled
                        if resting_order.quantity == 0:
                            level.orders.popleft()
                            del self.active_orders[resting_order.order_id]

                    # Remove empty level
                    if len(level.orders) == 0:
                        del self.ask_book[best_ask]
                else:
                    break

        else:  # SELL
            # Match against bid book
            while remaining_qty > 0 and self.bid_book:
                best_bid = self.get_best_bid()

                if order.order_type == OrderType.MARKET or order.price <= best_bid:
                    level = self.bid_book[best_bid]

                    while remaining_qty > 0 and len(level.orders) > 0:
                        resting_order = level.orders[0]

                        # Check self-trade
                        if self.enable_self_trade_prevention:
                            if (order.trader_id is not None and
                                order.trader_id == resting_order.trader_id):
                                level.orders.popleft()
                                continue

                        fill_qty = min(remaining_qty, resting_order.quantity)

                        fill = Fill(
                            fill_id=generate_fill_id(),
                            order_id=order.order_id,
                            instrument_id=self.instrument_id,
                            side=order.side,
                            price=best_bid,
                            quantity=fill_qty,
                            timestamp=order.timestamp,
                            is_aggressor=True
                        )
                        fills.append(fill)

                        remaining_qty -= fill_qty
                        resting_order.quantity -= fill_qty

                        if resting_order.quantity == 0:
                            level.orders.popleft()
                            del self.active_orders[resting_order.order_id]

                    if len(level.orders) == 0:
                        del self.bid_book[best_bid]
                else:
                    break

        # Record fills
        self.fills.extend(fills)
        self.total_trades += len(fills)
        self.total_volume += sum(f.quantity for f in fills)

        # If partially filled or not filled at all, add remainder to book
        if remaining_qty > 0 and order.order_type == OrderType.LIMIT:
            order.quantity = remaining_qty
            self.add_order_to_book(order)

        self.order_history.append(order)

        return fills

    def get_order_book_snapshot(self, n_levels: int = 10) -> Dict:
        """Get current order book snapshot"""
        snapshot = {
            "timestamp": pd.Timestamp.now(),
            "mid_price": self.get_mid_price()
        }

        # Bid side
        bid_prices = sorted(self.bid_book.keys(), reverse=True)[:n_levels]
        for i, price in enumerate(bid_prices, 1):
            snapshot[f"bid_px_{i}"] = price
            snapshot[f"bid_sz_{i}"] = self.bid_book[price].total_quantity

        # Ask side
        ask_prices = sorted(self.ask_book.keys())[:n_levels]
        for i, price in enumerate(ask_prices, 1):
            snapshot[f"ask_px_{i}"] = price
            snapshot[f"ask_sz_{i}"] = self.ask_book[price].total_quantity

        return snapshot


class ABMSimulator:
    """
    Agent-Based Model Simulator
    Simulates multiple trader types: informed, noise, market makers
    """

    def __init__(
        self,
        matching_engine: MatchingEngine,
        n_informed: int = ABM_CONFIG["n_informed_traders"],
        n_noise: int = ABM_CONFIG["n_noise_traders"],
        n_mm: int = ABM_CONFIG["n_market_makers"]
    ):
        self.engine = matching_engine
        self.n_informed = n_informed
        self.n_noise = n_noise
        self.n_mm = n_mm

        # Agent states
        self.mm_inventories = {f"MM_{i}": 0 for i in range(n_mm)}

        # Simulation parameters
        self.true_value = 100.0  # Informed traders' estimate
        self.tick_size = matching_engine.tick_size
        self.lot_size = matching_engine.lot_size

    def step(self, current_time: pd.Timestamp) -> List[Order]:
        """
        Execute one simulation step

        Args:
            current_time: Current simulation time

        Returns:
            List of orders generated
        """
        orders = []

        current_bid = self.engine.get_best_bid()
        current_ask = self.engine.get_best_ask()
        mid_price = self.engine.get_mid_price()

        if mid_price is None:
            mid_price = self.true_value
            current_bid = mid_price - self.tick_size
            current_ask = mid_price + self.tick_size

        # Informed traders
        for i in range(self.n_informed):
            order = generate_informed_order_flow(
                current_price=mid_price,
                true_value=self.true_value,
                intensity=ABM_CONFIG["informed_trade_intensity"],
                quantity_mean=50,
                quantity_std=20,
                tick_size=self.tick_size,
                lot_size=self.lot_size
            )
            if order is not None:
                order.trader_id = f"INF_{i}"
                order.timestamp = current_time
                orders.append(order)

        # Noise traders
        for i in range(self.n_noise):
            order = generate_noise_order(
                current_bid=current_bid,
                current_ask=current_ask,
                intensity=ABM_CONFIG["noise_trade_intensity"],
                quantity_mean=30,
                quantity_std=15,
                tick_size=self.tick_size,
                lot_size=self.lot_size
            )
            if order is not None:
                order.trader_id = f"NOISE_{i}"
                order.timestamp = current_time
                orders.append(order)

        # Market makers
        spread = 2 * self.tick_size
        for i in range(self.n_mm):
            mm_id = f"MM_{i}"
            inventory = self.mm_inventories[mm_id]

            bid_order, ask_order = generate_market_maker_quotes(
                mid_price=mid_price,
                spread=spread,
                inventory=inventory,
                max_inventory=100,
                risk_aversion=0.01,
                quantity=50,
                tick_size=self.tick_size,
                lot_size=self.lot_size
            )

            bid_order.trader_id = mm_id
            bid_order.timestamp = current_time
            ask_order.trader_id = mm_id
            ask_order.timestamp = current_time

            orders.append(bid_order)
            orders.append(ask_order)

        return orders

    def run_simulation(
        self,
        duration_seconds: float = ABM_CONFIG["simulation_duration_seconds"],
        time_step_ms: float = 100
    ) -> pd.DataFrame:
        """
        Run full ABM simulation

        Args:
            duration_seconds: Simulation duration
            time_step_ms: Time step in milliseconds

        Returns:
            DataFrame with simulation results
        """
        logger.info(f"Running ABM simulation for {duration_seconds} seconds")

        start_time = pd.Timestamp.now()
        current_time = start_time
        end_time = start_time + pd.Timedelta(seconds=duration_seconds)

        snapshots = []

        while current_time < end_time:
            # Generate agent orders
            orders = self.step(current_time)

            # Submit orders to matching engine
            for order in orders:
                fills = self.engine.match_order(order)

                # Update market maker inventories
                for fill in fills:
                    if fill.order_id in [o.order_id for o in orders]:
                        order_obj = [o for o in orders if o.order_id == fill.order_id][0]
                        if order_obj.trader_id and order_obj.trader_id.startswith("MM_"):
                            if fill.side == OrderSide.BUY:
                                self.mm_inventories[order_obj.trader_id] += fill.quantity
                            else:
                                self.mm_inventories[order_obj.trader_id] -= fill.quantity

            # Record snapshot
            snapshot = self.engine.get_order_book_snapshot()
            snapshot["timestamp"] = current_time
            snapshots.append(snapshot)

            # Advance time
            current_time += pd.Timedelta(milliseconds=time_step_ms)

        df = pd.DataFrame(snapshots)
        logger.info(f"Simulation complete: {len(df)} snapshots, {self.engine.total_trades} trades")

        return df


def main():
    """Main execution function"""
    logger.info("Starting exchange simulator and ABM")

    # Initialize matching engine
    engine = MatchingEngine(
        instrument_id="SIM_ASSET",
        tick_size=0.01,
        lot_size=1.0
    )

    # Run ABM simulation
    abm = ABMSimulator(engine)
    df_sim = abm.run_simulation(duration_seconds=60, time_step_ms=100)

    # Save simulation results
    output_file = SIMULATION_PATH / "abm_simulation_results.parquet"
    from utils.io_utils import write_parquet
    write_parquet(df_sim, output_file)

    logger.info(f"Simulation results saved to {output_file}")
    logger.info(f"Final mid price: {df_sim['mid_price'].iloc[-1]:.2f}")
    logger.info(f"Total trades: {engine.total_trades}")
    logger.info(f"Total volume: {engine.total_volume}")

    logger.info("Exchange simulator and ABM complete")


if __name__ == "__main__":
    main()
