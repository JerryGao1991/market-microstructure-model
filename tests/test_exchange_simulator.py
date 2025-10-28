"""
Unit tests for exchange simulator and matching engine
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import unittest
import pandas as pd
import numpy as np

from src.config import *
from utils.simulation_utils import Order, OrderSide, OrderType, generate_order_id
from src.four_exchange_simulator import MatchingEngine, ABMSimulator


class TestMatchingEngine(unittest.TestCase):
    """Test cases for matching engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.engine = MatchingEngine(
            instrument_id="TEST_ASSET",
            tick_size=0.01,
            lot_size=1.0
        )

    def test_add_limit_order(self):
        """Test adding limit order to book"""
        order = Order(
            order_id=generate_order_id(),
            instrument_id="TEST_ASSET",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=10,
            timestamp=pd.Timestamp.now()
        )

        self.engine.add_order_to_book(order)

        self.assertEqual(self.engine.get_best_bid(), 100.0)
        self.assertIn(order.order_id, self.engine.active_orders)

    def test_match_order(self):
        """Test order matching"""
        # Add resting sell order
        sell_order = Order(
            order_id=generate_order_id(),
            instrument_id="TEST_ASSET",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=10,
            timestamp=pd.Timestamp.now()
        )
        self.engine.add_order_to_book(sell_order)

        # Match with buy order
        buy_order = Order(
            order_id=generate_order_id(),
            instrument_id="TEST_ASSET",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=10,
            timestamp=pd.Timestamp.now()
        )

        fills = self.engine.match_order(buy_order)

        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].quantity, 10)
        self.assertEqual(fills[0].price, 100.0)

    def test_partial_fill(self):
        """Test partial order fill"""
        # Add resting sell order
        sell_order = Order(
            order_id=generate_order_id(),
            instrument_id="TEST_ASSET",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=5,
            timestamp=pd.Timestamp.now()
        )
        self.engine.add_order_to_book(sell_order)

        # Try to buy more than available
        buy_order = Order(
            order_id=generate_order_id(),
            instrument_id="TEST_ASSET",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=10,
            timestamp=pd.Timestamp.now()
        )

        fills = self.engine.match_order(buy_order)

        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].quantity, 5)

        # Remaining should be in book
        self.assertIn(buy_order.order_id, self.engine.active_orders)
        self.assertEqual(self.engine.active_orders[buy_order.order_id].quantity, 5)

    def test_price_time_priority(self):
        """Test price-time priority"""
        # Add two sell orders at same price
        order1 = Order(
            order_id=generate_order_id(),
            instrument_id="TEST_ASSET",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=5,
            timestamp=pd.Timestamp.now()
        )
        self.engine.add_order_to_book(order1)

        order2 = Order(
            order_id=generate_order_id(),
            instrument_id="TEST_ASSET",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=5,
            timestamp=pd.Timestamp.now() + pd.Timedelta(seconds=1)
        )
        self.engine.add_order_to_book(order2)

        # Buy order should match with order1 first (time priority)
        buy_order = Order(
            order_id=generate_order_id(),
            instrument_id="TEST_ASSET",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=5,
            timestamp=pd.Timestamp.now() + pd.Timedelta(seconds=2)
        )

        fills = self.engine.match_order(buy_order)

        self.assertEqual(len(fills), 1)
        # Order1 should be filled, order2 should remain
        self.assertNotIn(order1.order_id, self.engine.active_orders)
        self.assertIn(order2.order_id, self.engine.active_orders)


class TestABMSimulator(unittest.TestCase):
    """Test cases for ABM simulator"""

    def test_simulation_runs(self):
        """Test that simulation completes"""
        engine = MatchingEngine(
            instrument_id="SIM_ASSET",
            tick_size=0.01,
            lot_size=1.0
        )

        abm = ABMSimulator(engine, n_informed=2, n_noise=5, n_mm=2)
        df_sim = abm.run_simulation(duration_seconds=1, time_step_ms=100)

        self.assertGreater(len(df_sim), 0)
        self.assertIn("mid_price", df_sim.columns)


if __name__ == "__main__":
    unittest.main()
