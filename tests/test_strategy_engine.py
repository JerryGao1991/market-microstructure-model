"""
Unit tests for strategy engine
Tests signal generation, order creation, and risk management
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import unittest
import numpy as np
import pandas as pd

from src.five_strategy_engine import StrategyEngine, RiskManager, Signal
from utils.simulation_utils import OrderSide


class TestRiskManager(unittest.TestCase):
    """Test cases for risk manager"""

    def setUp(self):
        """Set up test fixtures"""
        self.risk_manager = RiskManager()

    def test_initial_state(self):
        """Test initial risk manager state"""
        self.assertEqual(self.risk_manager.daily_pnl, 0.0)
        self.assertFalse(self.risk_manager.kill_switch_active)

    def test_kill_switch_daily_loss(self):
        """Test kill switch triggers on daily loss"""
        # Set loss beyond limit
        self.risk_manager.daily_pnl = -15000  # Exceeds max_daily_loss

        triggered = self.risk_manager.check_kill_switch()

        self.assertTrue(triggered)
        self.assertTrue(self.risk_manager.kill_switch_active)

    def test_kill_switch_drawdown(self):
        """Test kill switch triggers on drawdown"""
        # Simulate profit then loss
        self.risk_manager.peak_pnl = 10000
        self.risk_manager.daily_pnl = 4000  # 60% drawdown

        triggered = self.risk_manager.check_kill_switch()

        self.assertTrue(triggered)

    def test_position_limits_respected(self):
        """Test position limit checking"""
        current_position = 900
        order_quantity = 200  # Would exceed 1000 limit

        valid, msg = self.risk_manager.check_position_limits(
            current_position, order_quantity
        )

        self.assertFalse(valid)
        self.assertIn("Position limit", msg)

    def test_order_size_limits(self):
        """Test order size limit checking"""
        large_order = 150  # Exceeds 100 limit

        valid, msg = self.risk_manager.check_order_size(large_order)

        self.assertFalse(valid)
        self.assertIn("Order size", msg)

    def test_price_collar(self):
        """Test price collar protection"""
        order_price = 105.0
        mid_price = 100.0  # 5% deviation, exceeds 1% collar

        valid, msg = self.risk_manager.check_price_collar(order_price, mid_price)

        self.assertFalse(valid)
        self.assertIn("collar", msg)

    def test_pnl_update(self):
        """Test P&L tracking"""
        initial_pnl = self.risk_manager.daily_pnl
        pnl_change = 100.0

        self.risk_manager.update_pnl(pnl_change)

        self.assertEqual(self.risk_manager.daily_pnl, initial_pnl + pnl_change)


class TestStrategyEngine(unittest.TestCase):
    """Test cases for strategy engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.strategy = StrategyEngine("TEST_ASSET")

    def test_initial_position(self):
        """Test initial position state"""
        self.assertEqual(self.strategy.position.quantity, 0)
        self.assertEqual(self.strategy.position.realized_pnl, 0.0)

    def test_signal_generation_long(self):
        """Test long signal generation"""
        # Probability heavily favoring up
        prediction_proba = np.array([0.1, 0.2, 0.7])

        signal = self.strategy.generate_signal(
            pd.Timestamp.now(),
            prediction_proba
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, "LONG")

    def test_signal_generation_short(self):
        """Test short signal generation"""
        # Probability heavily favoring down
        prediction_proba = np.array([0.7, 0.2, 0.1])

        signal = self.strategy.generate_signal(
            pd.Timestamp.now(),
            prediction_proba
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, "SHORT")

    def test_signal_generation_neutral(self):
        """Test neutral signal (no trade)"""
        # Balanced probabilities
        prediction_proba = np.array([0.33, 0.34, 0.33])

        signal = self.strategy.generate_signal(
            pd.Timestamp.now(),
            prediction_proba
        )

        # Should return None or NEUTRAL
        self.assertTrue(signal is None or signal.direction == "NEUTRAL")

    def test_signal_to_order_long(self):
        """Test converting long signal to buy order"""
        signal = Signal(
            timestamp=pd.Timestamp.now(),
            instrument_id="TEST_ASSET",
            direction="LONG",
            probability=0.7,
            strength=0.4
        )

        order = self.strategy.signal_to_order(
            signal,
            current_bid=99.99,
            current_ask=100.01,
            mid_price=100.0
        )

        self.assertIsNotNone(order)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertGreater(order.quantity, 0)

    def test_signal_to_order_short(self):
        """Test converting short signal to sell order"""
        signal = Signal(
            timestamp=pd.Timestamp.now(),
            instrument_id="TEST_ASSET",
            direction="SHORT",
            probability=0.7,
            strength=0.4
        )

        order = self.strategy.signal_to_order(
            signal,
            current_bid=99.99,
            current_ask=100.01,
            mid_price=100.0
        )

        self.assertIsNotNone(order)
        self.assertEqual(order.side, OrderSide.SELL)

    def test_process_fill_buy(self):
        """Test processing buy fill"""
        from utils.simulation_utils import Order, OrderType

        order = Order(
            order_id="TEST_001",
            instrument_id="TEST_ASSET",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=10,
            timestamp=pd.Timestamp.now()
        )

        initial_position = self.strategy.position.quantity

        self.strategy.process_fill(order, 100.0, 10)

        # Position should increase
        self.assertEqual(self.strategy.position.quantity, initial_position + 10)
        self.assertEqual(self.strategy.position.avg_price, 100.0)

    def test_process_fill_sell_with_profit(self):
        """Test processing sell fill with profit"""
        from utils.simulation_utils import Order, OrderType

        # First buy at 100
        buy_order = Order(
            order_id="TEST_001",
            instrument_id="TEST_ASSET",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=10,
            timestamp=pd.Timestamp.now()
        )
        self.strategy.process_fill(buy_order, 100.0, 10)

        # Then sell at 105
        sell_order = Order(
            order_id="TEST_002",
            instrument_id="TEST_ASSET",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=105.0,
            quantity=10,
            timestamp=pd.Timestamp.now()
        )
        self.strategy.process_fill(sell_order, 105.0, 10)

        # Should have realized profit
        expected_profit = (105.0 - 100.0) * 10
        self.assertAlmostEqual(self.strategy.position.realized_pnl, expected_profit, places=2)
        self.assertEqual(self.strategy.position.quantity, 0)

    def test_calculate_unrealized_pnl(self):
        """Test unrealized P&L calculation"""
        from utils.simulation_utils import Order, OrderType

        # Buy at 100
        buy_order = Order(
            order_id="TEST_001",
            instrument_id="TEST_ASSET",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=10,
            timestamp=pd.Timestamp.now()
        )
        self.strategy.process_fill(buy_order, 100.0, 10)

        # Current price is 105
        self.strategy.calculate_unrealized_pnl(105.0)

        expected_unrealized = (105.0 - 100.0) * 10
        self.assertAlmostEqual(
            self.strategy.position.unrealized_pnl,
            expected_unrealized,
            places=2
        )

    def test_get_position_summary(self):
        """Test position summary"""
        summary = self.strategy.get_position_summary()

        self.assertIn("instrument_id", summary)
        self.assertIn("quantity", summary)
        self.assertIn("realized_pnl", summary)
        self.assertIn("unrealized_pnl", summary)
        self.assertIn("total_pnl", summary)


if __name__ == "__main__":
    unittest.main()
