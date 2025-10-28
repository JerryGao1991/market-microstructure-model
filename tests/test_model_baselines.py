"""
Unit tests for baseline models
Tests Avellaneda-Stoikov, Almgren-Chriss, and Hawkes models
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import unittest
import numpy as np
import pandas as pd

from src.three_model_baselines import AvellanedaStoikov, AlmgrenChriss, HawkesProcess


class TestAvellanedaStoikov(unittest.TestCase):
    """Test cases for Avellaneda-Stoikov market making model"""

    def setUp(self):
        """Set up test fixtures"""
        self.model = AvellanedaStoikov(
            gamma=0.1,
            k=1.5,
            T=1.0,
            max_inventory=100
        )

    def test_reservation_price_zero_inventory(self):
        """Test reservation price with zero inventory"""
        mid_price = 100.0
        inventory = 0
        time_remaining = 1.0

        r = self.model.calculate_reservation_price(mid_price, inventory, time_remaining)

        # With zero inventory, reservation price should equal mid price
        self.assertAlmostEqual(r, mid_price, places=2)

    def test_reservation_price_positive_inventory(self):
        """Test reservation price with positive inventory"""
        mid_price = 100.0
        inventory = 10
        time_remaining = 1.0

        r = self.model.calculate_reservation_price(mid_price, inventory, time_remaining)

        # With positive inventory, reservation price should be below mid
        self.assertLess(r, mid_price)

    def test_optimal_spread_positive(self):
        """Test that optimal spread is always positive"""
        volatility = 0.01
        time_remaining = 1.0

        delta = self.model.calculate_optimal_spread(volatility, time_remaining)

        self.assertGreater(delta, 0)

    def test_get_quotes_spread(self):
        """Test that bid < ask"""
        mid_price = 100.0
        inventory = 0
        volatility = 0.01

        bid, ask = self.model.get_quotes(mid_price, inventory, volatility)

        self.assertLess(bid, ask)
        self.assertGreater(bid, 0)
        self.assertGreater(ask, 0)

    def test_inventory_skew(self):
        """Test that inventory affects quote skew"""
        mid_price = 100.0
        volatility = 0.01

        # Positive inventory should widen ask, tighten bid
        bid_pos, ask_pos = self.model.get_quotes(mid_price, 50, volatility)

        # Negative inventory should tighten ask, widen bid
        bid_neg, ask_neg = self.model.get_quotes(mid_price, -50, volatility)

        # Check that quotes are skewed appropriately
        self.assertLess(bid_pos, bid_neg)
        self.assertLess(ask_pos, ask_neg)


class TestAlmgrenChriss(unittest.TestCase):
    """Test cases for Almgren-Chriss optimal execution model"""

    def setUp(self):
        """Set up test fixtures"""
        self.model = AlmgrenChriss(
            lambda_=0.1,
            eta=0.01,
            sigma=0.02,
            gamma=0.1,
            T=60.0
        )

    def test_trajectory_sums_to_total(self):
        """Test that trade schedule sums to total quantity"""
        total_qty = 1000
        n_intervals = 10

        trades = self.model.calculate_trajectory(total_qty, n_intervals)

        self.assertEqual(len(trades), n_intervals)
        self.assertAlmostEqual(trades.sum(), total_qty, places=1)

    def test_trajectory_all_positive(self):
        """Test that all trades are positive"""
        total_qty = 1000
        trades = self.model.calculate_trajectory(total_qty, 10)

        self.assertTrue(np.all(trades >= 0))

    def test_expected_cost_positive(self):
        """Test that expected cost is positive"""
        total_qty = 1000
        trades = self.model.calculate_trajectory(total_qty, 10)
        cost = self.model.calculate_expected_cost(total_qty, trades)

        self.assertGreater(cost, 0)

    def test_cost_increases_with_quantity(self):
        """Test that cost increases with quantity"""
        qty1 = 500
        qty2 = 1000

        trades1 = self.model.calculate_trajectory(qty1, 10)
        trades2 = self.model.calculate_trajectory(qty2, 10)

        cost1 = self.model.calculate_expected_cost(qty1, trades1)
        cost2 = self.model.calculate_expected_cost(qty2, trades2)

        self.assertLess(cost1, cost2)


class TestHawkesProcess(unittest.TestCase):
    """Test cases for Hawkes process"""

    def setUp(self):
        """Set up test fixtures"""
        self.hawkes = HawkesProcess(
            kernel="exponential",
            baseline_intensity=1.0,
            decay=0.5
        )

    def test_intensity_with_no_history(self):
        """Test intensity equals baseline with no history"""
        t = 0
        event_times = np.array([])

        intensity = self.hawkes.intensity(t, event_times)

        self.assertEqual(intensity, self.hawkes.mu)

    def test_intensity_increases_after_event(self):
        """Test intensity increases after an event"""
        event_times = np.array([0.0])

        # Intensity just after event should be higher than baseline
        intensity_after = self.hawkes.intensity(0.1, event_times)

        self.assertGreater(intensity_after, self.hawkes.mu)

    def test_simulate_produces_events(self):
        """Test that simulation produces events"""
        T = 10.0
        events = self.hawkes.simulate(T)

        self.assertGreater(len(events), 0)
        self.assertTrue(np.all(events >= 0))
        self.assertTrue(np.all(events <= T))

    def test_simulate_events_sorted(self):
        """Test that simulated events are sorted"""
        T = 10.0
        events = self.hawkes.simulate(T)

        self.assertTrue(np.all(np.diff(events) >= 0))

    def test_fit_runs_without_error(self):
        """Test that fitting runs without error"""
        # Generate some events
        event_times = np.sort(np.random.exponential(1.0, 50).cumsum())

        # Fit should not raise
        try:
            self.hawkes.fit(event_times)
            fit_successful = True
        except:
            fit_successful = False

        self.assertTrue(fit_successful)


if __name__ == "__main__":
    unittest.main()
