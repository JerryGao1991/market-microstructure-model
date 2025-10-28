"""
Baseline Models: Avellaneda-Stoikov, Almgren-Chriss, and Hawkes Process
Interpretable models for comparison with deep learning approaches
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import expon
import logging
from typing import Tuple, Dict, Optional

from config import *

# logging.basicConfig(**LOGGING_CONFIG)
configure_logging()
logger = logging.getLogger(__name__)


class AvellanedaStoikov:
    """
    Avellaneda-Stoikov market making model
    Optimal bid/ask quotes with inventory management
    """

    def __init__(
        self,
        gamma: float = AS_CONFIG["gamma"],
        k: float = AS_CONFIG["k"],
        T: float = AS_CONFIG["T"],
        max_inventory: int = AS_CONFIG["max_inventory"]
    ):
        self.gamma = gamma  # Risk aversion
        self.k = k  # Order book liquidity parameter
        self.T = T  # Time horizon
        self.max_inventory = max_inventory

    def calculate_reservation_price(
        self,
        mid_price: float,
        inventory: int,
        time_remaining: float
    ) -> float:
        """
        Calculate reservation price r(s,q,t)

        r = s - q * gamma * sigma^2 * (T - t)

        Args:
            mid_price: Current mid price
            inventory: Current inventory position
            time_remaining: Time remaining until T

        Returns:
            Reservation price
        """
        # Simplified: assume sigma estimated from recent volatility
        reservation_price = mid_price - inventory * self.gamma * time_remaining

        return reservation_price

    def calculate_optimal_spread(
        self,
        volatility: float,
        time_remaining: float
    ) -> float:
        """
        Calculate optimal spread delta

        delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

        Args:
            volatility: Price volatility
            time_remaining: Time remaining

        Returns:
            Optimal half-spread
        """
        term1 = self.gamma * (volatility ** 2) * time_remaining
        term2 = (2 / self.gamma) * np.log(1 + self.gamma / self.k)

        delta = term1 + term2
        return delta / 2

    def get_quotes(
        self,
        mid_price: float,
        inventory: int,
        volatility: float,
        time_remaining: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask prices

        Args:
            mid_price: Current mid price
            inventory: Current inventory
            volatility: Recent volatility
            time_remaining: Time remaining

        Returns:
            Tuple of (bid_price, ask_price)
        """
        r = self.calculate_reservation_price(mid_price, inventory, time_remaining)
        delta = self.calculate_optimal_spread(volatility, time_remaining)

        bid_price = r - delta
        ask_price = r + delta

        return bid_price, ask_price


class AlmgrenChriss:
    """
    Almgren-Chriss optimal execution model
    Minimizes cost + risk for order slicing
    """

    def __init__(
        self,
        lambda_: float = AC_CONFIG["lambda_"],
        eta: float = AC_CONFIG["eta"],
        sigma: float = AC_CONFIG["sigma"],
        gamma: float = AC_CONFIG["gamma"],
        T: float = AC_CONFIG["T"]
    ):
        self.lambda_ = lambda_  # Permanent impact coefficient
        self.eta = eta  # Temporary impact coefficient
        self.sigma = sigma  # Volatility
        self.gamma = gamma  # Risk aversion
        self.T = T  # Execution horizon

    def calculate_trajectory(
        self,
        total_quantity: float,
        n_intervals: int = 10
    ) -> np.ndarray:
        """
        Calculate optimal execution trajectory

        Args:
            total_quantity: Total quantity to execute
            n_intervals: Number of time intervals

        Returns:
            Array of quantities to trade at each interval
        """
        tau = self.T / n_intervals
        kappa = np.sqrt(self.lambda_ / self.eta) * np.tanh(
            0.5 * np.sqrt(self.lambda_ * self.eta) * self.T
        )

        # Time points
        t = np.linspace(0, self.T, n_intervals + 1)

        # Holdings trajectory x(t)
        holdings = total_quantity * np.sinh(
            kappa * (self.T - t)
        ) / np.sinh(kappa * self.T)

        # Trade schedule (differences in holdings)
        trades = -np.diff(holdings)

        return trades

    def calculate_expected_cost(
        self,
        total_quantity: float,
        trades: np.ndarray
    ) -> float:
        """
        Calculate expected implementation shortfall

        Args:
            total_quantity: Total quantity
            trades: Trade schedule

        Returns:
            Expected cost
        """
        # Permanent impact cost
        permanent_cost = self.lambda_ * total_quantity ** 2 / 2

        # Temporary impact cost
        temporary_cost = self.eta * np.sum(trades ** 2)

        return permanent_cost + temporary_cost


class HawkesProcess:
    """
    Hawkes process for modeling order arrival intensity
    Self-exciting point process
    """

    def __init__(
        self,
        kernel: str = HAWKES_CONFIG["kernel"],
        baseline_intensity: float = HAWKES_CONFIG["baseline_intensity"],
        decay: float = HAWKES_CONFIG["decay"]
    ):
        self.kernel = kernel
        self.mu = baseline_intensity  # Baseline intensity
        self.alpha = 0.5  # Excitation parameter (to be fitted)
        self.beta = decay  # Decay parameter

    def intensity(self, t: float, event_times: np.ndarray) -> float:
        """
        Calculate intensity at time t given history

        lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

        Args:
            t: Current time
            event_times: Array of previous event times

        Returns:
            Intensity at time t
        """
        past_events = event_times[event_times < t]

        if len(past_events) == 0:
            return self.mu

        if self.kernel == "exponential":
            excitation = self.alpha * np.sum(
                np.exp(-self.beta * (t - past_events))
            )
        else:
            excitation = 0.0

        return self.mu + excitation

    def fit(self, event_times: np.ndarray, max_iter: int = 100):
        """
        Fit Hawkes process parameters using maximum likelihood

        Args:
            event_times: Array of event timestamps
            max_iter: Maximum iterations
        """
        def neg_log_likelihood(params):
            mu, alpha, beta = params

            if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
                return np.inf

            # Calculate log-likelihood
            T = event_times[-1]
            n = len(event_times)

            # Compensator term
            compensator = mu * T

            # Log-intensity sum
            log_intensity_sum = 0

            for i, t_i in enumerate(event_times):
                past_events = event_times[:i]
                if len(past_events) > 0:
                    excitation = alpha * np.sum(
                        np.exp(-beta * (t_i - past_events))
                    )
                else:
                    excitation = 0

                intensity = mu + excitation
                log_intensity_sum += np.log(intensity)

                # Add to compensator
                if len(past_events) > 0:
                    compensator += (alpha / beta) * np.sum(
                        1 - np.exp(-beta * (T - past_events))
                    )

            log_likelihood = log_intensity_sum - compensator
            return -log_likelihood

        # Initial guess
        initial_params = [self.mu, self.alpha, self.beta]

        # Optimize
        result = optimize.minimize(
            neg_log_likelihood,
            initial_params,
            method="L-BFGS-B",
            bounds=[(0.001, 10), (0.001, 1), (0.001, 10)]
        )

        if result.success:
            self.mu, self.alpha, self.beta = result.x
            logger.info(
                f"Hawkes fitted: mu={self.mu:.4f}, "
                f"alpha={self.alpha:.4f}, beta={self.beta:.4f}"
            )
        else:
            logger.warning("Hawkes fitting failed")

    def simulate(self, T: float) -> np.ndarray:
        """
        Simulate Hawkes process up to time T

        Args:
            T: Simulation time horizon

        Returns:
            Array of event times
        """
        events = []
        t = 0

        while t < T:
            # Calculate current intensity
            lambda_current = self.intensity(t, np.array(events))

            # Upper bound for intensity
            lambda_max = lambda_current + self.alpha

            # Sample next event time (thinning algorithm)
            dt = np.random.exponential(1 / lambda_max)
            t_candidate = t + dt

            if t_candidate > T:
                break

            # Accept/reject
            lambda_candidate = self.intensity(t_candidate, np.array(events))
            if np.random.random() < lambda_candidate / lambda_max:
                events.append(t_candidate)
                t = t_candidate
            else:
                t = t_candidate

        return np.array(events)


def main():
    """Main execution function for baseline models"""
    logger.info("Testing baseline models")

    # Test Avellaneda-Stoikov
    logger.info("--- Avellaneda-Stoikov Model ---")
    as_model = AvellanedaStoikov()

    mid_price = 100.0
    inventory = 10
    volatility = 0.01

    bid, ask = as_model.get_quotes(mid_price, inventory, volatility)
    logger.info(f"Mid: {mid_price}, Inventory: {inventory}")
    logger.info(f"Optimal Bid: {bid:.4f}, Ask: {ask:.4f}, Spread: {ask-bid:.4f}")

    # Test Almgren-Chriss
    logger.info("\n--- Almgren-Chriss Model ---")
    ac_model = AlmgrenChriss()

    total_qty = 1000
    trades = ac_model.calculate_trajectory(total_qty, n_intervals=10)
    cost = ac_model.calculate_expected_cost(total_qty, trades)

    logger.info(f"Total quantity: {total_qty}")
    logger.info(f"Trade schedule: {trades}")
    logger.info(f"Expected cost: {cost:.2f}")

    # Test Hawkes Process
    logger.info("\n--- Hawkes Process ---")
    hawkes = HawkesProcess()

    # Simulate some events
    simulated_events = hawkes.simulate(T=100.0)
    logger.info(f"Simulated {len(simulated_events)} events in T=100 seconds")

    # Fit to simulated data
    if len(simulated_events) > 10:
        hawkes_fit = HawkesProcess()
        hawkes_fit.fit(simulated_events)

    logger.info("Baseline models testing complete")


if __name__ == "__main__":
    main()
