"""
Simulation utilities for exchange replay and ABM
Helper functions for order event generation and queue tracking
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "B"
    SELL = "S"


class OrderType(Enum):
    """Order type enumeration"""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    CANCEL = "CANCEL"


class TraderType(Enum):
    """Trader type for ABM simulation"""
    INFORMED = "INFORMED"
    NOISE = "NOISE"
    MARKET_MAKER = "MARKET_MAKER"


@dataclass
class Order:
    """Order data structure"""
    order_id: str
    instrument_id: str
    side: OrderSide
    order_type: OrderType
    price: Optional[float]
    quantity: float
    timestamp: pd.Timestamp
    trader_id: Optional[str] = None
    trader_type: Optional[TraderType] = None


@dataclass
class Fill:
    """Fill/execution data structure"""
    fill_id: str
    order_id: str
    instrument_id: str
    side: OrderSide
    price: float
    quantity: float
    timestamp: pd.Timestamp
    is_aggressor: bool


@dataclass
class QueuePosition:
    """Queue position tracking"""
    order_id: str
    price: float
    side: OrderSide
    quantity: float
    position_in_queue: int
    total_ahead: float
    timestamp: pd.Timestamp


def generate_order_id(prefix: str = "ORD") -> str:
    """
    Generate unique order ID

    Args:
        prefix: Order ID prefix

    Returns:
        Unique order ID string
    """
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def generate_fill_id(prefix: str = "FILL") -> str:
    """
    Generate unique fill ID

    Args:
        prefix: Fill ID prefix

    Returns:
        Unique fill ID string
    """
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def round_to_tick(price: float, tick_size: float) -> float:
    """
    Round price to nearest tick size

    Args:
        price: Raw price
        tick_size: Minimum tick size

    Returns:
        Rounded price
    """
    return round(price / tick_size) * tick_size


def round_to_lot(quantity: float, lot_size: float) -> float:
    """
    Round quantity to nearest lot size

    Args:
        quantity: Raw quantity
        lot_size: Minimum lot size

    Returns:
        Rounded quantity
    """
    return round(quantity / lot_size) * lot_size


def calculate_queue_position(
    order_price: float,
    order_side: OrderSide,
    order_timestamp: pd.Timestamp,
    book_state: pd.DataFrame
) -> Tuple[int, float]:
    """
    Calculate position in queue for a limit order

    Args:
        order_price: Order price
        order_side: Order side
        order_timestamp: Order timestamp
        book_state: Current order book state

    Returns:
        Tuple of (position_in_queue, total_volume_ahead)
    """
    if order_side == OrderSide.BUY:
        same_price_orders = book_state[
            (book_state["side"] == "B") &
            (book_state["price"] == order_price) &
            (book_state["timestamp"] < order_timestamp)
        ]
    else:
        same_price_orders = book_state[
            (book_state["side"] == "S") &
            (book_state["price"] == order_price) &
            (book_state["timestamp"] < order_timestamp)
        ]

    position = len(same_price_orders) + 1
    total_ahead = same_price_orders["quantity"].sum()

    return position, total_ahead


def estimate_fill_probability(
    queue_position: int,
    total_ahead: float,
    recent_volume: float,
    time_window_seconds: float
) -> float:
    """
    Estimate probability of order being filled

    Args:
        queue_position: Position in queue
        total_ahead: Total volume ahead in queue
        recent_volume: Recent volume at this price level
        time_window_seconds: Time window for volume measurement

    Returns:
        Fill probability (0 to 1)
    """
    if recent_volume == 0 or time_window_seconds == 0:
        return 0.0

    # Volume rate per second
    volume_rate = recent_volume / time_window_seconds

    # Simple exponential decay model
    if volume_rate > 0:
        lambda_fill = volume_rate / max(total_ahead, 1)
        # Probability decreases with queue position
        prob = 1 - np.exp(-lambda_fill * (1 / max(queue_position, 1)))
    else:
        prob = 0.0

    return min(prob, 1.0)


def generate_poisson_arrivals(
    intensity: float,
    duration_seconds: float,
    start_time: pd.Timestamp
) -> List[pd.Timestamp]:
    """
    Generate Poisson process arrivals

    Args:
        intensity: Arrival rate (events per second)
        duration_seconds: Simulation duration
        start_time: Start timestamp

    Returns:
        List of arrival timestamps
    """
    n_events = np.random.poisson(intensity * duration_seconds)
    inter_arrival_times = np.random.exponential(1 / intensity, n_events)

    timestamps = []
    current_time = start_time

    for dt in inter_arrival_times:
        current_time = current_time + pd.Timedelta(seconds=dt)
        if (current_time - start_time).total_seconds() > duration_seconds:
            break
        timestamps.append(current_time)

    return timestamps


def generate_random_walk_price(
    initial_price: float,
    n_steps: int,
    volatility: float,
    drift: float = 0.0,
    tick_size: float = 0.01
) -> np.ndarray:
    """
    Generate random walk price path

    Args:
        initial_price: Starting price
        n_steps: Number of steps
        volatility: Price volatility (standard deviation)
        drift: Price drift (mean return)
        tick_size: Minimum tick size

    Returns:
        Array of prices
    """
    returns = np.random.normal(drift, volatility, n_steps)
    price_path = initial_price * np.exp(np.cumsum(returns))

    # Round to tick size
    price_path = np.vectorize(lambda p: round_to_tick(p, tick_size))(price_path)

    return price_path


def generate_informed_order_flow(
    current_price: float,
    true_value: float,
    intensity: float,
    quantity_mean: float,
    quantity_std: float,
    tick_size: float,
    lot_size: float
) -> Optional[Order]:
    """
    Generate order from informed trader based on price discrepancy

    Args:
        current_price: Current market price
        true_value: Informed trader's estimate of true value
        intensity: Order intensity
        quantity_mean: Mean order quantity
        quantity_std: Std of order quantity
        tick_size: Minimum tick size
        lot_size: Minimum lot size

    Returns:
        Order if generated, None otherwise
    """
    if np.random.random() > intensity:
        return None

    # Informed traders buy if price < value, sell if price > value
    if true_value > current_price:
        side = OrderSide.BUY
        # Limit order slightly above current price
        price = round_to_tick(current_price + tick_size, tick_size)
    elif true_value < current_price:
        side = OrderSide.SELL
        price = round_to_tick(current_price - tick_size, tick_size)
    else:
        return None

    quantity = max(np.random.normal(quantity_mean, quantity_std), lot_size)
    quantity = round_to_lot(quantity, lot_size)

    return Order(
        order_id=generate_order_id("INF"),
        instrument_id="SIM_ASSET",
        side=side,
        order_type=OrderType.LIMIT,
        price=price,
        quantity=quantity,
        timestamp=pd.Timestamp.now(),
        trader_type=TraderType.INFORMED
    )


def generate_noise_order(
    current_bid: float,
    current_ask: float,
    intensity: float,
    quantity_mean: float,
    quantity_std: float,
    tick_size: float,
    lot_size: float
) -> Optional[Order]:
    """
    Generate order from noise trader (random)

    Args:
        current_bid: Current best bid
        current_ask: Current best ask
        intensity: Order intensity
        quantity_mean: Mean order quantity
        quantity_std: Std of order quantity
        tick_size: Minimum tick size
        lot_size: Minimum lot size

    Returns:
        Order if generated, None otherwise
    """
    if np.random.random() > intensity:
        return None

    # Random side
    side = OrderSide.BUY if np.random.random() < 0.5 else OrderSide.SELL

    # Random price near BBO
    if side == OrderSide.BUY:
        price = round_to_tick(current_bid + np.random.randint(-2, 3) * tick_size, tick_size)
    else:
        price = round_to_tick(current_ask + np.random.randint(-2, 3) * tick_size, tick_size)

    quantity = max(np.random.normal(quantity_mean, quantity_std), lot_size)
    quantity = round_to_lot(quantity, lot_size)

    return Order(
        order_id=generate_order_id("NOISE"),
        instrument_id="SIM_ASSET",
        side=side,
        order_type=OrderType.LIMIT,
        price=price,
        quantity=quantity,
        timestamp=pd.Timestamp.now(),
        trader_type=TraderType.NOISE
    )


def generate_market_maker_quotes(
    mid_price: float,
    spread: float,
    inventory: int,
    max_inventory: int,
    risk_aversion: float,
    quantity: float,
    tick_size: float,
    lot_size: float
) -> Tuple[Optional[Order], Optional[Order]]:
    """
    Generate market maker bid and ask quotes with inventory management

    Args:
        mid_price: Current mid price
        spread: Desired spread
        inventory: Current inventory position
        max_inventory: Maximum allowed inventory
        risk_aversion: Risk aversion parameter
        quantity: Quote size
        tick_size: Minimum tick size
        lot_size: Minimum lot size

    Returns:
        Tuple of (bid_order, ask_order)
    """
    # Adjust spread based on inventory (Avellaneda-Stoikov style)
    inventory_skew = risk_aversion * (inventory / max_inventory)

    bid_price = round_to_tick(mid_price - spread / 2 - inventory_skew, tick_size)
    ask_price = round_to_tick(mid_price + spread / 2 - inventory_skew, tick_size)

    quantity = round_to_lot(quantity, lot_size)

    bid_order = Order(
        order_id=generate_order_id("MM_BID"),
        instrument_id="SIM_ASSET",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=bid_price,
        quantity=quantity,
        timestamp=pd.Timestamp.now(),
        trader_type=TraderType.MARKET_MAKER
    )

    ask_order = Order(
        order_id=generate_order_id("MM_ASK"),
        instrument_id="SIM_ASSET",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=ask_price,
        quantity=quantity,
        timestamp=pd.Timestamp.now(),
        trader_type=TraderType.MARKET_MAKER
    )

    return bid_order, ask_order


def calculate_impact_cost(
    quantity: float,
    side: OrderSide,
    book_liquidity: float,
    permanent_impact_coef: float,
    temporary_impact_coef: float
) -> float:
    """
    Calculate price impact cost

    Args:
        quantity: Order quantity
        side: Order side
        book_liquidity: Current book liquidity
        permanent_impact_coef: Permanent impact coefficient
        temporary_impact_coef: Temporary impact coefficient

    Returns:
        Impact cost in price units
    """
    # Simple square-root impact model
    permanent_impact = permanent_impact_coef * np.sqrt(quantity)
    temporary_impact = temporary_impact_coef * quantity / book_liquidity

    total_impact = permanent_impact + temporary_impact

    return total_impact if side == OrderSide.BUY else -total_impact


def simulate_partial_fill(
    order_quantity: float,
    available_liquidity: float,
    fill_probability: float
) -> float:
    """
    Simulate partial fill of order

    Args:
        order_quantity: Original order quantity
        available_liquidity: Available liquidity at price level
        fill_probability: Probability of getting filled

    Returns:
        Filled quantity
    """
    if np.random.random() > fill_probability:
        return 0.0

    max_fill = min(order_quantity, available_liquidity)
    # Random partial fill between 0 and max
    fill_pct = np.random.uniform(0.5, 1.0)  # At least 50% if filled
    filled_quantity = max_fill * fill_pct

    return filled_quantity


def check_self_trade(
    order: Order,
    book_orders: List[Order],
    trader_id: str
) -> bool:
    """
    Check if order would result in self-trade

    Args:
        order: Incoming order
        book_orders: Orders currently in book
        trader_id: Trader ID to check

    Returns:
        True if self-trade would occur
    """
    opposite_side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY

    for book_order in book_orders:
        if (book_order.trader_id == trader_id and
            book_order.side == opposite_side):
            # Check if prices would match
            if order.side == OrderSide.BUY and order.price >= book_order.price:
                return True
            elif order.side == OrderSide.SELL and order.price <= book_order.price:
                return True

    return False


def validate_order(
    order: Order,
    tick_size: float,
    lot_size: float,
    max_order_size: float,
    current_price: float,
    price_collar_pct: float
) -> Tuple[bool, str]:
    """
    Validate order against exchange rules

    Args:
        order: Order to validate
        tick_size: Minimum tick size
        lot_size: Minimum lot size
        max_order_size: Maximum order size
        current_price: Current market price
        price_collar_pct: Price collar percentage

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check quantity
    if order.quantity <= 0:
        return False, "Invalid quantity: must be positive"

    if order.quantity > max_order_size:
        return False, f"Order size exceeds maximum: {max_order_size}"

    # Check lot size
    if order.quantity % lot_size != 0:
        return False, f"Quantity must be multiple of lot size: {lot_size}"

    # Check price (for limit orders)
    if order.order_type == OrderType.LIMIT:
        if order.price <= 0:
            return False, "Invalid price: must be positive"

        # Check tick size
        if round(order.price / tick_size) * tick_size != order.price:
            return False, f"Price must be multiple of tick size: {tick_size}"

        # Check price collar
        collar_bound = current_price * price_collar_pct / 100
        if abs(order.price - current_price) > collar_bound:
            return False, f"Price outside collar: {price_collar_pct}%"

    return True, ""
