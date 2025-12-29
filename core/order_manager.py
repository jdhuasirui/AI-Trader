"""
Order Manager for AI-Trader

This module implements:
- Order state machine (CREATED → PENDING → PARTIAL → FILLED/CANCELLED/REJECTED)
- Idempotency via client_order_id (UUID)
- Retry logic with exponential backoff
- Partial fill handling
- Timeout-based cancellation
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import uuid

from .data_structures import (
    OrderIntent,
    ExecutionReport,
    OrderStatus,
    OrderType,
)

logger = logging.getLogger(__name__)


class OrderEvent(str, Enum):
    """Events that trigger state transitions."""
    SUBMIT = "SUBMIT"
    ACCEPT = "ACCEPT"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    REJECT = "REJECT"
    CANCEL = "CANCEL"
    EXPIRE = "EXPIRE"
    TIMEOUT = "TIMEOUT"


@dataclass
class OrderState:
    """Internal state for an order."""
    order_intent: OrderIntent
    status: OrderStatus = OrderStatus.CREATED
    exchange_order_id: Optional[str] = None

    # Fill tracking
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    fills: List[Tuple[float, float, datetime]] = field(default_factory=list)  # (qty, price, time)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # Retry tracking
    retry_count: int = 0
    last_error: Optional[str] = None

    # Rejection info
    reject_reason: Optional[str] = None

    @property
    def remaining_qty(self) -> float:
        return self.order_intent.quantity - self.filled_qty

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        }


class OrderStateMachine:
    """
    State machine for order lifecycle management.

    Valid transitions:
    CREATED → PENDING (on SUBMIT)
    PENDING → PARTIAL (on PARTIAL_FILL)
    PENDING → FILLED (on FILL)
    PENDING → REJECTED (on REJECT)
    PENDING → CANCELLED (on CANCEL)
    PENDING → EXPIRED (on EXPIRE/TIMEOUT)
    PARTIAL → FILLED (on FILL)
    PARTIAL → CANCELLED (on CANCEL)
    """

    # Valid state transitions
    TRANSITIONS: Dict[OrderStatus, Dict[OrderEvent, OrderStatus]] = {
        OrderStatus.CREATED: {
            OrderEvent.SUBMIT: OrderStatus.PENDING,
            OrderEvent.REJECT: OrderStatus.REJECTED,
        },
        OrderStatus.PENDING: {
            OrderEvent.ACCEPT: OrderStatus.PENDING,
            OrderEvent.PARTIAL_FILL: OrderStatus.PARTIAL,
            OrderEvent.FILL: OrderStatus.FILLED,
            OrderEvent.REJECT: OrderStatus.REJECTED,
            OrderEvent.CANCEL: OrderStatus.CANCELLED,
            OrderEvent.EXPIRE: OrderStatus.EXPIRED,
            OrderEvent.TIMEOUT: OrderStatus.EXPIRED,
        },
        OrderStatus.PARTIAL: {
            OrderEvent.PARTIAL_FILL: OrderStatus.PARTIAL,
            OrderEvent.FILL: OrderStatus.FILLED,
            OrderEvent.CANCEL: OrderStatus.CANCELLED,
            OrderEvent.EXPIRE: OrderStatus.EXPIRED,
        },
    }

    def __init__(self):
        self._states: Dict[str, OrderState] = {}  # client_order_id -> state

    def create_order(self, order_intent: OrderIntent, ttl_seconds: int = 86400) -> OrderState:
        """Create a new order and register it with the state machine."""
        state = OrderState(
            order_intent=order_intent,
            status=OrderStatus.CREATED,
            expires_at=datetime.now() + timedelta(seconds=ttl_seconds),
        )
        self._states[order_intent.client_order_id] = state
        logger.debug(f"Order created: {order_intent.client_order_id}")
        return state

    def transition(
        self,
        client_order_id: str,
        event: OrderEvent,
        **kwargs,
    ) -> Tuple[bool, OrderState]:
        """
        Attempt a state transition.

        Returns (success, current_state).
        """
        if client_order_id not in self._states:
            logger.error(f"Unknown order: {client_order_id}")
            raise ValueError(f"Unknown order: {client_order_id}")

        state = self._states[client_order_id]
        current_status = state.status

        if current_status not in self.TRANSITIONS:
            logger.warning(f"No transitions from terminal state {current_status}")
            return False, state

        valid_events = self.TRANSITIONS[current_status]
        if event not in valid_events:
            logger.warning(
                f"Invalid transition: {current_status} + {event} for order {client_order_id}"
            )
            return False, state

        # Perform transition
        new_status = valid_events[event]
        state.status = new_status
        state.last_updated = datetime.now()

        # Handle event-specific updates
        if event == OrderEvent.SUBMIT:
            state.submitted_at = datetime.now()
            state.exchange_order_id = kwargs.get("exchange_order_id")

        elif event == OrderEvent.ACCEPT:
            state.exchange_order_id = kwargs.get("exchange_order_id", state.exchange_order_id)

        elif event in (OrderEvent.PARTIAL_FILL, OrderEvent.FILL):
            fill_qty = kwargs.get("fill_qty", 0)
            fill_price = kwargs.get("fill_price", 0)
            fill_time = kwargs.get("fill_time", datetime.now())

            if fill_qty > 0:
                state.fills.append((fill_qty, fill_price, fill_time))
                # Recalculate average price
                total_qty = sum(f[0] for f in state.fills)
                total_value = sum(f[0] * f[1] for f in state.fills)
                state.filled_qty = total_qty
                state.avg_fill_price = total_value / total_qty if total_qty > 0 else 0

        elif event == OrderEvent.REJECT:
            state.reject_reason = kwargs.get("reason", "Unknown rejection reason")
            state.last_error = state.reject_reason

        elif event == OrderEvent.CANCEL:
            pass  # Just status change

        elif event in (OrderEvent.EXPIRE, OrderEvent.TIMEOUT):
            pass  # Just status change

        logger.info(
            f"Order {client_order_id}: {current_status.value} -> {new_status.value} ({event.value})"
        )
        return True, state

    def get_state(self, client_order_id: str) -> Optional[OrderState]:
        """Get current state of an order."""
        return self._states.get(client_order_id)

    def get_active_orders(self) -> List[OrderState]:
        """Get all non-terminal orders."""
        return [s for s in self._states.values() if not s.is_terminal]

    def cleanup_old_orders(self, max_age_hours: int = 24) -> int:
        """Remove old terminal orders from memory."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            oid for oid, state in self._states.items()
            if state.is_terminal and state.last_updated < cutoff
        ]
        for oid in to_remove:
            del self._states[oid]
        return len(to_remove)


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior (immutable)."""
    max_retries: int = 3
    initial_delay_ms: int = 100
    max_delay_ms: int = 5000
    exponential_base: float = 2.0
    jitter_pct: float = 0.1


class OrderManager:
    """
    High-level order management with execution logic.

    Features:
    - Idempotent order submission
    - Automatic retries with exponential backoff
    - Timeout handling
    - Partial fill tracking
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        default_timeout_seconds: int = 30,
    ):
        self.state_machine = OrderStateMachine()
        self.retry_config = retry_config or RetryConfig()
        self.default_timeout_seconds = default_timeout_seconds

        # Execution callback (set by trading engine)
        self._execute_callback: Optional[Callable] = None

        # Order update callbacks
        self._on_fill_callbacks: List[Callable[[OrderState], None]] = []
        self._on_reject_callbacks: List[Callable[[OrderState], None]] = []

    def set_execute_callback(self, callback: Callable) -> None:
        """Set the callback for executing orders (broker API call)."""
        self._execute_callback = callback

    def on_fill(self, callback: Callable[[OrderState], None]) -> None:
        """Register callback for order fills."""
        self._on_fill_callbacks.append(callback)

    def on_reject(self, callback: Callable[[OrderState], None]) -> None:
        """Register callback for order rejections."""
        self._on_reject_callbacks.append(callback)

    def submit_order(
        self,
        order_intent: OrderIntent,
        timeout_seconds: Optional[int] = None,
    ) -> OrderState:
        """
        Submit an order for execution.

        This is idempotent - if an order with the same client_order_id exists,
        returns the existing order state.
        """
        # Check for existing order (idempotency)
        existing = self.state_machine.get_state(order_intent.client_order_id)
        if existing:
            logger.info(f"Order {order_intent.client_order_id} already exists, returning existing state")
            return existing

        # Create new order
        timeout = timeout_seconds or self.default_timeout_seconds
        state = self.state_machine.create_order(order_intent, ttl_seconds=timeout)

        # Attempt submission with retries
        success = self._submit_with_retry(state)

        if not success:
            self.state_machine.transition(
                order_intent.client_order_id,
                OrderEvent.REJECT,
                reason=state.last_error or "Max retries exceeded",
            )
            self._notify_reject(state)

        return state

    def _submit_with_retry(self, state: OrderState) -> bool:
        """Submit order with exponential backoff retry."""
        import random

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Call execution callback if set
                if self._execute_callback:
                    result = self._execute_callback(state.order_intent)

                    if result.get("success"):
                        self.state_machine.transition(
                            state.order_intent.client_order_id,
                            OrderEvent.SUBMIT,
                            exchange_order_id=result.get("order_id"),
                        )
                        return True
                    else:
                        state.last_error = result.get("error", "Unknown error")
                else:
                    # No callback - just mark as submitted (for simulation)
                    self.state_machine.transition(
                        state.order_intent.client_order_id,
                        OrderEvent.SUBMIT,
                    )
                    return True

            except Exception as e:
                state.last_error = str(e)
                logger.warning(
                    f"Order submission failed (attempt {attempt + 1}): {e}"
                )

            state.retry_count = attempt + 1

            # Calculate delay with exponential backoff and jitter
            if attempt < self.retry_config.max_retries:
                delay_ms = min(
                    self.retry_config.initial_delay_ms * (self.retry_config.exponential_base ** attempt),
                    self.retry_config.max_delay_ms,
                )
                jitter = delay_ms * self.retry_config.jitter_pct * random.random()
                time.sleep((delay_ms + jitter) / 1000)

        return False

    def handle_execution_report(
        self,
        client_order_id: str,
        status: str,
        filled_qty: float = 0,
        fill_price: float = 0,
        reject_reason: Optional[str] = None,
    ) -> Optional[OrderState]:
        """
        Handle an execution report from the broker.

        This is called when receiving updates about order status.
        """
        state = self.state_machine.get_state(client_order_id)
        if not state:
            logger.warning(f"Received report for unknown order: {client_order_id}")
            return None

        if status == "ACCEPTED":
            self.state_machine.transition(client_order_id, OrderEvent.ACCEPT)

        elif status == "PARTIAL_FILL":
            self.state_machine.transition(
                client_order_id,
                OrderEvent.PARTIAL_FILL,
                fill_qty=filled_qty,
                fill_price=fill_price,
            )

        elif status == "FILLED":
            # Check if there's a new fill to record
            new_fill_qty = filled_qty - state.filled_qty
            if new_fill_qty > 0:
                event = OrderEvent.PARTIAL_FILL if state.status == OrderStatus.PARTIAL else OrderEvent.FILL
                self.state_machine.transition(
                    client_order_id,
                    event,
                    fill_qty=new_fill_qty,
                    fill_price=fill_price,
                )

            # Mark as fully filled
            if state.remaining_qty <= 0.001:  # Small tolerance
                self.state_machine.transition(client_order_id, OrderEvent.FILL)
                self._notify_fill(state)

        elif status == "REJECTED":
            self.state_machine.transition(
                client_order_id,
                OrderEvent.REJECT,
                reason=reject_reason,
            )
            self._notify_reject(state)

        elif status == "CANCELLED":
            self.state_machine.transition(client_order_id, OrderEvent.CANCEL)

        elif status == "EXPIRED":
            self.state_machine.transition(client_order_id, OrderEvent.EXPIRE)

        return state

    def cancel_order(self, client_order_id: str) -> bool:
        """
        Cancel an order.

        Returns True if cancellation was successful/pending.
        """
        state = self.state_machine.get_state(client_order_id)
        if not state:
            logger.warning(f"Cannot cancel unknown order: {client_order_id}")
            return False

        if state.is_terminal:
            logger.info(f"Order {client_order_id} already in terminal state: {state.status}")
            return False

        # Try to cancel via broker
        if self._execute_callback:
            try:
                # Broker-specific cancel logic would go here
                pass
            except Exception as e:
                logger.error(f"Failed to cancel order {client_order_id}: {e}")
                return False

        success, _ = self.state_machine.transition(client_order_id, OrderEvent.CANCEL)
        return success

    def check_timeouts(self) -> List[OrderState]:
        """
        Check for and handle timed-out orders.

        Should be called periodically (e.g., every second).
        """
        now = datetime.now()
        timed_out = []

        for state in self.state_machine.get_active_orders():
            if state.expires_at and now > state.expires_at:
                # For limit orders that haven't filled, cancel them
                if state.order_intent.order_type == OrderType.LIMIT:
                    success, _ = self.state_machine.transition(
                        state.order_intent.client_order_id,
                        OrderEvent.TIMEOUT,
                    )
                    if success:
                        timed_out.append(state)
                        logger.info(
                            f"Order {state.order_intent.client_order_id} timed out"
                        )

        return timed_out

    def get_order_status(self, client_order_id: str) -> Optional[ExecutionReport]:
        """Get execution report for an order."""
        state = self.state_machine.get_state(client_order_id)
        if not state:
            return None

        return ExecutionReport(
            order_id=state.exchange_order_id or "",
            client_order_id=client_order_id,
            symbol=state.order_intent.symbol,
            side=state.order_intent.side,
            order_type=state.order_intent.order_type,
            status=state.status,
            ordered_qty=state.order_intent.quantity,
            filled_qty=state.filled_qty,
            remaining_qty=state.remaining_qty,
            avg_fill_price=state.avg_fill_price,
            reject_reason=state.reject_reason,
            created_at=state.created_at,
            updated_at=state.last_updated,
            filled_at=state.fills[-1][2] if state.fills else None,
        )

    def get_active_orders(self) -> List[ExecutionReport]:
        """Get all active (non-terminal) orders."""
        return [
            self.get_order_status(s.order_intent.client_order_id)
            for s in self.state_machine.get_active_orders()
        ]

    def _notify_fill(self, state: OrderState) -> None:
        """Notify registered callbacks of a fill."""
        for callback in self._on_fill_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

    def _notify_reject(self, state: OrderState) -> None:
        """Notify registered callbacks of a rejection."""
        for callback in self._on_reject_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Reject callback error: {e}")

    def cleanup(self, max_age_hours: int = 24) -> int:
        """Clean up old completed orders."""
        return self.state_machine.cleanup_old_orders(max_age_hours)
