"""
Unit tests for OrderStateMachine transitions.

Tests:
- Valid state transitions
- Invalid state transitions
- Order lifecycle scenarios
- Edge cases
"""

import unittest
from datetime import datetime, timedelta

from core import (
    OrderStateMachine, OrderState, OrderStatus, OrderEvent,
    OrderIntent, OrderType, RetryConfig,
)


class TestOrderStateMachineTransitions(unittest.TestCase):
    """Test OrderStateMachine state transitions."""

    def setUp(self):
        self.state_machine = OrderStateMachine()
        self.order_intent = OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type=OrderType.MARKET,
        )

    def test_create_order(self):
        """Test order creation initializes to CREATED state."""
        state = self.state_machine.create_order(self.order_intent)
        self.assertEqual(state.status, OrderStatus.CREATED)
        self.assertIsNotNone(state.expires_at)

    def test_valid_transition_created_to_pending(self):
        """CREATED -> PENDING on SUBMIT is valid."""
        state = self.state_machine.create_order(self.order_intent)
        success, new_state = self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.SUBMIT
        )
        self.assertTrue(success)
        self.assertEqual(new_state.status, OrderStatus.PENDING)

    def test_valid_transition_pending_to_filled(self):
        """PENDING -> FILLED on FILL is valid."""
        state = self.state_machine.create_order(self.order_intent)
        self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.SUBMIT
        )
        success, new_state = self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.FILL
        )
        self.assertTrue(success)
        self.assertEqual(new_state.status, OrderStatus.FILLED)

    def test_valid_transition_pending_to_partial(self):
        """PENDING -> PARTIAL on PARTIAL_FILL is valid."""
        state = self.state_machine.create_order(self.order_intent)
        self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.SUBMIT
        )
        success, new_state = self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.PARTIAL_FILL,
            filled_qty=5
        )
        self.assertTrue(success)
        self.assertEqual(new_state.status, OrderStatus.PARTIAL)

    def test_valid_transition_partial_to_filled(self):
        """PARTIAL -> FILLED on FILL is valid."""
        state = self.state_machine.create_order(self.order_intent)
        self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.SUBMIT
        )
        self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.PARTIAL_FILL,
            filled_qty=5
        )
        success, new_state = self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.FILL
        )
        self.assertTrue(success)
        self.assertEqual(new_state.status, OrderStatus.FILLED)

    def test_valid_transition_pending_to_cancelled(self):
        """PENDING -> CANCELLED on CANCEL is valid."""
        state = self.state_machine.create_order(self.order_intent)
        self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.SUBMIT
        )
        success, new_state = self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.CANCEL
        )
        self.assertTrue(success)
        self.assertEqual(new_state.status, OrderStatus.CANCELLED)

    def test_valid_transition_pending_to_rejected(self):
        """PENDING -> REJECTED on REJECT is valid."""
        state = self.state_machine.create_order(self.order_intent)
        self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.SUBMIT
        )
        success, new_state = self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.REJECT,
            reason="Insufficient funds"
        )
        self.assertTrue(success)
        self.assertEqual(new_state.status, OrderStatus.REJECTED)

    def test_valid_transition_pending_to_expired(self):
        """PENDING -> EXPIRED on TIMEOUT is valid."""
        state = self.state_machine.create_order(self.order_intent)
        self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.SUBMIT
        )
        success, new_state = self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.TIMEOUT
        )
        self.assertTrue(success)
        self.assertEqual(new_state.status, OrderStatus.EXPIRED)

    def test_invalid_transition_from_terminal_state(self):
        """Cannot transition from terminal states (FILLED, REJECTED, etc.)."""
        state = self.state_machine.create_order(self.order_intent)
        self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.SUBMIT
        )
        self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.FILL
        )

        # Try to transition from FILLED (terminal state)
        success, current_state = self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.CANCEL
        )
        self.assertFalse(success)
        self.assertEqual(current_state.status, OrderStatus.FILLED)

    def test_invalid_transition_wrong_event(self):
        """Invalid event for current state should fail."""
        state = self.state_machine.create_order(self.order_intent)
        # Try to FILL directly from CREATED (should fail)
        success, current_state = self.state_machine.transition(
            self.order_intent.client_order_id, OrderEvent.FILL
        )
        self.assertFalse(success)
        self.assertEqual(current_state.status, OrderStatus.CREATED)

    def test_unknown_order_raises_error(self):
        """Transitioning unknown order should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.state_machine.transition("unknown_order_id", OrderEvent.SUBMIT)
        self.assertIn("Unknown order", str(ctx.exception))

    def test_get_order_state(self):
        """Should be able to retrieve order state by ID."""
        state = self.state_machine.create_order(self.order_intent)
        retrieved = self.state_machine.get_state(self.order_intent.client_order_id)
        self.assertEqual(retrieved.status, OrderStatus.CREATED)


class TestOrderStateExpiry(unittest.TestCase):
    """Test order expiration handling."""

    def setUp(self):
        self.state_machine = OrderStateMachine()

    def test_order_ttl(self):
        """Order should have correct expiry time based on TTL."""
        order_intent = OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type=OrderType.MARKET,
        )
        ttl = 3600  # 1 hour
        state = self.state_machine.create_order(order_intent, ttl_seconds=ttl)

        expected_expiry = datetime.now() + timedelta(seconds=ttl)
        # Allow 1 second tolerance
        diff = abs((state.expires_at - expected_expiry).total_seconds())
        self.assertLess(diff, 1)

    def test_cleanup_old_orders(self):
        """Cleanup should remove old terminal orders."""
        # Create an order and move it to terminal state
        order_intent = OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type=OrderType.MARKET,
        )
        state = self.state_machine.create_order(order_intent)

        # Move to terminal state
        self.state_machine.transition(order_intent.client_order_id, OrderEvent.SUBMIT)
        self.state_machine.transition(order_intent.client_order_id, OrderEvent.FILL)

        # Force old timestamp
        state = self.state_machine.get_state(order_intent.client_order_id)
        state.last_updated = datetime.now() - timedelta(hours=25)

        # Cleanup should remove orders older than 24 hours
        removed = self.state_machine.cleanup_old_orders(max_age_hours=24)
        self.assertEqual(removed, 1)


class TestRetryConfigValidation(unittest.TestCase):
    """Test RetryConfig is immutable."""

    def test_retry_config_frozen(self):
        """RetryConfig should be frozen (immutable)."""
        config = RetryConfig()
        with self.assertRaises(Exception):
            config.max_retries = 10


class TestOrderLifecycleScenarios(unittest.TestCase):
    """Test complete order lifecycle scenarios."""

    def setUp(self):
        self.state_machine = OrderStateMachine()

    def test_successful_market_order_lifecycle(self):
        """Test a successful market order from creation to fill."""
        order = OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type=OrderType.MARKET,
        )

        # Create
        state = self.state_machine.create_order(order)
        self.assertEqual(state.status, OrderStatus.CREATED)

        # Submit
        success, state = self.state_machine.transition(
            order.client_order_id, OrderEvent.SUBMIT
        )
        self.assertTrue(success)
        self.assertEqual(state.status, OrderStatus.PENDING)

        # Accept (optional)
        success, state = self.state_machine.transition(
            order.client_order_id, OrderEvent.ACCEPT
        )
        self.assertTrue(success)
        self.assertEqual(state.status, OrderStatus.PENDING)

        # Fill
        success, state = self.state_machine.transition(
            order.client_order_id, OrderEvent.FILL,
            filled_qty=10,
            avg_price=150.50
        )
        self.assertTrue(success)
        self.assertEqual(state.status, OrderStatus.FILLED)

    def test_partial_fill_lifecycle(self):
        """Test order with partial fills."""
        order = OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        )

        state = self.state_machine.create_order(order)
        self.state_machine.transition(order.client_order_id, OrderEvent.SUBMIT)

        # First partial fill
        success, state = self.state_machine.transition(
            order.client_order_id, OrderEvent.PARTIAL_FILL,
            filled_qty=30
        )
        self.assertTrue(success)
        self.assertEqual(state.status, OrderStatus.PARTIAL)

        # Second partial fill
        success, state = self.state_machine.transition(
            order.client_order_id, OrderEvent.PARTIAL_FILL,
            filled_qty=50
        )
        self.assertTrue(success)
        self.assertEqual(state.status, OrderStatus.PARTIAL)

        # Final fill
        success, state = self.state_machine.transition(
            order.client_order_id, OrderEvent.FILL,
            filled_qty=100
        )
        self.assertTrue(success)
        self.assertEqual(state.status, OrderStatus.FILLED)

    def test_cancelled_order_lifecycle(self):
        """Test order that gets cancelled."""
        order = OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=145.0,
        )

        state = self.state_machine.create_order(order)
        self.state_machine.transition(order.client_order_id, OrderEvent.SUBMIT)

        # Cancel
        success, state = self.state_machine.transition(
            order.client_order_id, OrderEvent.CANCEL
        )
        self.assertTrue(success)
        self.assertEqual(state.status, OrderStatus.CANCELLED)


if __name__ == "__main__":
    unittest.main()
