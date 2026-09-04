"""Order validation used by the service layer."""

from shop.models import Order


class ValidationError(Exception):
    """Raised when an order cannot be processed."""

    def __init__(self, message, field_name=None):
        super().__init__(message)
        self.field_name = field_name


class MissingUserError(ValidationError):
    """Raised when an order references a user that does not exist."""


def validate_order(order):
    """Validate an Order, raising ValidationError on the first problem."""
    if not order.order_id:
        raise ValidationError("order_id is required", "order_id")
    if not order.lines:
        raise ValidationError("order must have at least one line", "lines")
    for line in order.lines:
        if line.quantity <= 0:
            raise ValidationError("quantity must be positive", "quantity")
    return True


def validate_amount(amount):
    if amount <= 0:
        raise ValidationError("amount must be positive", "amount")
    return True
