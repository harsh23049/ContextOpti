"""Payment business logic."""

from shop.repository.payment_repository import PaymentRepository
from shop.services.user_service import UserService
from shop.utils.validation import ValidationError, validate_amount


class PaymentService:
    def __init__(self, payments=None, users=None):
        self.payments = payments or PaymentRepository()
        self.users = users or UserService()

    def charge(self, user_id, order_id, amount):
        validate_amount(amount)
        if not self.users.can_afford(user_id, amount):
            raise ValidationError("insufficient funds", "amount")
        return self.payments.record_charge(order_id, amount)

    def refund(self, payment_id):
        payment = self.payments.get_payment(payment_id)
        if payment is None:
            raise ValidationError("unknown payment", "payment_id")
        payment.status = "refunded"
        return self.payments.save_payment(payment)

    def total_charged(self, order_id):
        return sum(p.amount for p in self.payments.list_for_order(order_id))
