"""Order orchestration: the deepest cross-file entry point in the fixture."""

from shop.models import Order
from shop.repository.order_repository import OrderRepository
from shop.services.payment_service import PaymentService
from shop.services.user_service import UserService
from shop.utils.validation import validate_order


class OrderService:
    def __init__(self, orders=None, payments=None, users=None):
        self.orders = orders or OrderRepository()
        self.payments = payments or PaymentService()
        self.users = users or UserService()

    def create_order(self, user_id, lines):
        self.users.get_user(user_id)
        order = Order(order_id="ord-%d" % (self.orders.count() + 1), user_id=user_id, lines=lines)
        validate_order(order)
        return self.orders.save_order(order)

    def checkout(self, order_id):
        order = self.orders.get_order(order_id)
        validate_order(order)
        amount = order.total()
        self.payments.charge(order.user_id, order.order_id, amount)
        return self.orders.mark_status(order_id, "paid")

    def cancel(self, order_id):
        return self.orders.mark_status(order_id, "cancelled")

    def history(self, user_id):
        return self.orders.list_for_user(user_id)
