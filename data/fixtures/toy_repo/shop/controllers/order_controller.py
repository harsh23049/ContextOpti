"""Thin controller over OrderService."""

from shop.services.order_service import OrderService
from shop.utils.validation import ValidationError


class OrderController:
    def __init__(self, service=None):
        self.service = service or OrderService()

    def create(self, request):
        try:
            order = self.service.create_order(request["user_id"], request["lines"])
        except ValidationError as exc:
            return {"status": 400, "error": str(exc)}
        return {"status": 201, "order_id": order.order_id}

    def checkout(self, order_id):
        order = self.service.checkout(order_id)
        return {"status": 200, "order_status": order.status}

    def list_orders(self, user_id):
        orders = self.service.history(user_id)
        return {"status": 200, "orders": [o.order_id for o in orders]}
