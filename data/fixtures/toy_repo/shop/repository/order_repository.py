"""Order persistence."""

from shop.repository.base import BaseRepository


class OrderRepository(BaseRepository):
    """Stores Order records."""

    def get_order(self, order_id):
        return self.get(order_id)

    def save_order(self, order):
        return self.put(order.order_id, order)

    def list_for_user(self, user_id):
        return [order for order in self.all() if order.user_id == user_id]

    def mark_status(self, order_id, status):
        order = self.get_order(order_id)
        if order is not None:
            order.status = status
        return order
