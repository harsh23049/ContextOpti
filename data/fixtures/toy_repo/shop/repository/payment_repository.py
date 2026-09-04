"""Payment persistence."""

from shop.models import Payment
from shop.repository.base import BaseRepository


class PaymentRepository(BaseRepository):
    """Stores Payment records."""

    def get_payment(self, payment_id):
        return self.get(payment_id)

    def save_payment(self, payment):
        return self.put(payment.payment_id, payment)

    def record_charge(self, order_id, amount):
        payment = Payment(
            payment_id="pay-%s" % order_id,
            order_id=order_id,
            amount=amount,
        )
        return self.save_payment(payment)

    def list_for_order(self, order_id):
        return [p for p in self.all() if p.order_id == order_id]
