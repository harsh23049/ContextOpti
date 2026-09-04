"""Domain models for the toy shop."""

from dataclasses import dataclass, field


@dataclass
class User:
    user_id: str
    email: str
    account_id: str


@dataclass
class Account:
    account_id: str
    balance: int
    currency: str = "USD"

    def can_cover(self, amount):
        return self.balance >= amount


@dataclass
class OrderLine:
    sku: str
    quantity: int
    unit_price: int

    def subtotal(self):
        return self.quantity * self.unit_price


@dataclass
class Order:
    order_id: str
    user_id: str
    lines: list = field(default_factory=list)
    status: str = "pending"

    def total(self):
        return sum(line.subtotal() for line in self.lines)


@dataclass
class Payment:
    payment_id: str
    order_id: str
    amount: int
    status: str = "authorized"
