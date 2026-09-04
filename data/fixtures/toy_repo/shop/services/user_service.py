"""User-facing business logic.

Carries the data-flow chain used as a fixture example:

    user_id -> get_user() -> user.account_id -> get_account() -> account.balance
"""

from shop.repository.user_repository import UserRepository
from shop.utils.validation import MissingUserError


class UserService:
    def __init__(self, users=None):
        self.users = users or UserRepository()

    def get_user(self, user_id):
        user = self.users.get_user(user_id)
        if user is None:
            raise MissingUserError("no such user: %s" % user_id, "user_id")
        return user

    def get_account_for_user(self, user_id):
        user = self.get_user(user_id)
        return self.users.get_account(user.account_id)

    def get_balance(self, user_id):
        account = self.get_account_for_user(user_id)
        if account is None:
            return 0
        return account.balance

    def can_afford(self, user_id, amount):
        account = self.get_account_for_user(user_id)
        if account is None:
            return False
        return account.can_cover(amount)
