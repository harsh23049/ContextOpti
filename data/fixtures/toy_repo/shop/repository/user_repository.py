"""User and account persistence."""

from shop.models import Account, User
from shop.repository.base import BaseRepository


class UserRepository(BaseRepository):
    """Stores User records and the accounts they point at."""

    def __init__(self):
        super().__init__()
        self._accounts = {}

    def get_user(self, user_id):
        return self.get(user_id)

    def save_user(self, user):
        return self.put(user.user_id, user)

    def get_account(self, account_id):
        return self._accounts.get(account_id)

    def save_account(self, account):
        self._accounts[account.account_id] = account
        return account

    def seed(self):
        account = Account(account_id="acct-1", balance=10_000)
        self.save_account(account)
        self.save_user(User(user_id="u-1", email="a@example.com", account_id=account.account_id))
