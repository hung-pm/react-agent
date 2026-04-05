from typing import Optional
from .models import User, Order

class Database:
    def __init__(self):
        self._users: dict[int, User] = {}
        self._orders: dict[int, Order] = {}
        self._user_counter = 1
        self._order_counter = 1

    def save_user(self, username: str, email: str) -> User:
        user = User(user_id=self._user_counter, username=username, email=email)
        self._users[user.user_id] = user
        self._user_counter += 1
        return user

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        return self._users.get(user_id)

    def save_order(self, user_id: int, total_amount: float) -> Order:
        order = Order(order_id=self._order_counter, user_id=user_id, total_amount=total_amount)
        self._orders[order.order_id] = order
        self._order_counter += 1
        return order
        
    def fetch_user_orders(self, user_id: int) -> list[Order]:
        return [o for o in self._orders.values() if o.user_id == user_id]

# Singleton instance
db_instance = Database()
