class User:
    def __init__(self, user_id: int, username: str, email: str, is_active: bool = True):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.is_active = is_active

class Order:
    def __init__(self, order_id: int, user_id: int, total_amount: float, status: str = "pending"):
        self.order_id = order_id
        self.user_id = user_id
        self.total_amount = total_amount
        self.status = status
