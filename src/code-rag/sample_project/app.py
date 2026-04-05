"""
sample_project/app.py

Một mini e-commerce backend với vài bug cố ý để test agent.
"""


class Product:
    """Sản phẩm trong cửa hàng."""

    def __init__(self, id: int, name: str, price: float, stock: int):
        self.id = id
        self.name = name
        self.price = price
        self.stock = stock

    def is_available(self, quantity: int) -> bool:
        return self.stock >= quantity


class Cart:
    """Giỏ hàng của khách."""

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.items: dict[int, int] = {}   # product_id → quantity

    def add_item(self, product_id: int, quantity: int):
        if product_id in self.items:
            self.items[product_id] += quantity
        else:
            self.items[product_id] = quantity

    def remove_item(self, product_id: int):
        # BUG: không kiểm tra product_id có trong items không → KeyError
        del self.items[product_id]

    def get_total(self, products: dict) -> float:
        total = 0
        for product_id, qty in self.items.items():
            product = products.get(product_id)
            # BUG: không kiểm tra product là None → AttributeError
            total += product.price * qty
        return total


class OrderProcessor:
    """Xử lý đơn hàng."""

    def __init__(self):
        self.orders: list[dict] = []

    def process_order(self, cart: Cart, products: dict) -> dict:
        """Xử lý đơn hàng từ giỏ hàng."""
        order_items = []

        for product_id, quantity in cart.items.items():
            product = products.get(product_id)
            if product is None:
                raise ValueError(f"Product {product_id} not found")

            # BUG: giảm stock trước khi validate → stock âm nếu validate fail
            product.stock -= quantity

            if not product.is_available(0):
                # Đây là check sai — is_available(0) luôn True
                raise ValueError(f"Insufficient stock for {product.name}")

            order_items.append({
                "product_id": product_id,
                "name": product.name,
                "quantity": quantity,
                "unit_price": product.price,
            })

        total = self.calculate_total(order_items)
        order = {
            "id": len(self.orders) + 1,
            "user_id": cart.user_id,
            "items": order_items,
            "total": total,
            "status": "confirmed",
        }
        self.orders.append(order)
        return order

    def calculate_total(self, items: list) -> float:
        """Tính tổng tiền đơn hàng với discount."""
        subtotal = sum(i["unit_price"] * i["quantity"] for i in items)
        discount = self.get_discount(subtotal)
        # BUG: trả về subtotal thay vì subtotal - discount
        return subtotal

    def get_discount(self, subtotal: float) -> float:
        """10% discount nếu đơn hàng > 500k."""
        if subtotal > 500_000:
            return subtotal * 0.1
        return 0.0

    def get_order_by_id(self, order_id: int) -> dict:
        # BUG: index sai — order_id bắt đầu từ 1 nhưng index từ 0
        return self.orders[order_id]


class UserService:
    """Quản lý user."""

    def __init__(self):
        self.users: dict[int, dict] = {}

    def create_user(self, name: str, email: str) -> dict:
        user_id = len(self.users) + 1
        user = {"id": user_id, "name": name, "email": email, "active": True}
        self.users[user_id] = user
        return user

    def get_user(self, user_id: int) -> dict:
        # BUG: không handle user không tồn tại → KeyError
        return self.users[user_id]

    def deactivate_user(self, user_id: int):
        user = self.get_user(user_id)
        user["active"] = False
