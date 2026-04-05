from data.database import db_instance
from utils.helpers import validate_email_format, calculate_tax, generate_transaction_id

class UserService:
    def register_user(self, username: str, email: str) -> dict:
        if not validate_email_format(email):
            return {"error": "Invalid email format"}
        
        user = db_instance.save_user(username, email)
        return {"status": "success", "user_id": user.user_id, "email": user.email}

    def get_user_profile(self, user_id: int) -> dict:
        user = db_instance.get_user_by_id(user_id)
        if not user:
            return {"error": "User not found"}
        
        # Lấy lịch sử mua hàng
        orders = db_instance.fetch_user_orders(user_id)
        return {
            "username": user.username,
            "is_active": user.is_active,
            "orders_count": len(orders)
        }

class OrderService:
    def create_checkout(self, user_id: int, subtotal: float) -> dict:
        user = db_instance.get_user_by_id(user_id)
        if not user or not user.is_active:
            return {"error": "User inactive or not found"}
            
        tax = calculate_tax(subtotal)
        total = subtotal + tax
        
        # Lưu vào DB
        order = db_instance.save_order(user_id, total)
        txn_id = generate_transaction_id()
        
        return {
            "status": "success",
            "order_id": order.order_id,
            "transaction_id": txn_id,
            "final_amount": total
        }
