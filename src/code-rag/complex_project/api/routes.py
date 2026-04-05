from core.services import UserService, OrderService

user_service = UserService()
order_service = OrderService()

def handle_register(request_data: dict) -> dict:
    """API giả lập đăng ký user"""
    username = request_data.get("username")
    email = request_data.get("email")
    if not username or not email:
        return {"error": "Missing username or email"}
        
    return user_service.register_user(username, email)

def handle_get_profile(user_id: int) -> dict:
    """API giả lập lấy thông tin profile"""
    return user_service.get_user_profile(user_id)

def handle_buy_items(user_id: int, items_value: float) -> dict:
    """API giả lập thanh toán giỏ hàng"""
    return order_service.create_checkout(user_id, items_value)
