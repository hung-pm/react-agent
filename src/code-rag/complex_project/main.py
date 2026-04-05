import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from api.routes import handle_register, handle_buy_items, handle_get_profile

def run_simulation():
    print("=== BẮT ĐẦU GIẢ LẬP HỆ THỐNG ===")
    
    # 1. Đăng ký người dùng
    print("\n[Client] Gọi API đăng ký...")
    res1 = handle_register({"username": "quannguyen", "email": "quan@nport.com"})
    print(f"[Server] Trả về: {res1}")
    
    if "user_id" not in res1:
        print("Lỗi đăng ký. Dừng.")
        return
        
    uid = res1["user_id"]
    
    # 2. Thanh toán đơn hàng 1
    print(f"\n[Client] User {uid} thanh toán đơn 500k...")
    res2 = handle_buy_items(uid, 500.0)
    print(f"[Server] Trả về: {res2}")
    
    # 3. Xem profile
    print(f"\n[Client] User {uid} xem thông tin profile...")
    res3 = handle_get_profile(uid)
    print(f"[Server] Trả về: {res3}")

if __name__ == "__main__":
    run_simulation()
