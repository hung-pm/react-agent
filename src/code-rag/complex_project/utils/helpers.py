import random
from typing import Any

def generate_transaction_id() -> str:
    """Tạo mã giao dịch tự động"""
    return f"TXN-{random.randint(10000, 99999)}"

def validate_email_format(email: str) -> bool:
    """Hàm kiểm tra email cơ bản"""
    if "@" not in email or "." not in email:
        return False
    return True

def calculate_tax(amount: float, tax_rate: float = 0.08) -> float:
    """Tính thuế VAT cho đơn hàng"""
    return round(amount * tax_rate, 2)
