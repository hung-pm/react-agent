# 📊 THỐNG KÊ CHROMADB

- Tổng số code chunks: **19**

## 🔍 Chi Tiết Bản Ghi

### ID: `a46b52f2753128df`
- **Type:** class
- **Name:** `Product`
- **File:** `app.py`
- **Lines:** 8 -> 18

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "language": "python",
  "end_line": 18,
  "file_path": "app.py",
  "chunk_type": "class",
  "name": "Product",
  "parent_class": "",
  "signature": "class Product:",
  "start_line": 8
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: class  Name: Product
Signature: class Product:
Description: Sản phẩm trong cửa hàng.
class Product:
    """Sản phẩm trong cửa hàng."""

    def __init__(self, id: int, name: str, price: float, stock: int):
        self.id = id
        self.name = name
        self.price = price
        self.stock = stock

    def is_available(self, quantity: int) -> bool:
        return self.stock >= quantity
```

---

### ID: `bd0d9a8572d985d1`
- **Type:** function
- **Name:** `__init__`
- **File:** `app.py`
- **Lines:** 11 -> 15

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "chunk_type": "function",
  "parent_class": "Product",
  "end_line": 15,
  "start_line": 11,
  "file_path": "app.py",
  "name": "__init__",
  "signature": "def __init__(self, id: int, name: str, price: float, stock: int):",
  "language": "python"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: __init__
Class: Product
Signature: def __init__(self, id: int, name: str, price: float, stock: int):
def __init__(self, id: int, name: str, price: float, stock: int):
        self.id = id
        self.name = name
        self.price = price
        self.stock = stock
```

---

### ID: `bd9baba45723410b`
- **Type:** function
- **Name:** `is_available`
- **File:** `app.py`
- **Lines:** 17 -> 18

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "file_path": "app.py",
  "chunk_type": "function",
  "parent_class": "Product",
  "name": "is_available",
  "language": "python",
  "signature": "def is_available(self, quantity: int) -> bool:",
  "end_line": 18,
  "start_line": 17
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: is_available
Class: Product
Signature: def is_available(self, quantity: int) -> bool:
def is_available(self, quantity: int) -> bool:
        return self.stock >= quantity
```

---

### ID: `33896d6ddd6b5214`
- **Type:** class
- **Name:** `Cart`
- **File:** `app.py`
- **Lines:** 21 -> 44

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "parent_class": "",
  "language": "python",
  "file_path": "app.py",
  "chunk_type": "class",
  "end_line": 44,
  "start_line": 21,
  "signature": "class Cart:",
  "name": "Cart"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: class  Name: Cart
Signature: class Cart:
Description: Giỏ hàng của khách.
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
```

---

### ID: `14710c2935f7ff17`
- **Type:** function
- **Name:** `__init__`
- **File:** `app.py`
- **Lines:** 24 -> 26

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "file_path": "app.py",
  "start_line": 24,
  "chunk_type": "function",
  "signature": "def __init__(self, user_id: int):",
  "language": "python",
  "end_line": 26,
  "name": "__init__",
  "parent_class": "Cart"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: __init__
Class: Cart
Signature: def __init__(self, user_id: int):
def __init__(self, user_id: int):
        self.user_id = user_id
        self.items: dict[int, int] = {}   # product_id → quantity
```

---

### ID: `29f58f96b4615045`
- **Type:** function
- **Name:** `add_item`
- **File:** `app.py`
- **Lines:** 28 -> 32

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "end_line": 32,
  "parent_class": "Cart",
  "language": "python",
  "start_line": 28,
  "file_path": "app.py",
  "chunk_type": "function",
  "signature": "def add_item(self, product_id: int, quantity: int):",
  "name": "add_item"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: add_item
Class: Cart
Signature: def add_item(self, product_id: int, quantity: int):
def add_item(self, product_id: int, quantity: int):
        if product_id in self.items:
            self.items[product_id] += quantity
        else:
            self.items[product_id] = quantity
```

---

### ID: `c1714b55be99bfdf`
- **Type:** function
- **Name:** `remove_item`
- **File:** `app.py`
- **Lines:** 34 -> 36

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "signature": "def remove_item(self, product_id: int):",
  "parent_class": "Cart",
  "end_line": 36,
  "language": "python",
  "start_line": 34,
  "file_path": "app.py",
  "name": "remove_item",
  "chunk_type": "function"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: remove_item
Class: Cart
Signature: def remove_item(self, product_id: int):
def remove_item(self, product_id: int):
        # BUG: không kiểm tra product_id có trong items không → KeyError
        del self.items[product_id]
```

---

### ID: `3de5b5ef62acf8d2`
- **Type:** function
- **Name:** `get_total`
- **File:** `app.py`
- **Lines:** 38 -> 44

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "end_line": 44,
  "name": "get_total",
  "signature": "def get_total(self, products: dict) -> float:",
  "chunk_type": "function",
  "start_line": 38,
  "language": "python",
  "parent_class": "Cart",
  "file_path": "app.py"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: get_total
Class: Cart
Signature: def get_total(self, products: dict) -> float:
def get_total(self, products: dict) -> float:
        total = 0
        for product_id, qty in self.items.items():
            product = products.get(product_id)
            # BUG: không kiểm tra product là None → AttributeError
            total += product.price * qty
        return total
```

---

### ID: `d000d10f39efba6c`
- **Type:** class
- **Name:** `OrderProcessor`
- **File:** `app.py`
- **Lines:** 47 -> 102

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "signature": "class OrderProcessor:",
  "chunk_type": "class",
  "parent_class": "",
  "end_line": 102,
  "name": "OrderProcessor",
  "start_line": 47,
  "file_path": "app.py",
  "language": "python"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: class  Name: OrderProcessor
Signature: class OrderProcessor:
Description: Xử lý đơn hàng.
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
```

---

### ID: `8cbec1f948ca1dfa`
- **Type:** function
- **Name:** `__init__`
- **File:** `app.py`
- **Lines:** 50 -> 51

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "parent_class": "OrderProcessor",
  "file_path": "app.py",
  "chunk_type": "function",
  "name": "__init__",
  "start_line": 50,
  "language": "python",
  "signature": "def __init__(self):",
  "end_line": 51
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: __init__
Class: OrderProcessor
Signature: def __init__(self):
def __init__(self):
        self.orders: list[dict] = []
```

---

### ID: `9474ce797aba6ea8`
- **Type:** function
- **Name:** `process_order`
- **File:** `app.py`
- **Lines:** 53 -> 85

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "signature": "def process_order(self, cart: Cart, products: dict) -> dict:",
  "end_line": 85,
  "start_line": 53,
  "chunk_type": "function",
  "name": "process_order",
  "language": "python",
  "file_path": "app.py",
  "parent_class": "OrderProcessor"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: process_order
Class: OrderProcessor
Signature: def process_order(self, cart: Cart, products: dict) -> dict:
Description: Xử lý đơn hàng từ giỏ hàng.
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
```

---

### ID: `31303c7e6930815b`
- **Type:** function
- **Name:** `calculate_total`
- **File:** `app.py`
- **Lines:** 87 -> 92

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "start_line": 87,
  "chunk_type": "function",
  "parent_class": "OrderProcessor",
  "end_line": 92,
  "language": "python",
  "name": "calculate_total",
  "file_path": "app.py",
  "signature": "def calculate_total(self, items: list) -> float:"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: calculate_total
Class: OrderProcessor
Signature: def calculate_total(self, items: list) -> float:
Description: Tính tổng tiền đơn hàng với discount.
def calculate_total(self, items: list) -> float:
        """Tính tổng tiền đơn hàng với discount."""
        subtotal = sum(i["unit_price"] * i["quantity"] for i in items)
        discount = self.get_discount(subtotal)
        # BUG: trả về subtotal thay vì subtotal - discount
        return subtotal
```

---

### ID: `31c5fba04fb78ba3`
- **Type:** function
- **Name:** `get_discount`
- **File:** `app.py`
- **Lines:** 94 -> 98

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "language": "python",
  "file_path": "app.py",
  "name": "get_discount",
  "parent_class": "OrderProcessor",
  "signature": "def get_discount(self, subtotal: float) -> float:",
  "chunk_type": "function",
  "start_line": 94,
  "end_line": 98
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: get_discount
Class: OrderProcessor
Signature: def get_discount(self, subtotal: float) -> float:
Description: 10% discount nếu đơn hàng > 500k.
def get_discount(self, subtotal: float) -> float:
        """10% discount nếu đơn hàng > 500k."""
        if subtotal > 500_000:
            return subtotal * 0.1
        return 0.0
```

---

### ID: `de2f0cf721645c0f`
- **Type:** function
- **Name:** `get_order_by_id`
- **File:** `app.py`
- **Lines:** 100 -> 102

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "language": "python",
  "start_line": 100,
  "file_path": "app.py",
  "end_line": 102,
  "parent_class": "OrderProcessor",
  "name": "get_order_by_id",
  "chunk_type": "function",
  "signature": "def get_order_by_id(self, order_id: int) -> dict:"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: get_order_by_id
Class: OrderProcessor
Signature: def get_order_by_id(self, order_id: int) -> dict:
def get_order_by_id(self, order_id: int) -> dict:
        # BUG: index sai — order_id bắt đầu từ 1 nhưng index từ 0
        return self.orders[order_id]
```

---

### ID: `fff120e7c76cca8f`
- **Type:** class
- **Name:** `UserService`
- **File:** `app.py`
- **Lines:** 105 -> 123

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "signature": "class UserService:",
  "language": "python",
  "start_line": 105,
  "parent_class": "",
  "end_line": 123,
  "chunk_type": "class",
  "file_path": "app.py",
  "name": "UserService"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: class  Name: UserService
Signature: class UserService:
Description: Quản lý user.
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
```

---

### ID: `ae188f37029ca061`
- **Type:** function
- **Name:** `__init__`
- **File:** `app.py`
- **Lines:** 108 -> 109

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "signature": "def __init__(self):",
  "end_line": 109,
  "parent_class": "UserService",
  "language": "python",
  "start_line": 108,
  "name": "__init__",
  "file_path": "app.py",
  "chunk_type": "function"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: __init__
Class: UserService
Signature: def __init__(self):
def __init__(self):
        self.users: dict[int, dict] = {}
```

---

### ID: `887ce625950f430c`
- **Type:** function
- **Name:** `create_user`
- **File:** `app.py`
- **Lines:** 111 -> 115

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "parent_class": "UserService",
  "language": "python",
  "chunk_type": "function",
  "name": "create_user",
  "end_line": 115,
  "file_path": "app.py",
  "signature": "def create_user(self, name: str, email: str) -> dict:",
  "start_line": 111
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: create_user
Class: UserService
Signature: def create_user(self, name: str, email: str) -> dict:
def create_user(self, name: str, email: str) -> dict:
        user_id = len(self.users) + 1
        user = {"id": user_id, "name": name, "email": email, "active": True}
        self.users[user_id] = user
        return user
```

---

### ID: `ae45d8abf7b8d1fd`
- **Type:** function
- **Name:** `get_user`
- **File:** `app.py`
- **Lines:** 117 -> 119

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "end_line": 119,
  "chunk_type": "function",
  "start_line": 117,
  "parent_class": "UserService",
  "file_path": "app.py",
  "signature": "def get_user(self, user_id: int) -> dict:",
  "language": "python",
  "name": "get_user"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: get_user
Class: UserService
Signature: def get_user(self, user_id: int) -> dict:
def get_user(self, user_id: int) -> dict:
        # BUG: không handle user không tồn tại → KeyError
        return self.users[user_id]
```

---

### ID: `c8fb03b61eba7724`
- **Type:** function
- **Name:** `deactivate_user`
- **File:** `app.py`
- **Lines:** 121 -> 123

<details>
  <summary>Metadata (JSON)</summary>

```json
{
  "start_line": 121,
  "end_line": 123,
  "chunk_type": "function",
  "name": "deactivate_user",
  "file_path": "app.py",
  "parent_class": "UserService",
  "language": "python",
  "signature": "def deactivate_user(self, user_id: int):"
}
```
</details>

**Document Segment:**
```python
File: app.py
Type: function  Name: deactivate_user
Class: UserService
Signature: def deactivate_user(self, user_id: int):
def deactivate_user(self, user_id: int):
        user = self.get_user(user_id)
        user["active"] = False
```

---

