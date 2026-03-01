# Sample buggy Python file for testing the debug agent

def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    average = total / len(numbers)
    return average


def find_max(numbers):
    """Find the maximum number in a list."""
    max_num = 0
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num


def reverse_string(s):
    """Reverse a string."""
    reversed_s = ""
    for i in range(len(s) - 1, 0, -1):
        reversed_s += s[i - 1]
    return reversed_s


def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


def is_palindrome(s):
    """Check if a string is a palindrome."""
    s = s.lower()
    return s == s[::-1]


if __name__ == "__main__":
    print(calculate_average([1, 2, 3, 4, 5]))
    print(calculate_average([]))

    print(find_max([-5, -3, -1]))
    print(find_max([]))

    print(reverse_string("hello"))

    print(fibonacci(35))

    print(is_palindrome("A man, a plan, a canal: Panama"))