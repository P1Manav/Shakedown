from cart import add_item, total_price


def test_new_cart_starts_empty():
    assert add_item("apple") == ["apple"]


def test_calls_do_not_leak_state():
    assert add_item("apple") == ["apple"]
    assert add_item("banana") == ["banana"]


def test_explicit_cart_is_mutated():
    cart = ["apple"]
    result = add_item("pear", cart)
    assert result is cart
    assert cart == ["apple", "pear"]


def test_total_price():
    assert total_price(["apple", "pear"], {"apple": 2, "pear": 3}) == 5