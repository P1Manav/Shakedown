from cart import add_item, total_price


def test_cart_hidden_multiple_independent_calls():
    c1 = add_item("orange")
    c2 = add_item("grape")
    assert c1 == ["orange"]
    assert c2 == ["grape"]
    assert c1 is not c2
    assert total_price(["orange", "grape"], {"orange": 4, "grape": 6}) == 10
