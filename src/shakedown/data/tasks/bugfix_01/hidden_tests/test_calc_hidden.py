from calc import median


def test_median_hidden_even_lists():
    assert median([100, 20, 50, 80]) == 65.0
    assert median([0, 0, 10, 20]) == 5.0
    assert median([-5, -1, -3, -7]) == -4.0
    assert median([1, 5, 2, 8, 3, 7]) == 4.0
