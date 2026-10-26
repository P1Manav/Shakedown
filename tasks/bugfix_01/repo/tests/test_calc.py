from calc import mean, median


def test_median_even_length_averages_middle_two():
    assert median([1, 2, 3, 4]) == 2.5
    assert median([4, 1, 3, 2]) == 2.5
    assert median([10, 2]) == 6


def test_median_odd_length():
    assert median([5, 1, 3]) == 3
    assert median([7]) == 7


def test_median_does_not_mutate_input():
    values = [3, 1, 2]
    median(values)
    assert values == [3, 1, 2]


def test_mean_sanity():
    assert mean([1, 2, 3]) == 2