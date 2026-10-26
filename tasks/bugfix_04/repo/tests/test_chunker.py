import pytest

from chunker import chunk, flatten


def test_even_split():
    assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_keeps_partial_final_chunk():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_short_input_still_produces_a_chunk():
    assert chunk([1, 2], 5) == [[1, 2]]


def test_round_trip():
    data = list(range(11))
    assert flatten(chunk(data, 4)) == data


def test_rejects_bad_size():
    with pytest.raises(ValueError):
        chunk([1], 0)