from chunker import chunk, flatten


def test_chunker_hidden_various_sizes():
    assert chunk(list(range(10)), 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert chunk([], 3) == []
    assert flatten(chunk(list(range(25)), 7)) == list(range(25))
