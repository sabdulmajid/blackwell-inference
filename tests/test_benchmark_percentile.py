from benchmarks.serve_bench import percentile


def test_percentile_empty():
    assert percentile([], 50) is None


def test_percentile_values():
    assert percentile([1, 2, 3], 50) == 2
    assert percentile([1, 2, 3], 95) == 3
