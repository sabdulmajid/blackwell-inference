from scripts.gpu_guard import parse_int, parse_float


def test_parse_int():
    assert parse_int("101376 MiB") == 101376
    assert parse_int("N/A") is None
    assert parse_int("") is None


def test_parse_float():
    assert parse_float("600.00 W") == 600.0
    assert parse_float("N/A") is None
