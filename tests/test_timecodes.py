import pytest

from src.output import seconds_to_timecode


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00:00"), (5.9, "00:00:05"), (65, "00:01:05"), (3661, "01:01:01")],
)
def test_seconds_to_timecode(seconds: float, expected: str) -> None:
    assert seconds_to_timecode(seconds) == expected

