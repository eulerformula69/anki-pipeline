import io
import time

from src.progress import ActivityIndicator, format_elapsed


class TerminalBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_format_elapsed() -> None:
    assert format_elapsed(5.9) == "00:05"
    assert format_elapsed(3661) == "01:01:01"


def test_activity_indicator_writes_label() -> None:
    stream = TerminalBuffer()

    with ActivityIndicator("Ollama: окно 1/1", stream=stream, interval=0.01):
        time.sleep(0.03)

    assert "Ollama: окно 1/1" in stream.getvalue()
