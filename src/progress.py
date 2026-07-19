from __future__ import annotations

import sys
import threading
import time
from types import TracebackType
from typing import TextIO


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ActivityIndicator:
    """Reusable terminal spinner for work with unknowable completion percentage."""

    def __init__(
        self,
        label: str,
        *,
        stream: TextIO = sys.stderr,
        interval: float = 0.15,
    ) -> None:
        self.label = label
        self.stream = stream
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._last_width = 0

    def __enter__(self) -> ActivityIndicator:
        if not self.stream.isatty():
            return self
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self.stream.write("\r" + (" " * self._last_width) + "\r")
        self.stream.flush()

    def _animate(self) -> None:
        frames = "|/-\\"
        frame = 0
        while not self._stop.is_set():
            elapsed = format_elapsed(time.monotonic() - self._started_at)
            line = f"{self.label} {frames[frame % len(frames)]} {elapsed}"
            self._last_width = max(self._last_width, len(line))
            self.stream.write("\r" + line.ljust(self._last_width))
            self.stream.flush()
            frame += 1
            self._stop.wait(self.interval)
