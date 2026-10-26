from pathlib import Path

import pytest


@pytest.fixture
def speech_mp3():
    return (Path(__file__).parent / "fixtures" / "speech.mp3").read_bytes()
