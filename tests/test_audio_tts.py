import pytest

from src import audio_tts
from src.audio_sources import AudioDownload, AudioLookupError


@pytest.mark.parametrize("offline", [False, True])
def test_missing_or_unavailable_native_audio_uses_tts(monkeypatch, offline):
    def native(value, reading):
        if offline:
            raise AudioLookupError("offline")
        return None

    expected = AudioDownload(b"mp3", ".mp3", "", "", "Nanami")
    monkeypatch.setattr(audio_tts, "find_native_audio", native)
    monkeypatch.setattr(audio_tts, "generate_audio", lambda text: expected)
    assert audio_tts.find_or_generate_audio("信頼できる", "しんらいできる") == expected


def test_native_recording_does_not_call_tts(monkeypatch):
    expected = AudioDownload(b"native", ".mp3", "", "", "speaker")
    monkeypatch.setattr(audio_tts, "find_native_audio", lambda *args: expected)
    monkeypatch.setattr(audio_tts, "generate_audio", lambda text: pytest.fail("TTS called"))
    assert audio_tts.find_or_generate_audio("信頼できる", "しんらいできる") == expected


def test_synthesis_joins_audio_chunks_and_strips_furigana(monkeypatch):
    class Communicate:
        def __init__(self, text, voice):
            assert text == "信頼できる"
            assert voice == "ja-JP-NanamiNeural"

        async def stream(self):
            yield {"type": "audio", "data": b"ID3"}
            yield {"type": "WordBoundary", "text": "ignored"}
            yield {"type": "audio", "data": b"speech"}

    monkeypatch.setattr(audio_tts.edge_tts, "Communicate", Communicate)
    result = audio_tts.generate_audio("信頼[しんらい]できる")
    assert result.data == b"ID3speech"
    assert result.suffix == ".mp3"
    assert result.source_name == "Синтез речи · Nanami"


@pytest.mark.parametrize("chunks", [[], [b"12345"]])
def test_empty_or_oversized_audio_is_not_saved(monkeypatch, chunks):
    class Communicate:
        def __init__(self, *args):
            pass

        async def stream(self):
            for data in chunks:
                yield {"type": "audio", "data": data}

    monkeypatch.setattr(audio_tts.edge_tts, "Communicate", Communicate)
    monkeypatch.setattr(audio_tts, "MAX_AUDIO_BYTES", 4)
    with pytest.raises(AudioLookupError):
        audio_tts.generate_audio("猫")


def test_network_error_becomes_audio_lookup_error(monkeypatch):
    async def unavailable(text):
        raise TimeoutError()

    monkeypatch.setattr(audio_tts, "_synthesize", unavailable)
    with pytest.raises(AudioLookupError):
        audio_tts.generate_audio("猫")
    assert audio_tts.generate_audio(" ") is None
