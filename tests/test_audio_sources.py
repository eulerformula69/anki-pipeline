from src import audio_sources
from src.audio_sources import (
    find_commons_audio,
    find_jpod101_audio,
    find_native_audio,
    forvo_search_url,
    furigana_reading,
    infer_audio_reading,
    infer_audio_text,
    strip_furigana,
)


def test_furigana_is_removed_without_removing_written_form() -> None:
    assert strip_furigana("について 関[かん]する") == "について関する"
    assert strip_furigana("今日[きょう]は休み") == "今日は休み"
    assert furigana_reading("について 関[かん]する") == "についてかんする"


def test_audio_text_prefers_japanese_answer_then_question() -> None:
    assert infer_audio_text("Что значит?", "について 関[かん]する") == "について関する"
    assert infer_audio_text("猫[ねこ]がいる。", "Кошка здесь.") == "猫がいる。"
    assert infer_audio_text("Question", "Answer") == ""
    assert infer_audio_reading("Что значит?", "検索[けんさく]エンジン") == "けんさくエンジン"


def test_forvo_url_uses_clean_encoded_text() -> None:
    assert forvo_search_url("日本語") == "https://forvo.com/word/%E6%97%A5%E6%9C%AC%E8%AA%9E/#ja"


def test_commons_lookup_batches_candidates_and_accepts_open_japanese_audio(monkeypatch) -> None:
    responses = [
        {"query": {"search": [{"title": "File:wrong.ogg"}, {"title": "File:right.ogg"}]}},
        {"query": {"pages": [{"imageinfo": [{
            "url": "https://upload.example/right.ogg",
            "descriptionurl": "https://commons.example/File:right.ogg",
            "mime": "application/ogg",
            "extmetadata": {
                "ImageDescription": {"value": "Pronunciation of Japanese word 「寿司」"},
                "Categories": {"value": "Japanese pronunciation"},
                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                "Artist": {"value": "Native speaker"},
            },
        }]}]}},
    ]
    calls = []
    monkeypatch.setattr(audio_sources, "_request_json", lambda parameters: calls.append(parameters) or responses.pop(0))
    monkeypatch.setattr(audio_sources, "_request_bytes", lambda url: b"OggSaudio")

    result = find_commons_audio("寿司")

    assert result is not None and result.data == b"OggSaudio"
    assert calls[1]["titles"] == "File:wrong.ogg|File:right.ogg"


def test_jpod101_lookup_rejects_missing_audio_marker(monkeypatch) -> None:
    missing = b"missing-audio"
    monkeypatch.setattr(audio_sources, "JPOD101_MISSING_MD5", audio_sources.hashlib.md5(
        missing, usedforsecurity=False
    ).hexdigest())
    monkeypatch.setattr(audio_sources, "_request_bytes", lambda url: missing)

    assert find_jpod101_audio("存在しない語", "そんざいしないご") is None


def test_jpod101_lookup_returns_mp3(monkeypatch) -> None:
    monkeypatch.setattr(audio_sources, "_request_bytes", lambda url: b"ID3native-audio")

    result = find_jpod101_audio("論文", "ろんぶん")

    assert result is not None and result.suffix == ".mp3"
    assert result.source_name == "JapanesePod101"


def test_native_lookup_uses_commons_when_dictionary_is_unavailable(monkeypatch) -> None:
    expected = audio_sources.AudioDownload(b"audio", ".ogg", "source", "CC0", "speaker")

    def unavailable(value: str, reading: str) -> None:
        raise audio_sources.AudioLookupError("offline")

    monkeypatch.setattr(audio_sources, "find_jpod101_audio", unavailable)
    monkeypatch.setattr(audio_sources, "find_commons_audio", lambda value: expected)

    assert find_native_audio("寿司", "すし") == expected
