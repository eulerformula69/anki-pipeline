from src import audio_sources
from src.audio_sources import find_commons_audio, forvo_search_url, infer_audio_text, strip_furigana


def test_furigana_is_removed_without_removing_written_form() -> None:
    assert strip_furigana("について 関[かん]する") == "について関する"
    assert strip_furigana("今日[きょう]は休み") == "今日は休み"


def test_audio_text_prefers_japanese_answer_then_question() -> None:
    assert infer_audio_text("Что значит?", "について 関[かん]する") == "について関する"
    assert infer_audio_text("猫[ねこ]がいる。", "Кошка здесь.") == "猫がいる。"
    assert infer_audio_text("Question", "Answer") == ""


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
