from __future__ import annotations

import json
import os
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from ..anki_export import SAFE_CARD_ID


class ReviewError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class ReviewStore:
    def __init__(self) -> None:
        self._locks: dict[Path, threading.RLock] = {}
        self._guard = threading.Lock()

    def _lock(self, path: Path) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(path.resolve(), threading.RLock())

    def load_or_create(self, output: Path) -> dict[str, Any]:
        approved = output / "approved_cards.json"
        with self._lock(approved):
            candidates = self._read(output / "card_candidates.json")
            raw_cards = candidates.get("card_candidates")
            if not isinstance(raw_cards, list):
                raise ReviewError("card_candidates.json не содержит card_candidates.")
            self._unique_ids(raw_cards)
            if approved.exists():
                review = self._read(approved)
                known = {card.get("card_id") for card in review.get("cards", [])}
                for raw in raw_cards:
                    if raw["card_id"] not in known:
                        review["cards"].append(self._draft_card(raw, output))
                current = {raw["card_id"] for raw in raw_cards}
                for card in review["cards"]:
                    card["stale"] = card["card_id"] not in current
                self.validate(review, output, approving=False)
                atomic_json(review, approved)
                return review
            review = {
                "source": candidates.get("source"), "review_version": 1,
                "status": "draft", "updated_at": now_iso(), "approved_at": None,
                "cards": [self._draft_card(raw, output) for raw in raw_cards],
            }
            self.validate(review, output, approving=False)
            atomic_json(review, approved)
            return review

    def _draft_card(self, raw: dict[str, Any], output: Path) -> dict[str, Any]:
        media = self._automatic_media(output).get(raw.get("card_id"))
        return {
            "card_id": raw.get("card_id"), "included": True,
            "front": raw.get("front", ""), "back": raw.get("back", ""),
            "extra": raw.get("extra", ""), "tags": raw.get("tags", []),
            "difficulty": raw.get("difficulty", 1), "importance": raw.get("importance", 1),
            "image": ({"mode": "automatic", "path": media[0], "timestamp": media[1]} if media and media[0] else {"mode": "none"}),
            "overrides": {}, "reviewed": False, "changed": False,
            "updated_at": now_iso(), "stale": False,
        }

    def save_card(self, output: Path, card_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        path = output / "approved_cards.json"
        with self._lock(path):
            review = self.load_or_create(output)
            card = next((item for item in review["cards"] if item["card_id"] == card_id), None)
            if card is None:
                raise ReviewError("Неизвестный card_id.")
            allowed = {"included", "front", "back", "extra", "tags", "difficulty", "importance", "image", "overrides", "reviewed"}
            card.update({key: deepcopy(value) for key, value in updates.items() if key in allowed})
            card["changed"] = True
            card["reviewed"] = True
            card["updated_at"] = now_iso()
            review.update(status="draft", approved_at=None, updated_at=now_iso())
            self.validate(review, output, approving=False)
            atomic_json(review, path)
            return card

    def approve(self, output: Path) -> dict[str, Any]:
        path = output / "approved_cards.json"
        with self._lock(path):
            review = self.load_or_create(output)
            self.validate(review, output, approving=True)
            review["status"] = "approved"
            review["approved_at"] = review["updated_at"] = now_iso()
            atomic_json(review, path)
            return review

    def validate(self, review: dict[str, Any], output: Path, *, approving: bool) -> None:
        if review.get("review_version") != 1 or not isinstance(review.get("cards"), list):
            raise ReviewError("Некорректный контракт approved_cards.json.")
        self._unique_ids(review["cards"])
        for card in review["cards"]:
            card_id = card.get("card_id")
            if not isinstance(card_id, str) or not SAFE_CARD_ID.fullmatch(card_id):
                raise ReviewError("Некорректный card_id.")
            if not isinstance(card.get("included"), bool):
                raise ReviewError(f"{card_id}: included должен быть boolean.")
            for key in ("front", "back", "extra"):
                if not isinstance(card.get(key), str):
                    raise ReviewError(f"{card_id}: {key} должен быть строкой.")
            if approving and card["included"] and (not card["front"].strip() or not card["back"].strip()):
                raise ReviewError(f"{card_id}: заполните front и back.")
            if not isinstance(card.get("tags"), list) or any(not isinstance(tag, str) for tag in card["tags"]):
                raise ReviewError(f"{card_id}: некорректные tags.")
            for key in ("difficulty", "importance"):
                if isinstance(card.get(key), bool) or not isinstance(card.get(key), int) or not 1 <= card[key] <= 5:
                    raise ReviewError(f"{card_id}: {key} должен быть от 1 до 5.")
            image = card.get("image") or {"mode": "none"}
            if image.get("mode") not in {"automatic", "selected_candidate", "uploaded", "none"}:
                raise ReviewError(f"{card_id}: неизвестный режим изображения.")
            if image["mode"] != "none":
                relative = Path(str(image.get("path", "")))
                resolved = (output / relative).resolve()
                if relative.is_absolute() or ".." in relative.parts or not resolved.is_relative_to(output.resolve()) or not resolved.is_file():
                    raise ReviewError(f"{card_id}: недопустимый путь изображения.")

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewError(f"Не удалось прочитать {path.name}: {exc}") from exc
        if not isinstance(value, dict):
            raise ReviewError(f"{path.name} должен содержать JSON-объект.")
        return value

    @staticmethod
    def _unique_ids(cards: list[Any]) -> None:
        ids = [card.get("card_id") for card in cards if isinstance(card, dict)]
        if len(ids) != len(cards) or len(ids) != len(set(ids)):
            raise ReviewError("card_id должны быть уникальными.")

    def _automatic_media(self, output: Path) -> dict[str, tuple[str | None, float | None]]:
        path = output / "card_media.json"
        if not path.exists():
            return {}
        data = self._read(path)
        result = {}
        for item in data.get("cards", []):
            image = item.get("image") or {}
            result[item.get("card_id")] = (image.get("path"), image.get("timestamp", item.get("start")))
        return result


def normalize_uploaded_image(data: bytes, destination: Path, *, max_bytes: int = 10 * 1024 * 1024) -> Path:
    if len(data) > max_bytes:
        raise ReviewError("Изображение превышает лимит 10 МБ.")
    import io
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ReviewError("Разрешены только JPEG, PNG и WebP.")
            normalized = image.convert("RGB")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_suffix(".tmp")
            normalized.save(temp, format="JPEG", quality=92)
            os.replace(temp, destination)
    except ReviewError:
        raise
    except Exception as exc:
        raise ReviewError("Файл не является корректным изображением.") from exc
    return destination
