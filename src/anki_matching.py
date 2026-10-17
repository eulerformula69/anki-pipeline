from __future__ import annotations

from typing import Any

from .anki_export import MODEL_NAME


class NoteMatchError(ValueError):
    pass


def field_value(note: dict[str, Any], name: str) -> str:
    return note.get("fields", {}).get(name, {}).get("value", "")


def find_import_targets(connection: Any, source: str,
                        prepared: list[tuple[str, dict[str, str]]]) -> list[dict[str, Any] | None]:
    """Match legacy package notes and tagged notes before changing the collection."""
    ids = connection.call("findNotes", query=f'"note:{MODEL_NAME}"')
    notes = []
    for offset in range(0, len(ids), 100):
        notes.extend(connection.call("notesInfo", notes=ids[offset:offset + 100]))
    notes = [note for note in notes if note.get("modelName") == MODEL_NAME]
    targets = []
    used = set()
    for marker, fields in prepared:
        candidates = [note for note in notes if marker in note.get("tags", []) or (
            field_value(note, "Source") == source
            and field_value(note, "Front") == fields["Front"]
            and field_value(note, "CardType") == fields["CardType"]
        )]
        if len(candidates) > 1:
            raise NoteMatchError(
                "В Anki уже есть несколько копий карточки из этого проекта. "
                "Импорт остановлен до изменений. Сначала разберите дубли в Anki."
            )
        target = candidates[0] if candidates else None
        if target:
            foreign_markers = [tag for tag in target.get("tags", [])
                               if tag.startswith("anki_pipeline::") and tag != marker]
            if foreign_markers or target["noteId"] in used:
                raise NoteMatchError(
                    "Карточка совпала с другой карточкой проекта. "
                    "Импорт остановлен до изменений, чтобы не заменить её."
                )
            used.add(target["noteId"])
        targets.append(target)
    return targets
