from __future__ import annotations

import re

from .card_input import CARD_TYPES, SOURCE_TAG


TAG_PREFIX = "anki-pipeline::"
LEGACY_ID_PREFIX = "anki_pipeline::"
SYSTEM_TAGS = {
    SOURCE_TAG, "manual", *CARD_TYPES,
    "basic", "cause_effect", "comparison", "definition", "example", "fact", "pitfall", "process",
}


def system_tag(tag: str) -> str:
    if tag.startswith(LEGACY_ID_PREFIX):
        return TAG_PREFIX + tag[len(LEGACY_ID_PREFIX):]
    if tag in SYSTEM_TAGS:
        return TAG_PREFIX + tag
    return tag


def note_tags(card_tags: list[str], user_tags: list[str]) -> list[str]:
    tags = [system_tag(tag) for tag in card_tags]
    tags.extend(user_tags)
    return list(dict.fromkeys(re.sub(r"\s+", "_", tag.strip()) for tag in tags if tag.strip()))


def identity_tags(marker: str) -> set[str]:
    identity = marker.removeprefix(TAG_PREFIX).removeprefix(LEGACY_ID_PREFIX)
    return {TAG_PREFIX + identity, LEGACY_ID_PREFIX + identity}
