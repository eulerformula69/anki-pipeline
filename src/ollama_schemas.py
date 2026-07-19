from __future__ import annotations


def review_response_schema(
    statuses: set[str], issues: set[str], actions: set[str], fields: set[str]
) -> dict:
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "knowledge_id": {"type": "string"},
                        "status": {"type": "string", "enum": sorted(statuses)},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "issues": {
                            "type": "array",
                            "items": {"type": "string", "enum": sorted(issues)},
                        },
                        "recommended_action": {
                            "type": "string",
                            "enum": sorted(actions),
                        },
                        "reason": {"type": "string"},
                    },
                    "required": sorted(fields),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }


def card_response_schema(card_types: set[str], fields: set[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "card_candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "knowledge_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "card_type": {
                            "type": "string",
                            "enum": sorted(card_types),
                        },
                        "front": {"type": "string"},
                        "back": {"type": "string"},
                        "extra": {"type": "string"},
                        "difficulty": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                        },
                    },
                    "required": sorted(fields),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["card_candidates"],
        "additionalProperties": False,
    }


def target_expression_response_schema() -> dict:
    fields = [
        "surface", "reading", "romanization", "meaning", "knowledge_ids",
        "source_transcript_segment_ids", "confidence",
    ]
    return {
        "type": "object",
        "properties": {
            "expressions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "surface": {"type": "string"},
                        "reading": {"type": "string"},
                        "romanization": {"type": "string"},
                        "meaning": {"type": "string"},
                        "knowledge_ids": {"type": "array", "items": {"type": "string"}},
                        "source_transcript_segment_ids": {
                            "type": "array", "items": {"type": "integer"}
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": fields,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["expressions"],
        "additionalProperties": False,
    }
