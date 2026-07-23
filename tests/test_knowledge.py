import json

import numpy as np
import pytest

from src.knowledge import (
    KnowledgeExtractionError,
    build_knowledge_prompt,
    build_evidence,
    deduplicate_items,
    evidence_matches,
    extract_knowledge,
    normalize_evidence,
    parse_knowledge_response,
    statements_are_obvious_duplicates,
)
from src.models import KnowledgeItem, SemanticSegment, TranscriptSegment


def source_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(0, 0.0, 2.0, "Yojijukugo are four-character idioms."),
        TranscriptSegment(1, 2.0, 4.0, "They can express a complete idea."),
        TranscriptSegment(2, 4.0, 6.0, "This is another topic."),
    ]


def semantic_segment() -> SemanticSegment:
    return SemanticSegment(
        id=0,
        title="Yojijukugo",
        start=0.0,
        end=4.0,
        summary="Definition",
        transcript_segment_ids=[0, 1],
        text="Yojijukugo are four-character idioms. They can express a complete idea.",
    )


def test_prompt_filters_video_meta_and_author_preferences() -> None:
    prompt = build_knowledge_prompt(
        semantic_segment(),
        {item.id: item for item in source_segments()},
        result_language="en",
        max_items=5,
    )

    assert "use outside this specific video" in prompt
    assert "author's biography, tastes, preferences" in prompt
    assert "When uncertain, omit the item" in prompt


def valid_item(**overrides):
    item = {
        "type": "definition",
        "statement": "Yojijukugo are four-character idioms.",
        "explanation": "They can express a complete idea.",
        "importance": 4,
        "source_transcript_segment_ids": [0, 1],
    }
    item.update(overrides)
    return {"knowledge_items": [item]}


def parse(data, *, max_items=12, raw=""):
    transcript = {item.id: item for item in source_segments()}
    return parse_knowledge_response(
        data,
        semantic_segment(),
        transcript,
        max_items=max_items,
        raw_response=raw,
    )


def test_correct_model_response() -> None:
    proposals = parse(valid_item())

    assert proposals[0].type == "definition"
    assert proposals[0].importance == 4


@pytest.mark.parametrize("importance", [0, 6])
def test_importance_out_of_range(importance) -> None:
    with pytest.raises(KnowledgeExtractionError, match="importance"):
        parse(valid_item(importance=importance))


def test_unknown_type() -> None:
    with pytest.raises(KnowledgeExtractionError, match="неизвестный type"):
        parse(valid_item(type="opinion"))


def test_empty_statement() -> None:
    with pytest.raises(KnowledgeExtractionError, match="statement пустой"):
        parse(valid_item(statement="  "))


def test_question_statement() -> None:
    with pytest.raises(KnowledgeExtractionError, match="не должен быть вопросом"):
        parse(valid_item(statement="What is yojijukugo?"))


def test_missing_transcript_id() -> None:
    with pytest.raises(KnowledgeExtractionError, match="не существуют"):
        parse(valid_item(source_transcript_segment_ids=[99]))


def test_id_from_another_semantic_segment() -> None:
    with pytest.raises(KnowledgeExtractionError, match="не принадлежат"):
        parse(valid_item(source_transcript_segment_ids=[2]))


def test_unordered_and_duplicate_ids_are_canonicalized() -> None:
    proposals = parse(valid_item(source_transcript_segment_ids=[1, 0, 1]))

    assert proposals[0].source_transcript_segment_ids == [0, 1]


def test_ordered_non_adjacent_ids_are_allowed() -> None:
    block = semantic_segment()
    block = SemanticSegment(
        id=block.id,
        title=block.title,
        start=block.start,
        end=6.0,
        summary=block.summary,
        transcript_segment_ids=[0, 1, 2],
        text=block.text,
    )
    transcript = {item.id: item for item in source_segments()}

    proposals = parse_knowledge_response(
        valid_item(source_transcript_segment_ids=[0, 2]),
        block,
        transcript,
        max_items=12,
    )

    assert proposals[0].source_transcript_segment_ids == [0, 2]


def test_item_limit() -> None:
    data = valid_item()
    data["knowledge_items"].append(dict(data["knowledge_items"][0]))

    with pytest.raises(KnowledgeExtractionError, match="Превышен лимит"):
        parse(data, max_items=1)


def test_think_content_is_rejected() -> None:
    with pytest.raises(KnowledgeExtractionError, match="think"):
        parse(valid_item(), raw="<think>secret</think>")


def test_evidence_is_assembled_exactly_from_original() -> None:
    transcript = {item.id: item for item in source_segments()}
    evidence = build_evidence([0, 1], transcript)

    assert evidence == (
        "Yojijukugo are four-character idioms. They can express a complete idea."
    )
    assert evidence_matches("  YOJIJUKUGO are four-character idioms.\nThey can express a complete idea. ", [0, 1], transcript)
    assert not evidence_matches("Yojijukugo means an ancient proverb.", [0], transcript)
    assert normalize_evidence("A\n  B") == "a b"


class SequenceClient:
    model = "qwen3:8b"

    def __init__(self, responses) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def generate_json(self, prompt: str):
        self.calls += 1
        return next(self.responses)


def write_sources(tmp_path):
    transcript = {
        "source": "video.webm",
        "language": "en",
        "segments": [item.__dict__ for item in source_segments()],
    }
    semantic = {
        "source": "video.webm",
        "model": "qwen3:8b",
        "segments": [semantic_segment().__dict__],
    }
    transcript_path = tmp_path / "transcript.json"
    semantic_path = tmp_path / "semantic.json"
    transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    return transcript_path, semantic_path


def test_retry_filter_numbering_and_utf8(tmp_path) -> None:
    transcript_path, semantic_path = write_sources(tmp_path)
    low = valid_item(statement="Второстепенная деталь", importance=1)
    corrected = valid_item(statement="Ёмкое определение", importance=4)
    corrected["knowledge_items"].extend(low["knowledge_items"])
    client = SequenceClient(
        [({"knowledge_items": [{"bad": True}]}, "bad"), (corrected, "corrected")]
    )
    destination = tmp_path / "knowledge.json"

    result = extract_knowledge(
        transcript_path,
        semantic_path,
        destination,
        tmp_path / "diagnostic.json",
        client,
        language="ru",
        min_importance=2,
    )

    assert client.calls == 2
    assert [item.id for item in result.knowledge_items] == ["k_0001"]
    assert result.knowledge_items[0].statement == "Ёмкое определение"
    assert "Ёмкое" in destination.read_text(encoding="utf-8")


def test_invalid_block_uses_fallback_and_preserves_diagnostic(tmp_path) -> None:
    transcript_path, semantic_path = write_sources(tmp_path)
    invalid = ({"knowledge_items": [{"bad": True}]}, "invalid raw")
    client = SequenceClient([invalid, invalid])
    destination = tmp_path / "knowledge.json"
    destination.write_text("existing", encoding="utf-8")
    diagnostic = tmp_path / "diagnostic.json"

    result = extract_knowledge(
        transcript_path,
        semantic_path,
        destination,
        diagnostic,
        client,
    )

    assert result.knowledge_items[0].statement == "Definition"
    assert "Automatic fallback" in result.knowledge_items[0].explanation
    diagnostic_data = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert diagnostic_data["semantic_segment_id"] == 0
    assert len(diagnostic_data["ollama_responses"]) == 2
    assert diagnostic_data["model"] == "qwen3:8b"
    assert diagnostic_data["created_at"]


def test_valid_items_survive_invalid_siblings() -> None:
    data = valid_item()
    data["knowledge_items"].append(
        valid_item(source_transcript_segment_ids=[99])["knowledge_items"][0]
    )

    proposals = parse(data)

    assert len(proposals) == 1


def test_exact_duplicate_is_deduplicated() -> None:
    transcript = {item.id: item for item in source_segments()}
    first = KnowledgeItem("", 0, "fact", "Same fact.", "", 3, [0], source_segments()[0].text)
    duplicate = KnowledgeItem("", 1, "fact", "Same fact!", "", 4, [2], source_segments()[2].text)

    result = deduplicate_items([first, duplicate], transcript)

    assert len(result) == 1
    assert result[0].duplicate_occurrences[0].semantic_segment_id == 1


def test_same_source_and_same_core_meaning_is_deduplicated() -> None:
    evidence = source_segments()[0].text
    first = KnowledgeItem(
        "", 0, "definition", "Datte has multiple different meanings.", "", 4, [0], evidence
    )
    second = KnowledgeItem(
        "", 0, "fact", "Dictionaries list multiple different meanings for datte.", "", 3, [0], evidence
    )

    assert statements_are_obvious_duplicates(first, second)


def test_distinct_facts_from_same_source_are_preserved() -> None:
    evidence = source_segments()[0].text
    first = KnowledgeItem(
        "", 0, "fact", "Datte is used in Japanese.", "", 3, [0], evidence
    )
    second = KnowledgeItem(
        "", 0, "fact", "The expression contains four characters.", "", 3, [0], evidence
    )

    assert not statements_are_obvious_duplicates(first, second)


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, statements: list[str]) -> np.ndarray:
        return np.asarray([self.vectors[item] for item in statements], dtype=np.float32)


def test_semantic_paraphrases_are_deduplicated_across_video() -> None:
    transcript = {item.id: item for item in source_segments()}
    first = KnowledgeItem(
        "", 0, "fact", "Water freezes at zero Celsius.", "", 4, [0],
        source_segments()[0].text,
    )
    second = KnowledgeItem(
        "", 1, "fact", "At 0°C water turns into ice.", "", 3, [2],
        source_segments()[2].text,
    )
    embedder = FakeEmbedder({
        first.statement: [1.0, 0.0],
        second.statement: [0.99, 0.01],
    })

    result = deduplicate_items(
        [first, second],
        transcript,
        embedder=embedder,
    )

    assert len(result) == 1
    assert result[0].duplicate_occurrences[0].semantic_segment_id == 1


def test_parser_marks_non_atomic_statement() -> None:
    data = valid_item(
        statement=(
            "Alpha is first and beta is second, but gamma is third "
            "and delta is fourth."
        )
    )

    proposal = parse(data)[0]

    assert proposal.atomicity_issues == ("multiple_clauses",)
