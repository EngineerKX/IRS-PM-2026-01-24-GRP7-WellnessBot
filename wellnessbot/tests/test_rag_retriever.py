from __future__ import annotations

from wellnessbot.rag import retrieve_recommendation_chunks


def test_retrieve_recommendation_chunks_prefers_exact_chunk_ids():
    payload = retrieve_recommendation_chunks(
        exercise_id="P3_E7",
        evidence_chunks=[
            {"chunk_id": "P3_E7_C1", "source_id": "S9"},
            {"chunk_id": "P3_E7_C2", "source_id": "S10"},
        ],
        top_k=2,
    )

    rows = payload["results"]

    assert payload["query_mode"] == "chunk_id"
    assert [row["chunk_id"] for row in rows] == ["P3_E7_C1", "P3_E7_C2"]
    assert [row["source_id"] for row in rows] == ["S9", "S10"]
    assert rows[0]["source_url"]
    assert all(row["text"] for row in rows)


def test_retrieve_recommendation_chunks_returns_empty_without_evidence_chunks():
    payload = retrieve_recommendation_chunks(
        exercise_id="P4_E6",
        evidence_chunks=[],
        top_k=2,
    )

    assert payload["query_mode"] == "none"
    assert payload["query_text"] == ""
    assert payload["results"] == []