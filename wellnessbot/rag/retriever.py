from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

from wellnessbot.kg.loader import load_protocols
from wellnessbot.rag.bm25 import BM25Index, tokenize


@dataclass(frozen=True)
class ChunkDocument:
    chunk_id: str
    exercise_id: str
    exercise_name: str
    source_id: str
    source_url: str
    surgery_type: str
    phase_id: str
    phase_name: str
    tool: str
    text: str
    tokens: List[str]


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
_CHUNK_DB_PATH = _REPO_ROOT / "database_v2_csv.json"
_VERSIONED_CHUNK_DB_PATH = _REPO_ROOT / "data" / "database_v2_csv_version.json"


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _build_document_tokens(
    chunk_id: str,
    exercise_id: str,
    exercise_name: str,
    source_id: str,
    source_url: str,
    surgery_type: str,
    phase_id: str,
    phase_name: str,
    tool: str,
    alternate_names: Sequence[str],
    text: str,
) -> List[str]:
    parts = [
        chunk_id,
        exercise_id,
        exercise_name,
        source_id,
        source_url,
        surgery_type,
        phase_id,
        phase_name,
        tool,
        *alternate_names,
        text,
    ]
    joined = " ".join(part for part in parts if part)
    return tokenize(joined)


def _load_versioned_chunk_documents() -> List[ChunkDocument]:
    with _VERSIONED_CHUNK_DB_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    surgery_type = _safe_str(raw.get("condition") or raw.get("protocol_name") or raw.get("protocol_id"))
    exercises = raw.get("exercises", []) or []
    documents: List[ChunkDocument] = []

    for exercise in exercises:
        exercise_id = _safe_str(exercise.get("exercise_id"))
        exercise_name = _safe_str(exercise.get("canonical_name") or exercise.get("summary") or exercise_id)
        alternate_names = [
            _safe_str(name) for name in (exercise.get("alternate_names", []) or []) if _safe_str(name)
        ]
        phase_id = _safe_str(exercise.get("phase_id"))
        phase_name = _safe_str(exercise.get("phase_name"))
        tool = _safe_str(exercise.get("tool"))

        for chunk in exercise.get("instruction_chunks", []) or []:
            chunk_id = _safe_str(chunk.get("chunk_id"))
            text = _safe_str(chunk.get("text"))
            source_ids = [
                _safe_str(source_id)
                for source_id in (chunk.get("source_ids", []) or [])
                if _safe_str(source_id)
            ]
            source_urls = [
                _safe_str(source_url)
                for source_url in (chunk.get("source_urls", []) or [])
                if _safe_str(source_url)
            ]

            if not chunk_id or not text:
                continue

            source_count = max(len(source_ids), len(source_urls), 1)
            for idx in range(source_count):
                source_id = source_ids[idx] if idx < len(source_ids) else (source_ids[0] if source_ids else "")
                source_url = source_urls[idx] if idx < len(source_urls) else (source_urls[0] if source_urls else "")

                documents.append(
                    ChunkDocument(
                        chunk_id=chunk_id,
                        exercise_id=exercise_id,
                        exercise_name=exercise_name,
                        source_id=source_id,
                        source_url=source_url,
                        surgery_type=surgery_type,
                        phase_id=phase_id,
                        phase_name=phase_name,
                        tool=tool,
                        text=text,
                        tokens=_build_document_tokens(
                            chunk_id,
                            exercise_id,
                            exercise_name,
                            source_id,
                            source_url,
                            surgery_type,
                            phase_id,
                            phase_name,
                            tool,
                            alternate_names,
                            text,
                        ),
                    )
                )

    return documents


@lru_cache(maxsize=1)
def _load_chunk_documents() -> List[ChunkDocument]:
    if _VERSIONED_CHUNK_DB_PATH.exists():
        versioned_docs = _load_versioned_chunk_documents()
        if versioned_docs:
            return versioned_docs

    with _CHUNK_DB_PATH.open("r", encoding="utf-8") as handle:
        raw_chunks = json.load(handle)

    chunk_texts: Dict[str, str] = {}
    for item in raw_chunks:
        chunk_id = _safe_str(item.get("chunk_id"))
        text = _safe_str(item.get("chunk_text") or item.get("text"))
        if not chunk_id or not text or chunk_id in chunk_texts:
            continue
        chunk_texts[chunk_id] = text

    protocols = load_protocols(_DATA_DIR)
    documents: List[ChunkDocument] = []

    for protocol in protocols.values():
        for exercise in protocol.exercises.values():
            for evidence_chunk in exercise.evidence_chunks:
                chunk_id = _safe_str(evidence_chunk.chunk_id)
                text = chunk_texts.get(chunk_id, "")
                if not chunk_id or not text:
                    continue

                exercise_id = _safe_str(exercise.exercise_id)
                exercise_name = _safe_str(exercise.name)
                source_id = _safe_str(evidence_chunk.source_id)
                documents.append(
                    ChunkDocument(
                        chunk_id=chunk_id,
                        exercise_id=exercise_id,
                        exercise_name=exercise_name,
                        source_id=source_id,
                        source_url="",
                        surgery_type=_safe_str(protocol.surgery_type),
                        phase_id="",
                        phase_name="",
                        tool="",
                        text=text,
                        tokens=_build_document_tokens(
                            chunk_id,
                            exercise_id,
                            exercise_name,
                            source_id,
                            "",
                            _safe_str(protocol.surgery_type),
                            "",
                            "",
                            "",
                            [],
                            text,
                        ),
                    )
                )

    return documents


@lru_cache(maxsize=1)
def _load_bm25_index() -> BM25Index:
    docs = _load_chunk_documents()
    return BM25Index(doc.tokens for doc in docs)


def _build_query(chunk_ids: Sequence[str]) -> tuple[str, str]:
    normalized_chunk_ids = [_safe_str(chunk_id) for chunk_id in chunk_ids if _safe_str(chunk_id)]
    if normalized_chunk_ids:
        return "chunk_id", " ".join(normalized_chunk_ids)

    return "none", ""


def retrieve_recommendation_chunks(
    *,
    exercise_id: str | None = None,
    evidence_chunks: Sequence[Dict[str, Any]] | None = None,
    required_tools: Sequence[str] | None = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    documents = _load_chunk_documents()
    bm25 = _load_bm25_index()

    chunk_ids = [
        _safe_str(item.get("chunk_id"))
        for item in (evidence_chunks or [])
        if _safe_str(item.get("chunk_id"))
    ]
    query_mode, query_text = _build_query(chunk_ids)
    if not chunk_ids:
        return {
            "query_mode": query_mode,
            "query_text": query_text,
            "results": [],
        }

    normalized_tools = [_safe_str(tool) for tool in (required_tools or []) if _safe_str(tool)]
    if normalized_tools:
        suffix = " ".join(normalized_tools)
        query_text = f"{query_text} {suffix}".strip()
    query_tokens = tokenize(query_text)

    scores = bm25.score(query_tokens)
    exact_chunk_ids = set(chunk_ids)
    normalized_tool_set = {tool.lower() for tool in normalized_tools}

    ranked: List[Dict[str, Any]] = []
    for doc, score in zip(documents, scores):
        if doc.chunk_id not in exact_chunk_ids:
            continue

        boosted_score = score

        if doc.chunk_id in exact_chunk_ids:
            boosted_score += 100.0

        if normalized_tool_set and _safe_str(doc.tool).lower() in normalized_tool_set:
            boosted_score += 5.0

        if boosted_score <= 0.0:
            continue

        ranked.append(
            {
                "chunk_id": doc.chunk_id,
                "exercise_id": doc.exercise_id,
                "exercise_name": doc.exercise_name,
                "source_id": doc.source_id,
                "source_url": doc.source_url,
                "surgery_type": doc.surgery_type,
                "phase_id": doc.phase_id,
                "phase_name": doc.phase_name,
                "tool": doc.tool,
                "text": doc.text,
                "score": round(boosted_score, 4),
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            item["chunk_id"],
        )
    )

    return {
        "query_mode": query_mode,
        "query_text": query_text,
        "results": ranked[: max(top_k, 1)],
    }