"""PretrainedZeroShotMultiTask `extractions` -> flat row-per-annotation records.

SHARED by the service (``runner.py``) and ``build_multitask_pipeline.ipynb``. Stdlib only, for
the same reason as ``zeroshot_dsl`` -- the notebook imports it from the host.

All four tasks arrive in one `extractions` column, discriminated by `annotatorType`:

    annotation                        rows    label            span_text     start/end
    chunk (ner)                       1       metadata.entity  chunk text    annotation begin/end
    category WITH task (classify)     1       metadata.task    predicted     annotation begin/end
    category WITHOUT task (relation)  2       relation name    chunk1/chunk2 that entity's offsets
    struct                            1/field field name       field text    field start/end

`label` therefore means something different per task -- consumers must read
`raw_metadata.task_type` to interpret it. Relation rows carry `raw_metadata.role` (head/tail) and
share a `relation_id`; structure rows share a `structure_id` + `instance_idx`, so a full relation
or structure instance can be regrouped.

An annotation's `metadata["sentence"]` is only an integer INDEX into the pipeline's sentence
column (Spark NLP convention), which is useless on its own. Pass `sentences` (that column) and
the row's `sentence` field carries the real sentence text, with the index kept as
`raw_metadata["sentence_number"]`.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

ENGINE_MULTITASK = "jsl_zero_shot_multitask"
MODEL_TYPE = "multitask"
SOURCE_MODE = "custom_nlp_service"

# The four task types a row's raw_metadata.task_type can carry.
TASK_TYPES = ("ner", "structure", "classification", "relation")

COMMON_KEYS = ("row_id", "document_id", "patient_id", "visit_id", "job_id", "batch_id")


def _get(annotation: Any, name: str) -> Any:
    """Read a field off a pyspark Row or a plain dict, so both sides share this code."""
    if isinstance(annotation, dict):
        return annotation.get(name)
    return getattr(annotation, name, None)


def _p_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _p_flt(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _sentence_text(sentences: Optional[Sequence[Any]], index: Optional[int]) -> Optional[str]:
    """Resolve metadata["sentence"] (an index) to the real sentence text.

    `sentences` is the pipeline's sentence column: Annotation objects with `.result`. Plain
    strings and dicts are accepted too, so a caller can pass already-extracted text.
    """
    if sentences is None or index is None or not 0 <= index < len(sentences):
        return None
    sentence = sentences[index]
    if isinstance(sentence, str):
        return sentence
    if isinstance(sentence, dict):
        return sentence.get("result")
    return getattr(sentence, "result", None)


def _p_struct(text: Any) -> Any:
    """Parse a struct field payload. The annotator emits Python-repr dicts, so literal_eval first."""
    if not text:
        return None
    if not isinstance(text, str):
        return text
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def relation_id(document_id: Any, relation_name: str, e1b: Any, e1e: Any, e2b: Any, e2e: Any) -> str:
    key = f"{document_id}|{relation_name}|{e1b}|{e1e}|{e2b}|{e2e}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def structure_id(document_id: Any, structure_name: str, sentence: Any, instance_idx: Any) -> str:
    # instance_idx restarts per sentence, so sentence must be part of the key.
    key = f"{document_id}|{structure_name}|{sentence}|{instance_idx}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def result_id(session_id: Any, pipeline_id: Any, engine: Any, row: Dict[str, Any]) -> str:
    key = "|".join(
        str(part)
        for part in (
            session_id,
            pipeline_id,
            engine,
            row.get("row_id"),
            row.get("label"),
            row.get("start"),
            row.get("end"),
            row.get("span_text"),
        )
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def collect_service_rows(
    extractions: Iterable[Any],
    common: Optional[Dict[str, Any]] = None,
    sentences: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """Split one document's `extractions` into flat rows, one per annotation (or per field/role).

    `sentences` is the document's sentence column; without it, `sentence` on every row is None
    (the bare index is not worth emitting).
    """
    common = common or {key: None for key in COMMON_KEYS}
    rows: List[Dict[str, Any]] = []

    for extraction in extractions or []:
        metadata = dict(_get(extraction, "metadata") or {})
        annotator_type = _get(extraction, "annotatorType")
        result = _get(extraction, "result")
        base = {**common, "start": _get(extraction, "begin"), "end": _get(extraction, "end")}

        sentence_number = _p_int(metadata.get("sentence"))
        sentence_text = _sentence_text(sentences, sentence_number)
        # The raw index moves to sentence_number; the "sentence" key would otherwise shadow the
        # resolved text and mean two different things under one name.
        metadata_without_sentence = {k: v for k, v in metadata.items() if k != "sentence"}

        # NER and classification differ only in where the label comes from: the entity type for
        # a chunk, the task name for a category that names its task.
        if annotator_type == "chunk":
            task_type, label = "ner", metadata.get("entity")
        elif annotator_type == "category" and metadata.get("task"):
            task_type, label = "classification", metadata.get("task")
        else:
            task_type = label = None

        if task_type:
            rows.append(
                {
                    **base,
                    "label": str(label or ""),
                    "span_text": result,
                    "sentence": sentence_text,
                    "confidence": _p_flt(metadata.get("confidence")),
                    "raw_metadata": {
                        "task_type": task_type,
                        **metadata_without_sentence,
                        "sentence_number": sentence_number,
                    },
                }
            )

        elif annotator_type == "category":
            name = str(result or "")
            e1b, e1e = _p_int(metadata.get("entity1_begin")), _p_int(metadata.get("entity1_end"))
            e2b, e2e = _p_int(metadata.get("entity2_begin")), _p_int(metadata.get("entity2_end"))
            c1, c2 = _p_flt(metadata.get("chunk1_confidence")), _p_flt(metadata.get("chunk2_confidence"))
            shared = {
                "task_type": "relation",
                "relation": name,
                "relation_id": relation_id(common.get("document_id"), name, e1b, e1e, e2b, e2e),
                "category_type": metadata.get("category_type"),
                "uuid": metadata.get("uuid"),
                "sentence_number": sentence_number,
            }
            for role, chunk, entity, begin, end, confidence in (
                ("head", metadata.get("chunk1"), metadata.get("entity1"), e1b, e1e, c1),
                ("tail", metadata.get("chunk2"), metadata.get("entity2"), e2b, e2e, c2),
            ):
                rows.append(
                    {
                        **base,
                        "start": begin,
                        "end": end,
                        "label": name,
                        "span_text": chunk,
                        "sentence": sentence_text,
                        "confidence": confidence,
                        "raw_metadata": {
                            **shared,
                            "role": role,
                            "entity": entity,
                            "chunk": chunk,
                            "begin": begin,
                            "end": end,
                            "chunk_confidence": confidence,
                        },
                    }
                )

        elif annotator_type == "struct":
            name = str(result or "")
            instance_idx = _p_int(metadata.get("instance_idx"))
            shared = {
                "task_type": "structure",
                "structure_name": name,
                "structure_id": structure_id(
                    common.get("document_id"), name, sentence_number, instance_idx
                ),
                "instance_idx": instance_idx,
                "uuid": metadata.get("uuid"),
                "sentence_number": sentence_number,
            }
            for field_name, raw in metadata.items():
                field = _p_struct(raw)
                # Field payloads are dicts with a "text" key; everything else in metadata is
                # a meta key (instance_idx / sentence / document / uuid).
                if not isinstance(field, dict) or "text" not in field:
                    continue
                begin, end = _p_int(field.get("start")), _p_int(field.get("end"))
                confidence = _p_flt(field.get("confidence"))
                rows.append(
                    {
                        **base,
                        "start": begin,
                        "end": end,
                        "label": field_name,
                        "span_text": field.get("text"),
                        "sentence": sentence_text,
                        "confidence": confidence,
                        "raw_metadata": {
                            **shared,
                            "field": field_name,
                            "chunk": field.get("text"),
                            "begin": begin,
                            "end": end,
                            "chunk_confidence": confidence,
                        },
                    }
                )

    return rows


def to_service_rows(
    extractions: Iterable[Any],
    *,
    pipeline_id: str,
    source_index: str,
    common: Optional[Dict[str, Any]] = None,
    sentences: Optional[Sequence[Any]] = None,
    created_at: Optional[str] = None,
    session_id: Optional[str] = None,
    job_details: Optional[str] = None,
    with_result_id: bool = False,
) -> List[Dict[str, Any]]:
    """Wrap collect_service_rows() output in the response envelope clients consume."""
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    out: List[Dict[str, Any]] = []

    for row in collect_service_rows(extractions, common, sentences):
        record = {
            **{key: row.get(key) for key in COMMON_KEYS},
            "pipeline_id": pipeline_id,
            "engine": ENGINE_MULTITASK,
            "label": row.get("label"),
            "span_text": row.get("span_text"),
            "start": row.get("start"),
            "end": row.get("end"),
            "sentence": row.get("sentence"),
            "confidence": row.get("confidence"),
            "source_index": source_index,
            "source_mode": SOURCE_MODE,
            "raw_metadata": {**(row.get("raw_metadata") or {}), "model_type": MODEL_TYPE},
            "created_at": created_at,
            "session_id": session_id,
            "job_details": job_details,
        }
        record["result_id"] = (
            result_id(session_id, pipeline_id, ENGINE_MULTITASK, row) if with_result_id else None
        )
        out.append(record)

    return out
