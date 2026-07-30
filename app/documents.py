"""Both input sources -> one uniform document record.

Inline text and ES-fetched documents project into the same 8 string columns with an explicit
Spark schema. custom_nlp_service instead lets pyspark infer a schema from ~17-key dicts, which
breaks when a doc is missing a key or a nested field drifts shape. Only `text` reaches the model;
the id columns ride along so rows can be attributed back.

`load()` is the single entry point: callers hand it a request payload and get one result back,
without knowing which source (or both) produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pyspark.sql.types import StringType, StructField, StructType

import es_client

DOC_COLUMNS = (
    "row_id",
    "document_id",
    "patient_id",
    "visit_id",
    "job_id",
    "batch_id",
    "source_index",
    "text",
)

DOC_SCHEMA = StructType([StructField(name, StringType(), True) for name in DOC_COLUMNS])

INLINE_SOURCE_INDEX = "inline"


@dataclass
class LoadResult:
    docs: List[Dict[str, Optional[str]]] = field(default_factory=list)
    skipped: int = 0
    # Human-readable notes for the session status: blank documents, ids not in the index.
    warnings: List[str] = field(default_factory=list)


def _record(**values: Any) -> Dict[str, Optional[str]]:
    return {
        name: (None if values.get(name) is None else str(values.get(name)))
        for name in DOC_COLUMNS
    }


def _inline_documents(items: List[Any]) -> List[Dict[str, Optional[str]]]:
    """Inline payload documents -> doc records.

    document_id falls back to `inline-{i}` because rows.py hashes it into relation_id /
    structure_id; a None there would collapse distinct documents' ids onto each other.
    """
    docs = []
    for index, item in enumerate(items or []):
        data = item if isinstance(item, dict) else item.model_dump()
        docs.append(
            _record(
                row_id=data.get("row_id") or f"{INLINE_SOURCE_INDEX}-{index}",
                document_id=data.get("document_id") or f"{INLINE_SOURCE_INDEX}-{index}",
                patient_id=data.get("patient_id"),
                visit_id=data.get("visit_id"),
                job_id=data.get("job_id"),
                batch_id=data.get("batch_id"),
                source_index=INLINE_SOURCE_INDEX,
                text=data.get("text"),
            )
        )
    return docs


def load(payload: Dict[str, Any]) -> LoadResult:
    """Load every document a request asked for, from either source or both."""
    result = LoadResult(docs=_inline_documents(payload.get("documents")))

    document_ids = payload.get("document_ids") or []
    if not document_ids:
        return result

    fetched = es_client.fetch_source_documents(document_ids, payload.get("input_index"))
    result.docs.extend(_record(**doc) for doc in fetched.docs)
    result.skipped = fetched.skipped_count

    if fetched.skipped_count:
        result.warnings.append(f"Skipped {fetched.skipped_count} document(s) with empty text.")
    # matched_ids includes blank-text docs, so this is only ids the index truly lacks. Reporting
    # a found-but-empty doc as "not found" would send the caller hunting for an indexing bug
    # that does not exist.
    unmatched = set(document_ids) - fetched.matched_ids
    if unmatched:
        result.warnings.append(f"No document found in Elasticsearch for: {sorted(unmatched)}")

    return result
