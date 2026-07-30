"""Optional Elasticsearch access: fetch source documents, optionally mirror results.

ES is only needed when a request supplies `document_ids`, or when RESULTS_ES_WRITE is on. The
client is built lazily and never at import, so the service boots and serves inline-text runs with
no ES reachable at all.

The query and text-extraction logic is carried over from custom_nlp_service/app/repository.py --
the field/field.keyword duality and the file_metadata.* nesting fallbacks are load-bearing
against real TPJ indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set

from logging_setup import logger
from settings import get_settings

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:  # pragma: no cover - absent during static checks outside the container
    Elasticsearch = helpers = None

# A ping on the /health path must not outlast Docker's healthcheck budget when ES is wedged.
PING_TIMEOUT_SECONDS = 2


class ElasticsearchUnavailable(RuntimeError):
    """ES was needed (document_ids input / result mirroring) but is disabled or unreachable."""


@dataclass
class DocumentFetchResult:
    docs: List[Dict[str, Any]] = field(default_factory=list)
    skipped_count: int = 0
    # Every document_id the query hit, blank-text ones included -- so callers can tell
    # "not in the index" apart from "found but empty", which are different problems.
    matched_ids: Set[str] = field(default_factory=set)


_client: Optional[Any] = None


def get_client():
    global _client
    settings = get_settings()
    if not settings.es_enabled:
        raise ElasticsearchUnavailable(
            "Elasticsearch is disabled (ES_ENABLED=false). Send inline `documents` instead of "
            "`document_ids`, or enable ES."
        )
    if Elasticsearch is None:
        raise ElasticsearchUnavailable("The 'elasticsearch' package is not installed.")
    if _client is None:
        logger.info("Connecting to Elasticsearch at %s", settings.es_url)
        _client = Elasticsearch(hosts=[settings.es_url], basic_auth=settings.es_basic_auth)
    return _client


def ping() -> bool:
    try:
        return bool(get_client().options(request_timeout=PING_TIMEOUT_SECONDS).ping())
    except Exception as exc:
        logger.warning("Elasticsearch ping failed: %s", exc)
        return False


def _document_id_query(document_ids: List[str]) -> Dict[str, Any]:
    """Match document_id whether it is nested under file_metadata or top-level, keyword or text."""
    fields = (
        "file_metadata.document_id",
        "file_metadata.document_id.keyword",
        "document_id",
        "document_id.keyword",
    )
    return {
        "query": {
            "bool": {
                "filter": [
                    {
                        "bool": {
                            "should": [{"terms": {f: document_ids}} for f in fields],
                            "minimum_should_match": 1,
                        }
                    }
                ]
            }
        }
    }


def _scroll_hits(client, index_name: str, query: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield every hit for a query, owning the scroll cursor and its cleanup."""
    response = client.search(index=index_name, body=query, size=1000, scroll="2m")
    scroll_id = response.get("_scroll_id")
    try:
        while True:
            hits = response["hits"]["hits"]
            if not hits:
                return
            yield from hits
            if not scroll_id:
                return
            response = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response.get("_scroll_id")
    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception:
                logger.debug("Failed to clear scroll %s", scroll_id)


def fetch_source_documents(
    document_ids: List[str],
    index_name: Optional[str] = None,
) -> DocumentFetchResult:
    """Load documents by id from the input index. Blank-text docs are skipped, not failed."""
    index_name = index_name or get_settings().input_index
    client = get_client()
    result = DocumentFetchResult()

    for hit in _scroll_hits(client, index_name, _document_id_query(document_ids)):
        src = hit.get("_source", {})
        meta = src.get("file_metadata") or {}
        document_id = meta.get("document_id") or src.get("document_id")
        if document_id is not None:
            result.matched_ids.add(str(document_id))

        text = (
            (src.get("content") or {}).get("note_text")
            or src.get("text")
            or src.get("note_text")
            or ""
        )
        if not str(text).strip():
            result.skipped_count += 1
            continue

        result.docs.append(
            {
                "row_id": hit["_id"],
                "document_id": document_id,
                "patient_id": meta.get("patient_id") or src.get("patient_id"),
                "visit_id": meta.get("visit_id") or src.get("visit_id"),
                "job_id": src.get("job_id"),
                "batch_id": src.get("batch_id"),
                "source_index": index_name,
                "text": str(text),
            }
        )

    return result


RESULTS_MAPPING = {
    "mappings": {
        "properties": {
            "result_id": {"type": "keyword"},
            "session_id": {"type": "keyword"},
            "row_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "patient_id": {"type": "keyword"},
            "visit_id": {"type": "keyword"},
            "job_id": {"type": "keyword"},
            "batch_id": {"type": "keyword"},
            "pipeline_id": {"type": "keyword"},
            "engine": {"type": "keyword"},
            "label": {"type": "keyword"},
            "span_text": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "sentence": {"type": "keyword"},
            "confidence": {"type": "float"},
            "source_index": {"type": "keyword"},
            "source_mode": {"type": "keyword"},
            "raw_metadata": {"type": "object", "enabled": True},
            "created_at": {"type": "date"},
            "job_details": {"type": "text"},
        }
    }
}


def mirror_results(docs: List[Dict[str, Any]]) -> int:
    """Mirror result rows into ES, if RESULTS_ES_WRITE is on. A no-op otherwise.

    Owning the flag here keeps the caller from having to know that ES writes are optional.
    """
    settings = get_settings()
    if not (settings.results_es_write and docs):
        return 0

    client = get_client()
    index = settings.results_index
    if not client.indices.exists(index=index):
        logger.info("Creating results index %s", index)
        client.indices.create(index=index, body=RESULTS_MAPPING)

    # One refresh at the end rather than per chunk: refreshing every 250 rows forces a Lucene
    # segment refresh per chunk, which is cluster-wide work for no benefit mid-write.
    written, _ = helpers.bulk(
        client,
        (
            {"_index": index, "_id": doc.get("result_id"), "_source": doc}
            for doc in docs
        ),
        chunk_size=250,
        refresh=False,
    )
    client.indices.refresh(index=index)
    return written
