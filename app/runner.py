"""Executes one session: load documents -> transform -> flatten to rows -> persist.

The three stages mirror custom_nlp_service's, so /status reports the same shape clients already
poll. Row assembly is delegated to rows.to_service_rows -- the same function the notebook calls,
which is the whole point of this service existing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import documents
import es_client
import multitask_model
from documents import DOC_COLUMNS, DOC_SCHEMA
from logging_setup import logger
from rows import COMMON_KEYS, to_service_rows
from runtime import get_spark
from store import SessionStore


class SessionRunner:
    def __init__(self, store: SessionStore):
        self.store = store

    def run(self, session_id: str, payload: Dict[str, Any]) -> None:
        zero_shot = payload["zero_shot"]
        # main.py already resolved model_id against the pinned model before queueing.
        model_id = zero_shot["model_id"]

        docs = self._load_documents(session_id, payload)
        if not docs:
            self.store.update_stage(
                session_id,
                stage_name="load_documents",
                stage_status="completed",
                progress=100,
                status="completed",
                message="No documents with text to process.",
                documents_loaded=0,
                documents_completed=0,
                results_written=0,
            )
            return

        rows = self._extract(session_id, docs, zero_shot["config"], model_id, payload)
        self._persist(session_id, rows)

    # --- stage 1 -------------------------------------------------------------------------
    def _load_documents(self, session_id: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.store.update_stage(
            session_id,
            stage_name="load_documents",
            stage_status="processing",
            progress=5,
            status="processing",
            message="Loading documents.",
        )

        loaded = documents.load(payload)
        for warning in loaded.warnings:
            self.store.append_warning(session_id, warning)

        self.store.update_stage(
            session_id,
            stage_name="load_documents",
            stage_status="completed",
            progress=25,
            message=f"Loaded {len(loaded.docs)} document(s).",
            documents_loaded=len(loaded.docs),
            documents_skipped=loaded.skipped,
        )
        return loaded.docs

    # --- stage 2 -------------------------------------------------------------------------
    def _extract(
        self,
        session_id: str,
        docs: List[Dict[str, Any]],
        config: Dict[str, Any],
        model_id: str,
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        self.store.update_stage(
            session_id,
            stage_name="extract_annotations",
            stage_status="processing",
            progress=30,
            message=f"Running {model_id} over {len(docs)} document(s).",
        )

        created_at = datetime.now(timezone.utc).isoformat()
        frame = get_spark().createDataFrame(docs, schema=DOC_SCHEMA)
        rows: List[Dict[str, Any]] = []

        # Rows are drained inside the `with`: the annotator is shared and Spark is lazy, so the
        # model is only safe to act on while configured() holds its lock.
        with multitask_model.configured(config, model_id) as model:
            # The sentence column rides along so rows.py can turn each annotation's
            # metadata["sentence"] index into the real sentence text.
            transformed = model.transform(frame).select(
                *DOC_COLUMNS[:-1], "sentence", "extractions"
            )
            for row in transformed.toLocalIterator():
                rows.extend(
                    to_service_rows(
                        row["extractions"],
                        pipeline_id=model_id,
                        source_index=row["source_index"],
                        common={key: row[key] for key in COMMON_KEYS},
                        sentences=row["sentence"],
                        created_at=created_at,
                        session_id=session_id,
                        job_details=payload.get("job_details"),
                        with_result_id=True,
                    )
                )

        self.store.update_stage(
            session_id,
            stage_name="extract_annotations",
            stage_status="completed",
            progress=75,
            message=f"Extracted {len(rows)} annotation row(s).",
            documents_completed=len(docs),
        )
        return rows

    # --- stage 3 -------------------------------------------------------------------------
    def _persist(self, session_id: str, rows: List[Dict[str, Any]]) -> None:
        self.store.update_stage(
            session_id,
            stage_name="persist_results",
            stage_status="processing",
            progress=80,
            message="Persisting results.",
        )

        written = self.store.write_results(session_id, rows)

        try:
            es_client.mirror_results(rows)
        except Exception as exc:
            # The rows are already readable from this service, so mirroring is best-effort.
            logger.warning("Failed to mirror results to Elasticsearch: %s", exc)
            self.store.append_warning(session_id, f"Elasticsearch mirroring failed: {exc}")

        self.store.update_stage(
            session_id,
            stage_name="persist_results",
            stage_status="completed",
            progress=100,
            status="completed",
            message="Session completed successfully.",
            results_written=written,
        )
