"""In-memory session state: status, result rows, and logs.

Nothing is written to disk. A restart drops every session, so a client polling an id from before
the restart gets a 404 rather than a status. For durable results, set RESULTS_ES_WRITE=true and
read them from Elasticsearch.

All three kinds of per-session state live under one key in one dict, on purpose. When they were
separate stores, eviction had to remember to clear each one -- and the first version of this file
shipped with a bug where expiry dropped session status but orphaned the result rows. One entry
per session makes eviction whole-by-construction.

custom_nlp_service uses Redis for status so several uvicorn workers can share it. This service is
pinned to --workers 1 by the annotator (see multitask_model), so a dict under one lock does the
same job with no sidecar and no serialization.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Deque, Dict, Iterator, List, Optional

from logging_setup import logger
from settings import get_settings

DEFAULT_STAGES = ("load_documents", "extract_annotations", "persist_results")

# A stage's progress follows from its status, so callers only pass the status.
STAGE_PROGRESS = {"pending": 0, "processing": 0, "completed": 100, "failed": 100}

# Spark and the JSL annotator are chatty, so a session's log is capped rather than unbounded.
LOG_LINES_PER_SESSION = 2000


def default_stage_details() -> Dict[str, Dict[str, Any]]:
    return {stage: {"status": "pending", "progress": 0} for stage in DEFAULT_STAGES}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class _Session:
    status: Dict[str, Any]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=LOG_LINES_PER_SESSION))


class _DequeLogHandler(logging.Handler):
    """Feeds formatted log records into one session's ring buffer."""

    def __init__(self, buffer: Deque[str]):
        super().__init__(level=logging.INFO)
        self.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buffer.append(self.format(record))
        except Exception:  # noqa: BLE001 - logging must never break the run
            self.handleError(record)


class SessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        # Ordered by creation, so evicting the oldest is popitem(last=False).
        self._sessions: "OrderedDict[str, _Session]" = OrderedDict()

    # --- lifecycle -----------------------------------------------------------------------
    def initialize_session(self, session_doc: Dict[str, Any]) -> None:
        payload = deepcopy(session_doc)
        payload.setdefault("stage_details", default_stage_details())
        payload.setdefault("progress", 0)
        payload.setdefault("status", "queued")
        payload.setdefault("created_at", _now())
        payload.setdefault("warnings", [])
        payload.setdefault("errors", [])
        payload["updated_at"] = _now()

        with self._lock:
            self._sessions[payload["session_id"]] = _Session(status=payload)
            self._evict_locked()

    def _evict_locked(self) -> None:
        """Drop the oldest sessions past the history limit. Caller holds the lock.

        Memory is bounded by (retained sessions x their rows), so a run over thousands of
        documents is worth keeping few of.
        """
        limit = max(1, get_settings().session_history_limit)
        while len(self._sessions) > limit:
            evicted_id, _ = self._sessions.popitem(last=False)
            logger.info("Evicted session %s (history limit %d reached)", evicted_id, limit)

    def _get_locked(self, session_id: str) -> Optional[_Session]:
        return self._sessions.get(session_id)

    def session_exists(self, session_id: str) -> bool:
        """For callers that only need the 404 check, without copying the status out."""
        with self._lock:
            return session_id in self._sessions

    # --- status --------------------------------------------------------------------------
    # The mutators below return nothing: an evicted session simply has nothing to update, and
    # nobody can read it anyway. Only get_status hands data out, so only it copies.
    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            session = self._get_locked(session_id)
            return deepcopy(session.status) if session else None

    def update_status(self, session_id: str, patch: Dict[str, Any]) -> None:
        with self._lock:
            session = self._get_locked(session_id)
            if session is None:
                return
            session.status.update(patch)
            session.status["updated_at"] = _now()

    def update_stage(
        self,
        session_id: str,
        *,
        stage_name: str,
        stage_status: str,
        progress: int,
        message: Optional[str] = None,
        **fields: Any,
    ) -> None:
        with self._lock:
            session = self._get_locked(session_id)
            if session is None:
                return
            stage_details = session.status.setdefault("stage_details", default_stage_details())
            stage_details[stage_name] = {
                "status": stage_status,
                "progress": STAGE_PROGRESS.get(stage_status, 0),
            }
            # `stage` duplicates `current_stage` to match custom_nlp_service's response shape.
            session.status.update(
                {"stage": stage_name, "current_stage": stage_name, "progress": progress, **fields}
            )
            if message:
                session.status["message"] = message
            session.status["updated_at"] = _now()

        logger.info(
            "Session %s stage=%s stage_status=%s progress=%s%s",
            session_id,
            stage_name,
            stage_status,
            progress,
            f" message={message}" if message else "",
        )

    def _append(self, session_id: str, key: str, value: str) -> None:
        with self._lock:
            session = self._get_locked(session_id)
            if session is None:
                return
            session.status.setdefault(key, []).append(value)
            session.status["updated_at"] = _now()

    def append_warning(self, session_id: str, warning: str) -> None:
        self._append(session_id, "warnings", warning)

    def append_error(self, session_id: str, error: str) -> None:
        self._append(session_id, "errors", error)

    # --- results -------------------------------------------------------------------------
    def write_results(self, session_id: str, rows: List[Dict[str, Any]]) -> int:
        with self._lock:
            session = self._get_locked(session_id)
            if session is None:
                return 0
            session.rows.extend(rows)
            return len(rows)

    def count_results(self, session_id: str) -> int:
        with self._lock:
            session = self._get_locked(session_id)
            return len(session.rows) if session else 0

    def fetch_results(self, session_id: str, *, limit: int, offset: int) -> List[Dict[str, Any]]:
        """One page of rows, insertion-ordered.

        Rows are write-once (nothing mutates them after write_results), so the slice hands out
        the stored dicts rather than copies of them.

        Insertion order matters: custom_nlp_service pages ES results sorted by created_at, but
        every row in a session shares one created_at, so that sort is non-deterministic and rows
        can repeat or vanish across offset windows.
        """
        with self._lock:
            session = self._get_locked(session_id)
            if session is None:
                return []
            return session.rows[offset : offset + limit]

    # --- logs ----------------------------------------------------------------------------
    @contextmanager
    def session_logging(self, session_id: str) -> Iterator[None]:
        """Capture everything logged during a run into this session's ring buffer.

        Attached to the root logger, so Spark and JSL output is captured too -- which is most of
        what makes these logs worth reading.
        """
        with self._lock:
            session = self._get_locked(session_id)
            buffer = session.log if session else deque(maxlen=LOG_LINES_PER_SESSION)

        handler = _DequeLogHandler(buffer)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            yield
        finally:
            root_logger.removeHandler(handler)
            handler.close()

    def read_log(self, session_id: str, *, tail: Optional[int] = 200, full: bool = False) -> Optional[str]:
        """None when the session is unknown -- same convention as get_status."""
        with self._lock:
            session = self._get_locked(session_id)
            if session is None:
                return None
            lines = list(session.log)
        if not full:
            lines = lines[-max(1, int(tail or 200)) :]
        return "\n".join(lines)
