"""Single-worker session queue.

One daemon thread drains a stdlib queue, so sessions execute strictly one at a time. That is not
a throughput choice -- PretrainedZeroShotMultiTask is reconfigured in place per request, so
concurrent runs would corrupt each other (see multitask_model). Combined with uvicorn
--workers 1, this makes the whole service single-threaded where it counts.

Head-of-line blocking is the accepted cost: a large session delays every queued session behind
it. /health reports queue depth so that is at least visible.
"""

from __future__ import annotations

import queue
import threading
import uuid
from typing import Any, Dict, Optional

import multitask_model
from logging_setup import logger
from runner import SessionRunner
from settings import get_settings
from store import SessionStore


class SessionQueueManager:
    def __init__(self, store: SessionStore):
        self.store = store
        self.runner = SessionRunner(store=store)
        self._queue: "queue.Queue[tuple[str, Dict[str, Any]]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # --- lifecycle -----------------------------------------------------------------------
    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._work, name="session-worker", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()

    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def queue_size(self) -> int:
        return self._queue.qsize()

    # --- api -----------------------------------------------------------------------------
    def submit(self, session_doc: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = session_doc["session_id"]
        self.store.initialize_session(session_doc)
        self._queue.put((session_id, payload))
        logger.info("Queued session %s (depth=%d)", session_id, self._queue.qsize())
        return session_doc

    @staticmethod
    def new_session_id() -> str:
        return str(uuid.uuid4())

    # --- worker --------------------------------------------------------------------------
    def _work(self) -> None:
        # Pay the model load once, at boot, rather than making the first caller wait for it.
        try:
            multitask_model.preload()
        except Exception as exc:
            logger.exception("Model preload failed; sessions will fail until this is fixed: %s", exc)

        poll_seconds = get_settings().queue_poll_seconds
        while not self._stop.is_set():
            try:
                session_id, payload = self._queue.get(timeout=poll_seconds)
            except queue.Empty:
                continue

            try:
                with self.store.session_logging(session_id):
                    logger.info("Starting session %s", session_id)
                    self.runner.run(session_id, payload)
                    logger.info("Finished session %s", session_id)
            except Exception as exc:
                logger.exception("Session %s failed: %s", session_id, exc)
                self.store.append_error(session_id, str(exc))
                self.store.update_status(
                    session_id,
                    {"status": "failed", "progress": 100, "message": f"Session failed: {exc}"},
                )
            finally:
                self._queue.task_done()
