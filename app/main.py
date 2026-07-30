"""Standalone multitask zero-shot NLP service.

POST /api/v1/runs -> 202 + session_id; poll /api/v1/status/{id}; GET /api/v1/results/{id}.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

import es_client
import multitask_model
from logging_setup import logger
from models import (
    ModelInfoResponse,
    QueuedSessionResponse,
    RunRequest,
    SessionResultsResponse,
    SessionStatusResponse,
)
from queue_manager import SessionQueueManager
from rows import ENGINE_MULTITASK, TASK_TYPES
from runtime import spark_is_initialized
from settings import get_settings
from store import SessionStore
from zeroshot_dsl import DEFAULT_THRESHOLD, THRESHOLD_KEYS

store = SessionStore()
queue_manager = SessionQueueManager(store=store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sessions are in-memory, so a restart leaves none behind to reconcile.
    queue_manager.start()
    yield
    queue_manager.stop()


app = FastAPI(title="Multitask Zero-Shot NLP Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "service": "Multitask Zero-Shot NLP Service",
        "version": "0.1.0",
        "model_id": get_settings().model_id,
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> JSONResponse:
    settings = get_settings()
    worker_alive = queue_manager.worker_alive()
    model_loaded = multitask_model.is_loaded()
    load_error = multitask_model.load_error()

    if settings.es_enabled:
        es_status = "ok" if es_client.ping() else "down"
    else:
        es_status = "disabled"

    # "warming" is healthy: the model load takes minutes on a cold cache, and 503-ing through it
    # would let Docker kill the container mid-load. A load that has actually failed is degraded,
    # though -- otherwise a bad license reads as "warming" forever.
    if not worker_alive or load_error:
        status = "degraded"
    elif not model_loaded:
        status = "warming"
    else:
        status = "ok"

    payload = {
        "status": status,
        "service": "nlp-services-standalone",
        "model_id": settings.model_id,
        "model_loaded": model_loaded,
        "spark_initialized": spark_is_initialized(),
        "queue": {"worker_alive": worker_alive, "depth": queue_manager.queue_size()},
        "dependencies": {"elasticsearch": es_status},
    }
    if load_error:
        payload["load_error"] = load_error
    return JSONResponse(status_code=200 if status != "degraded" else 503, content=payload)


@app.get("/api/v1/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_id=get_settings().model_id,
        engine=ENGINE_MULTITASK,
        model_loaded=multitask_model.is_loaded(),
        tasks=list(TASK_TYPES),
        supported_thresholds=multitask_model.supported_thresholds(),
        defaults={key: DEFAULT_THRESHOLD for key in THRESHOLD_KEYS},
    )


@app.post("/api/v1/runs", response_model=QueuedSessionResponse, status_code=202)
def create_run(payload: RunRequest) -> QueuedSessionResponse:
    settings = get_settings()

    try:
        model_id = multitask_model.check_model_id(payload.zero_shot.model_id)
    except multitask_model.ModelMismatchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if payload.document_ids and not settings.es_enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "'document_ids' requires Elasticsearch, which is disabled (ES_ENABLED=false). "
                "Send inline 'documents' instead, or enable ES."
            ),
        )

    session_id = queue_manager.new_session_id()
    config = payload.zero_shot.to_annotator_config()

    session_doc: Dict[str, Any] = {
        "session_id": session_id,
        "status": "queued",
        "message": "Session queued.",
        "input_index": payload.input_index or settings.input_index,
        "documents_requested": len(payload.documents) + len(payload.document_ids),
        "document_ids": payload.document_ids,
        "job_details": payload.job_details,
        "job_name": payload.job_name,
        "user_id": payload.user_id,
        "dataset_id": payload.dataset_id,
        "zero_shot": {"model_id": model_id, **config},
        "queue_depth_on_submit": queue_manager.queue_size(),
    }

    queue_manager.submit(
        session_doc,
        {
            "documents": [doc.model_dump() for doc in payload.documents],
            "document_ids": payload.document_ids,
            "input_index": payload.input_index,
            "job_details": payload.job_details,
            "zero_shot": {"model_id": model_id, "config": config},
        },
    )
    logger.info("Accepted session %s for model %s", session_id, model_id)

    return QueuedSessionResponse(
        status="queued",
        session_id=session_id,
        status_url=settings.route_url(f"/api/v1/status/{session_id}"),
        results_url=settings.route_url(f"/api/v1/results/{session_id}"),
        logs_url=settings.route_url(f"/api/v1/logs/{session_id}"),
    )


def _not_found(session_id: str) -> HTTPException:
    """The store returns None for an unknown session; the API turns that into one 404."""
    return HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


@app.get("/api/v1/status/{session_id}", response_model=SessionStatusResponse)
def get_status(session_id: str) -> SessionStatusResponse:
    status = store.get_status(session_id)
    if status is None:
        raise _not_found(session_id)
    return SessionStatusResponse(**status)


@app.get("/api/v1/results/{session_id}", response_model=SessionResultsResponse)
def get_results(
    session_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> SessionResultsResponse:
    if not store.session_exists(session_id):
        raise _not_found(session_id)
    return SessionResultsResponse(
        session_id=session_id,
        total=store.count_results(session_id),
        limit=limit,
        offset=offset,
        results=store.fetch_results(session_id, limit=limit, offset=offset),
    )


@app.get("/api/v1/logs/{session_id}")
def get_logs(
    session_id: str,
    tail: int = Query(default=200, ge=1, le=5000),
    full: bool = Query(default=False),
) -> PlainTextResponse:
    content = store.read_log(session_id, tail=tail, full=full)
    if content is None:
        raise _not_found(session_id)
    return PlainTextResponse(content)
