"""Request/response models.

Response shapes mirror custom_nlp_service so existing clients (and
nlp-services-multitask-zeroshot-test-nb.ipynb) need no changes.

Config normalization happens in exactly one place -- ZeroShotConfig.to_annotator_config() ->
zeroshot_dsl.normalize_config(). The reference splits it between a `before` validator in
models.py and normalize_* in runtime.py, which is how the two drifted apart from the notebook.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zeroshot_dsl import DEFAULT_THRESHOLD, normalize_config, validate_annotator_config


class InlineDocument(BaseModel):
    text: str = Field(..., min_length=1)
    document_id: Optional[str] = None
    row_id: Optional[str] = None
    patient_id: Optional[str] = None
    visit_id: Optional[str] = None
    job_id: Optional[str] = None
    batch_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_string(cls, value: Any) -> Any:
        """Allow "documents": ["some text", ...] alongside the full object form."""
        return {"text": value} if isinstance(value, str) else value


class ZeroShotConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: Optional[str] = None
    entities: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    structures: List[Any] = Field(default_factory=list)
    classifications: List[Any] = Field(default_factory=list)
    relations: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    entity_threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    structure_threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    classification_threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    relation_threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _accept_entity_aliases(cls, values: Any) -> Any:
        if isinstance(values, dict) and not values.get("entities"):
            for alias in ("selected_labels", "supported_labels"):
                if values.get(alias):
                    values["entities"] = values.pop(alias)
                    break
        return values

    @model_validator(mode="after")
    def _validate(self) -> "ZeroShotConfig":
        if not (self.entities or self.structures or self.classifications or self.relations):
            raise ValueError(
                "zero_shot must define at least one task: entities, structures, "
                "classifications, or relations."
            )
        # Fail here rather than minutes later inside the worker.
        validate_annotator_config(self.to_annotator_config())
        return self

    def to_annotator_config(self) -> Dict[str, Any]:
        return normalize_config(self.model_dump(exclude={"model_id"}, mode="json"))


class RunRequest(BaseModel):
    documents: List[InlineDocument] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    input_index: Optional[str] = None
    zero_shot: ZeroShotConfig
    job_details: Optional[str] = None
    job_name: Optional[str] = None
    user_id: Optional[str] = None
    dataset_id: Optional[str] = None

    @model_validator(mode="after")
    def _need_a_document_source(self) -> "RunRequest":
        if not self.documents and not self.document_ids:
            raise ValueError(
                "Provide 'documents' (inline text) and/or 'document_ids' "
                "(fetched from Elasticsearch)."
            )
        return self


class QueuedSessionResponse(BaseModel):
    status: str
    session_id: str
    status_url: str
    results_url: str
    logs_url: str


class SessionStatusResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str
    status: str
    progress: int = 0
    message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    stage: Optional[str] = None
    current_stage: Optional[str] = None
    stage_details: Dict[str, Any] = Field(default_factory=dict)
    documents_loaded: int = 0
    documents_completed: int = 0
    documents_skipped: int = 0
    results_written: int = 0
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class SessionResultsResponse(BaseModel):
    session_id: str
    total: int
    limit: int
    offset: int
    results: List[Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    engine: str
    model_loaded: bool
    tasks: List[str]
    supported_thresholds: List[str]
    defaults: Dict[str, float]
