r"""The single PretrainedZeroShotMultiTask instance this container owns.

READ THIS BEFORE CHANGING ANYTHING HERE.

PretrainedZeroShotMultiTask has a JVM-level defect: constructing a SECOND instance in the same
Spark session corrupts it. The 2nd+ model's constructSchema reads a java.util.ArrayList where it
expects Tuple2[], throwing ClassCastException on EVERY subsequent .transform() -- not just the
new model's. Only a restart clears it. It is unrelated to payload contents; purely ">=2 instances
per JVM". `.copy()` is not a workaround: it returns an object with uninitialized weights.

So: load ONCE per JVM, keep that instance, and reconfigure its params per request. Passing empty
lists is how a previous request's structures/classifications get cleared. This mirrors
build_multitask_pipeline.ipynb, which also loads the annotator once and reuses it.

Invariant worth enforcing in review -- this must name only this file:
    grep -rlE 'import .*PretrainedZeroShotMultiTask|PretrainedZeroShotMultiTask\.pretrained' app/
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock, RLock
from typing import Any, Dict, Iterator, List, Optional

from pyspark.ml import Pipeline
from sparknlp.annotator import SentenceDetectorDLModel
from sparknlp.base import DocumentAssembler
from sparknlp_jsl.annotator import PretrainedZeroShotMultiTask

from logging_setup import logger
from runtime import get_spark
from settings import get_settings

_LOAD_LOCK = Lock()

# Held across reconfigure -> fit -> transform -> row materialization, by configured() below.
# Spark is lazy: a reconfigure racing an in-flight toLocalIterator() would silently apply the
# wrong params to an already-planned job. Reentrant so callers can nest helpers under it.
_EXEC_LOCK = RLock()

_STATE: Dict[str, Any] = {
    "annotator": None,
    "pretrained_name": None,
    "sentence_detector": None,
    "load_error": None,
}

# setClassificationThreshold / setRelationThreshold exist in spark-nlp-jsl 6.4.1 but are not used
# by custom_nlp_service, so they are the likeliest setters to be missing on another build. Probed
# once at load and reported by GET /api/v1/model rather than exploding per request.
THRESHOLD_SETTERS = {
    "entity_threshold": "setEntityThreshold",
    "structure_threshold": "setStructureThreshold",
    "classification_threshold": "setClassificationThreshold",
    "relation_threshold": "setRelationThreshold",
}

TASK_SETTERS = {
    "entities": "setEntities",
    "structures": "setStructures",
    "classifications": "setClassifications",
    "relations": "setRelations",
}


class ModelMismatchError(RuntimeError):
    """A request asked for a model this container is not pinned to."""

    def __init__(self, pinned: str, requested: str):
        self.pinned = pinned
        self.requested = requested
        super().__init__(
            f"This container serves '{pinned}'. Model '{requested}' cannot be served: "
            "PretrainedZeroShotMultiTask can only be instantiated once per JVM, so each "
            f"container serves exactly one model. Run a second container with "
            f"MULTITASK_MODEL_ID={requested}, or restart this one with that value."
        )


def check_model_id(requested: Optional[str]) -> str:
    """Resolve a requested model_id against the model this container serves.

    Checked against the ALREADY-LOADED annotator once there is one, and only against the env
    setting before that. The loaded instance is the real constraint -- it is the thing that
    cannot be replaced -- so an env value that has drifted from it must not be what decides
    whether a request is safe.
    """
    pinned = _STATE["pretrained_name"] or get_settings().model_id
    if requested and requested != pinned:
        raise ModelMismatchError(pinned, requested)
    return pinned


def is_loaded() -> bool:
    return _STATE["annotator"] is not None


def load_error() -> Optional[str]:
    """The last preload failure, so /health can say "degraded" instead of "warming" forever."""
    return _STATE["load_error"]


def supported_thresholds() -> List[str]:
    annotator = _STATE["annotator"]
    if annotator is None:
        return list(THRESHOLD_SETTERS)
    return [key for key, setter in THRESHOLD_SETTERS.items() if hasattr(annotator, setter)]


def get_annotator(pretrained_name: Optional[str] = None):
    """Return this JVM's one PretrainedZeroShotMultiTask, loading it on first call."""
    pretrained_name = check_model_id(pretrained_name)

    if _STATE["annotator"] is not None:
        return _STATE["annotator"]

    with _LOAD_LOCK:
        if _STATE["annotator"] is not None:
            return _STATE["annotator"]

        get_spark()
        logger.info("Loading PretrainedZeroShotMultiTask '%s' (once per JVM)", pretrained_name)
        try:
            annotator = PretrainedZeroShotMultiTask.pretrained(
                pretrained_name, "en", "clinical/models"
            )
        except Exception as exc:
            # Remembered so /health can distinguish "still loading" from "will never load".
            _STATE["load_error"] = f"{type(exc).__name__}: {exc}"
            raise
        _STATE["annotator"] = annotator
        _STATE["pretrained_name"] = pretrained_name
        _STATE["load_error"] = None

        missing = set(THRESHOLD_SETTERS) - set(supported_thresholds())
        if missing:
            logger.warning(
                "Annotator '%s' lacks threshold setters %s; those thresholds will be ignored",
                pretrained_name,
                sorted(missing),
            )
        logger.info("Annotator '%s' loaded", pretrained_name)
        return annotator


def _sentence_detector():
    """The DL sentence detector, also cached once per JVM (it is a pretrained model too)."""
    if _STATE["sentence_detector"] is None:
        with _LOAD_LOCK:
            if _STATE["sentence_detector"] is None:
                name = get_settings().sentence_detector_model
                logger.info("Loading sentence detector '%s' (once per JVM)", name)
                _STATE["sentence_detector"] = (
                    SentenceDetectorDLModel.pretrained(name, "en", "clinical/models")
                    .setInputCols(["document"])
                    .setOutputCol("sentence")
                )
    return _STATE["sentence_detector"]


def preload() -> None:
    """Warm the annotator + sentence detector so the first request does not pay for the load."""
    get_annotator()
    _sentence_detector()


@contextmanager
def configured(config: Dict[str, Any], pretrained_name: Optional[str] = None) -> Iterator[Any]:
    """Yield a PipelineModel configured for `config`, holding the exec lock throughout.

    The lock is taken here rather than by callers because it is half of the single-instance
    secret this module owns: the annotator is mutated in place, and Spark is lazy, so the lock
    has to outlive not just the reconfigure but every action taken on the returned model. Stay
    inside the `with` until the rows are fully materialized:

        with multitask_model.configured(cfg, model_id) as model:
            rows = list(model.transform(frame).toLocalIterator())   # drained inside

    `config` is already in annotator DSL (see zeroshot_dsl.normalize_config).
    """
    with _EXEC_LOCK:
        yield _build_pipeline_model(config, pretrained_name)


def _build_pipeline_model(config: Dict[str, Any], pretrained_name: Optional[str] = None):
    """Mirror of the notebook's build_multitask_pipeline(**normalize_config(cfg))."""
    annotator = get_annotator(pretrained_name)

    # The annotator consumes `sentence` directly; no tokenizer stage.
    zero_shot = annotator.setInputCols(["sentence"]).setOutputCol("extractions")

    # Every task param is set on every request -- empty lists clear the previous payload's
    # config. Skipping a setter would leak the last request's entities into this one.
    for key, setter in TASK_SETTERS.items():
        getattr(zero_shot, setter)(config.get(key) or [])
    for key, setter in THRESHOLD_SETTERS.items():
        if hasattr(zero_shot, setter):
            getattr(zero_shot, setter)(config[key])

    pipeline = Pipeline(
        stages=[
            DocumentAssembler().setInputCol("text").setOutputCol("document"),
            _sentence_detector(),
            zero_shot,
        ]
    )
    return pipeline.fit(get_spark().createDataFrame([[""]]).toDF("text"))
