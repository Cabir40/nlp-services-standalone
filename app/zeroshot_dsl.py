"""Zero-shot multitask config -> PretrainedZeroShotMultiTask annotator DSL.

SHARED by the service (``runner.py``) and ``build_multitask_pipeline.ipynb``. Import it from
both rather than copying it -- the notebook and the service drifting apart is exactly what this
module exists to prevent.

Stdlib only, deliberately: the notebook imports this from the host, where pyspark/pydantic may
be different versions than the container's.

The four task DSLs the annotator accepts:

    entities         setEntities        ["LABEL::dtype::description", ...]
    structures       setStructures      [("name", ["field::dtype::desc", "field::[a|b|c]"]), ...]
    classifications  setClassifications [("task", ["label1", "label2"]), ...]
    relations        setRelations       ["subject_verb_object", ...]
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_THRESHOLD = 0.6

THRESHOLD_KEYS = (
    "entity_threshold",
    "structure_threshold",
    "classification_threshold",
    "relation_threshold",
)


def _join(*parts: Any) -> str:
    return "::".join(str(p) for p in parts if p)


def normalize_entities(entities: Iterable[Any]) -> List[str]:
    """[{label, dtype, description}] or ["LABEL"] -> ["LABEL::dtype::description", ...].

    Input order and casing are preserved. custom_nlp_service's normalize_labels_with_desc()
    uppercases and sorts instead; since these specs are assembled into the model's prompt,
    reordering them is not an output-neutral change. The notebook is the reference here.
    """
    out: List[str] = []
    seen = set()
    for entity in entities or []:
        if isinstance(entity, str):
            spec = entity.strip()
        else:
            spec = _join(
                entity.get("label") or entity.get("ner_label") or entity.get("name"),
                entity.get("dtype"),
                entity.get("description") or entity.get("label_description"),
            )
        if spec and spec not in seen:
            seen.add(spec)
            out.append(spec)
    return out


def _field(field: Any) -> str:
    if isinstance(field, str):
        return field
    choices = field.get("choices") or field.get("options")
    if choices:
        return f"{field['name']}::[{'|'.join(str(c) for c in choices)}]"
    return _join(field["name"], field.get("dtype"), field.get("description"))


def normalize_structures(structures: Iterable[Any]) -> List[Tuple[str, List[str]]]:
    """[{name, fields}] -> [("name", ["field::dtype::desc", ...]), ...].

    Already-normalized ``(name, [spec, ...])`` pairs pass through, so a caller may hand back
    this function's own output.
    """
    out: List[Tuple[str, List[str]]] = []
    for structure in structures or []:
        if isinstance(structure, (list, tuple)) and len(structure) == 2:
            name, fields = structure
        else:
            name, fields = structure.get("name", ""), structure.get("fields") or []
        out.append((str(name), [_field(f) for f in fields]))
    return out


def normalize_classifications(classifications: Iterable[Any]) -> List[Tuple[str, List[str]]]:
    """[{task, labels}] -> [("task", ["label1", ...]), ...]."""
    out: List[Tuple[str, List[str]]] = []
    for classification in classifications or []:
        if isinstance(classification, (list, tuple)) and len(classification) == 2:
            task, labels = classification
        else:
            task = classification.get("task") or classification.get("name", "")
            labels = classification.get("labels") or []
        out.append((str(task), [str(label) for label in labels]))
    return out


def normalize_relations(relations: Iterable[Any]) -> List[str]:
    """"MEDICATION treats PROBLEM" -> "MEDICATION_treats_PROBLEM".

    setRelations() takes single-token names. custom_nlp_service passes the spaced form straight
    through; the underscore form is what the notebook and JSL's own 1.8 demo use.
    """
    out: List[str] = []
    for relation in relations or []:
        if isinstance(relation, dict):
            out.append("_".join([relation["subject"], relation["relation"], relation["object"]]))
        else:
            out.append("_".join(str(relation).split()))
    return out


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a rich request config into build_multitask_pipeline()'s exact kwargs."""
    return {
        "entities": normalize_entities(config.get("entities")),
        "structures": normalize_structures(config.get("structures")),
        "classifications": normalize_classifications(config.get("classifications")),
        "relations": normalize_relations(config.get("relations")),
        **{
            key: _threshold(config.get(key)) for key in THRESHOLD_KEYS
        },
    }


def _threshold(value: Any) -> float:
    return DEFAULT_THRESHOLD if value is None else float(value)


def validate_annotator_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Assert normalize_config() output is well-formed; raise ValueError if not.

    Called from the request model so a malformed config fails at POST rather than minutes
    later inside the worker. The notebook calls it in place of its inline asserts.
    """
    if set(config) != {"entities", "structures", "classifications", "relations", *THRESHOLD_KEYS}:
        raise ValueError(
            f"normalize_config() produced {sorted(config)}, which does not match "
            f"build_multitask_pipeline()'s parameters"
        )

    if not all(isinstance(e, str) and e for e in config["entities"]):
        raise ValueError(f"entities must be non-empty strings: {config['entities']}")

    for name, fields in config["structures"]:
        if not (isinstance(name, str) and name):
            raise ValueError(f"structure name must be a non-empty string: {name!r}")
        if not fields or not all(isinstance(f, str) and f for f in fields):
            raise ValueError(f"structure '{name}' needs at least one non-empty field spec")

    for task, labels in config["classifications"]:
        if not (isinstance(task, str) and task):
            raise ValueError(f"classification task must be a non-empty string: {task!r}")
        if not labels or not all(isinstance(l, str) and l for l in labels):
            raise ValueError(f"classification '{task}' needs at least one non-empty label")

    for relation in config["relations"]:
        if not (isinstance(relation, str) and relation):
            raise ValueError(f"relation must be a non-empty string: {relation!r}")
        if " " in relation:
            raise ValueError(
                f"relation names passed to setRelations() must be single tokens "
                f"(underscore-joined): {relation!r}"
            )

    for key in THRESHOLD_KEYS:
        value = config[key]
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be a float in [0, 1]: {value!r}")

    return config
