# nlp-services-standalone

`build_multitask_pipeline.ipynb` as a Docker service. POST a zero-shot config plus documents, poll
a session, get back flat JSON annotation rows.

One JSL `PretrainedZeroShotMultiTask` model runs four tasks in a single pass — NER, classification,
structured extraction, and relation extraction — and all four land in one `extractions` column that
this service flattens into one row per annotation.

## Why this exists

`jsl-omop/custom_nlp_service` already runs this model, but it is bound to Elasticsearch, Redis,
`pj-lib`, and the TPJ Bronze pipeline, and it serves several other engines besides. This service is
a trimmed fork: multitask only, ES optional, no Redis, no private packages.

More importantly, section 6 of `build_multitask_pipeline.ipynb` carried a hand-copied duplicate of
the service's row-assembly logic, with the comment *"keep the two in sync (or import a shared module
once one exists)"*. `app/zeroshot_dsl.py` and `app/rows.py` are that module — stdlib-only so the
notebook can import them straight off the host:

```python
import sys; sys.path.insert(0, ".../nlp-services-standalone/app")
from zeroshot_dsl import normalize_config
from rows import to_service_rows
```

They had already drifted. The notebook underscore-joins relation names (`"MEDICATION treats PROBLEM"`
→ `MEDICATION_treats_PROBLEM`, which is what `setRelations()` actually wants) while the reference
passes the spaced string through; the notebook sets all four thresholds while the reference sets two;
the reference uppercases and sorts entity labels while the notebook preserves them. This service
follows the notebook on all three.

## Run it

```bash
export SPARK_NLP_SECRET=$(python3 -c \
  "import json;print(json.load(open('/home/ubuntu/cabir/keys/6.4.1.spark_nlp_for_healthcare.json'))['SECRET'])")

DOCKER_BUILDKIT=1 docker compose -f dc.nlp-services-standalone.yaml up --build -d
docker compose -f dc.nlp-services-standalone.yaml logs -f
```

Serves on **:5002** (5001 belongs to `custom_nlp_service`). The model loads once at startup —
`/health` reports `"status": "warming"` until it is ready, which is healthy.

## API

| Method | Path | |
|---|---|---|
| GET | `/health` | status, `model_loaded`, queue depth, ES reachability |
| GET | `/api/v1/model` | the pinned model and its defaults |
| POST | `/api/v1/runs` | → `202` + `session_id` |
| GET | `/api/v1/status/{session_id}` | poll until `completed` |
| GET | `/api/v1/results/{session_id}` | `?limit=500&offset=0` |
| GET | `/api/v1/logs/{session_id}` | `?tail=200&full=false` |

### Payload

`documents` (inline text) and `document_ids` (fetched from Elasticsearch) may both appear; at
least one is required.

```jsonc
{
  "documents": [
    {"document_id": "doc-0", "text": "Patient started on metformin 500mg oral twice daily..."}
  ],
  "document_ids": ["10082662-RR-37"],          // optional; needs ES_ENABLED=true
  "zero_shot": {
    "model_id": "zeroshot_multitask_base",     // optional; must match the pinned model
    "entities": [
      {"label": "PROBLEM", "dtype": "str", "description": "A medical condition, symptom, or diagnosis"},
      "MEDICATION"                              // bare labels work too
    ],
    "structures": [
      {"name": "medication_item", "fields": [
        {"name": "drug_name", "dtype": "str", "description": "Name of the drug"},
        {"name": "route", "choices": ["oral", "IV", "topical"]}   // enum field
      ]}
    ],
    "classifications": [
      {"task": "document_type", "labels": ["Radiology Report", "Discharge Summary", "Progress Note"]}
    ],
    "relations": ["MEDICATION treats PROBLEM"],
    "entity_threshold": 0.6,
    "structure_threshold": 0.6,
    "classification_threshold": 0.6,
    "relation_threshold": 0.6
  },
  "job_details": "{\"source\": \"jupyter\", \"mode\": \"zero_shot_multitask\"}"
}
```

At least one of `entities` / `structures` / `classifications` / `relations` is required. A malformed
config is rejected at POST with 422, not minutes later inside the worker.

### Results

One row per annotation. `label` means something different per task, so read
`raw_metadata.task_type` to interpret it:

| task_type | rows | `label` | `span_text` |
|---|---|---|---|
| `ner` | 1 | entity type | the chunk |
| `classification` | 1 | task name | predicted label |
| `relation` | **2** (head + tail) | relation name | `chunk1` / `chunk2` |
| `structure` | **1 per field** | field name | field text |

Relation rows share a `relation_id` and carry `raw_metadata.role` (`head`/`tail`); structure rows
share a `structure_id` + `instance_idx`. Either can be regrouped into whole instances.

`sentence` is the **sentence's text**; its index is `raw_metadata.sentence_number`:

```jsonc
{
  "label": "PROCEDURE",
  "span_text": "CT abdomen",
  "start": 14, "end": 23,
  "sentence": "EXAMINATION:  CT abdomen and pelvis.",
  "raw_metadata": {"task_type": "ner", "entity": "PROCEDURE", "sentence_number": 0, ...}
}
```

The annotator only emits the index (`metadata["sentence"]`); the service resolves it against the
pipeline's sentence column. This matches `custom_nlp_service` as of commit `5157117`
("fixed sentence_text instead of sentence_id").

`sentence` is the **full text** of the sentence the annotation came from; its integer index is
kept as `raw_metadata.sentence_number`. (The annotator's metadata only carries the index; the
service resolves it against the pipeline's sentence column.)

## One model per container

`PretrainedZeroShotMultiTask` has a JVM-level defect: a **second** instance in the same Spark
session corrupts it — `constructSchema` then reads a `java.util.ArrayList` where it expects
`Tuple2[]`, and *every* later `.transform()` throws `ClassCastException` until the container
restarts. `.copy()` returns uninitialized weights, so it is not a way out.

So the annotator is loaded once per JVM and reconfigured per request, execution is serialized
(`--workers 1`, one worker thread, and `multitask_model.configured()` — a context manager that
holds the exec lock across reconfigure→transform→materialize, so callers can't get it wrong), and
each container serves exactly one model. `MULTITASK_MODEL_ID` picks it; any other `model_id` in a
request is rejected with 422. To serve `zeroshot_multitask_generic`, run a second container — see
the commented-out service in the compose file.

**Invariant to keep** — only `multitask_model.py` may import or instantiate the annotator, so this
must print nothing but `app/multitask_model.py`:

```bash
grep -rlE 'import .*PretrainedZeroShotMultiTask|PretrainedZeroShotMultiTask\.pretrained' app/
```

## Configuration

| Env | Default | |
|---|---|---|
| `MULTITASK_MODEL_ID` | `zeroshot_multitask_base` | the one model this container serves |
| `PRETRAINED_CACHE_FOLDER` | `/root/cache_pretrained` | where spark-nlp caches models |
| `LICENSE_FILE_PATH` | `/tmp/license/license.json` | JSL license JSON |
| `SESSION_HISTORY_LIMIT` | `50` | sessions kept in memory before the oldest is dropped |
| `ES_ENABLED` | `false` | needed only for `document_ids` |
| `ES_URL` / `ES_USER` / `ES_PASSWORD` | `http://pj-nosql:9200` | |
| `INPUT_INDEX` | `raw_extractions` | |
| `RESULTS_ES_WRITE` | `false` | mirror results into ES as well |

**The model cache mount matters.** Under JDK 8 the JVM resolves `user.home=/root` even though the
image sets `HOME=/app`, so spark-nlp caches into `/root/cache_pretrained`. `custom_nlp_service`
mounts an empty `/tmp/cache_pretrained` there and re-downloads ~800MB on every restart; this compose
mounts the host's real cache instead, so a warm start takes ~30s.

Note that spark-nlp prints `<model> download started ... Download done!` **even on a cache hit**, so
the logs cannot tell you whether it actually transferred anything. To confirm the mount is working,
check that nothing new was written:

```bash
find /home/ubuntu/cache_pretrained -maxdepth 1 -newermt "-10 minutes"   # empty = served from cache
```

## Verify

```bash
python3 tools/parity_check.py --dsl-only                       # config normalization, no Spark
python3 tools/parity_check.py --run                            # POST the notebook's 3 samples
python3 tools/parity_check.py --run --expect /tmp/notebook_rows.json   # diff against the notebook
```

`--run` asserts all four task types come back. `notebook_rows.json` is written by section 6 of
`build_multitask_pipeline.ipynb` (to `/tmp/notebook_rows.json`) once you re-run it — it uses
`document_id=f"doc-{i}"`, and giving the service the same ids makes even the `relation_id` /
`structure_id` hashes comparable. The diff drops volatile fields (`created_at`, `session_id`,
`result_id`) and sorts both sides, since Spark partition order is not guaranteed.

When writing a test payload, note that relations are extracted **within a sentence**: a config
asking for `MEDICATION treats PROBLEM` yields nothing if the drug and the condition sit in
different sentences. That looks like a broken service but is not one.

## Storage: memory only

Sessions, result rows, and logs live in RAM and are never written to disk — there is no data
volume. Consequences worth knowing:

- **A restart drops everything.** Polling a session id from before a restart returns 404, not a
  status. Fetch your results before restarting the container.
- **Only the last `SESSION_HISTORY_LIMIT` (50) sessions are readable.** Older ones are evicted
  oldest-first, taking their rows and logs with them.
- **Memory scales with retained sessions × their rows.** Small runs are nothing (the notebook's 3
  samples are ~82 rows); a 1000-document session is tens of MB, so lower the limit if you run
  those. Per-session logs are capped at 2000 lines.
- **For durable results**, set `RESULTS_ES_WRITE=true` and read them from the
  `custom_nlp_results` index.

## Layout

```
app/
  zeroshot_dsl.py     ★ config -> annotator DSL         (shared with the notebook; stdlib only)
  rows.py             ★ extractions -> flat rows        (shared with the notebook; stdlib only)
  multitask_model.py    the one annotator; configured() lends it out under the exec lock
  runner.py             load -> transform -> flatten -> persist
  queue_manager.py      single worker thread + preload
  store.py              in-memory sessions: status + rows + logs, evicted as one unit
  documents.py          load() -> docs from inline text and/or ES, one Spark schema
  es_client.py          optional ES (fetch_source_documents / mirror_results)
  runtime.py            license + Spark session
  models.py, main.py    API
```
