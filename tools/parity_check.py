#!/usr/bin/env python3
"""Prove the service returns the same rows as build_multitask_pipeline.ipynb.

    python3 tools/parity_check.py --dsl-only              # config normalization, no Spark
    python3 tools/parity_check.py --run                   # POST the notebook's samples, print rows
    python3 tools/parity_check.py --run --expect notebook_rows.json   # ... and diff against them

Since both sides now import zeroshot_dsl + rows, parity is mostly structural; what --run tests
is that the Spark/model layer agrees.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from zeroshot_dsl import normalize_config, validate_annotator_config  # noqa: E402

BASE_URL = "http://localhost:5002"

# build_multitask_pipeline.ipynb, "## 1. Config".
NOTEBOOK_CONFIG = {
    "entities": [
        {"label": "PROBLEM", "dtype": "str", "description": "A medical condition, symptom, or diagnosis"},
        {"label": "MEDICATION", "dtype": "str", "description": "Drug or pharmaceutical treatment"},
        {"label": "PROCEDURE", "dtype": "str", "description": "Medical or surgical procedure"},
        {"label": "TEST", "dtype": "str", "description": "Diagnostic test or lab result"},
    ],
    "structures": [
        {
            "name": "medication_item",
            "fields": [
                {"name": "drug_name", "dtype": "str", "description": "Name of the drug"},
                {"name": "dosage", "dtype": "str", "description": "Dose amount and unit"},
                {"name": "frequency", "dtype": "str", "description": "How often taken"},
                {"name": "route", "choices": ["oral", "IV", "topical", "subcutaneous"]},
            ],
        }
    ],
    "classifications": [
        {"task": "document_type", "labels": ["Radiology Report", "Discharge Summary", "Progress Note"]}
    ],
    "relations": ["MEDICATION treats PROBLEM", "TEST diagnoses PROBLEM"],
    "entity_threshold": 0.6,
    "structure_threshold": 0.6,
    "classification_threshold": 0.6,
    "relation_threshold": 0.6,
}

# build_multitask_pipeline.ipynb, "## 4. Run on data".
SAMPLE_TEXTS = [
    "The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby-girl also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.",
    "Progress Note: Jennifer Smith is a 58-year-old woman with type 2 diabetes mellitus and hypertension. She was started on metformin 500mg oral twice daily to control her blood sugar. An HbA1c test was ordered to assess glycemic control. Discharge Summary: The patient underwent an appendectomy for acute appendicitis. Postoperatively he received ibuprofen 400mg tablet orally every 6 hours for pain. A complete blood count was performed to rule out infection.",
    "He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety. Repleted with 20 meq kcl po, 30 mmol K-phos iv and 2 gms mag so4 iv. Size: Prostate gland measures 10x1.1x4.9 cm (LS x AP x TS). Estimated volume is 51.9 ml  and is mildly enlarged in size. Normal delineation pattern of the prostate gland is preserved.",
]

# Differ per run or per environment; not part of what parity means here.
VOLATILE_FIELDS = ("created_at", "session_id", "result_id", "job_details", "source_index")


def _post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())


def check_dsl() -> int:
    config = validate_annotator_config(normalize_config(NOTEBOOK_CONFIG))
    for key in ("entities", "structures", "classifications", "relations"):
        print(f"  {key:16} {config[key]}")

    expected = {
        "entities": [
            "PROBLEM::str::A medical condition, symptom, or diagnosis",
            "MEDICATION::str::Drug or pharmaceutical treatment",
            "PROCEDURE::str::Medical or surgical procedure",
            "TEST::str::Diagnostic test or lab result",
        ],
        "structures": [
            (
                "medication_item",
                [
                    "drug_name::str::Name of the drug",
                    "dosage::str::Dose amount and unit",
                    "frequency::str::How often taken",
                    "route::[oral|IV|topical|subcutaneous]",
                ],
            )
        ],
        "classifications": [
            ("document_type", ["Radiology Report", "Discharge Summary", "Progress Note"])
        ],
        # The space form is what custom_nlp_service sends today; setRelations needs single tokens.
        "relations": ["MEDICATION_treats_PROBLEM", "TEST_diagnoses_PROBLEM"],
    }
    for key, want in expected.items():
        if config[key] != want:
            print(f"\n[FAIL] {key}\n  expected {want}\n  got      {config[key]}")
            return 1

    print("\n[PASS] normalize_config() matches the notebook's annotator syntax")
    return 0


def _sort_key(row: dict) -> tuple:
    meta = row.get("raw_metadata") or {}
    return (
        str(row.get("document_id")),
        str(meta.get("task_type")),
        str(row.get("label")),
        row.get("start") if row.get("start") is not None else -1,
        row.get("end") if row.get("end") is not None else -1,
        str(row.get("span_text")),
        str(meta.get("role") or meta.get("field") or ""),
    )


def _canonical(row: dict) -> dict:
    out = {k: v for k, v in row.items() if k not in VOLATILE_FIELDS}
    meta = {k: v for k, v in (out.get("raw_metadata") or {}).items() if k != "uuid"}
    out["raw_metadata"] = meta
    if isinstance(out.get("confidence"), float):
        out["confidence"] = round(out["confidence"], 6)
    return out


def run(expect_path: str | None, dump_path: str | None) -> int:
    payload = {
        "documents": [
            # doc-{i} rather than null: rows.relation_id/structure_id hash document_id, so
            # matching ids make even those hashes comparable against the notebook.
            {"document_id": f"doc-{i}", "text": text}
            for i, text in enumerate(SAMPLE_TEXTS)
        ],
        # No model_id: the service uses whichever model it is pinned to.
        "zero_shot": dict(NOTEBOOK_CONFIG),
        "job_details": json.dumps({"source": "parity", "mode": "zero_shot_multitask"}),
    }

    session = _post(f"{BASE_URL}/api/v1/runs", payload)
    session_id = session["session_id"]
    print(f"session_id: {session_id}")

    deadline = time.time() + 900
    while time.time() < deadline:
        status = _get(f"{BASE_URL}/api/v1/status/{session_id}")
        print(
            f"  status={status['status']} progress={status['progress']}% "
            f"stage={status.get('current_stage')}"
        )
        if status["status"] in ("completed", "completed_with_errors", "failed"):
            break
        time.sleep(3)
    else:
        print("[FAIL] timed out waiting for the session")
        return 1

    if status["status"] == "failed":
        print(f"[FAIL] session failed: {status.get('message')} errors={status.get('errors')}")
        return 1

    body = _get(f"{BASE_URL}/api/v1/results/{session_id}?limit=5000&offset=0")
    rows = body["results"]
    counts = Counter((r.get("raw_metadata") or {}).get("task_type") for r in rows)
    print(f"\n{body['total']} rows | by task_type: {dict(counts)}")

    if dump_path:
        Path(dump_path).write_text(json.dumps(rows, indent=2))
        print(f"wrote {dump_path}")

    missing_tasks = {"ner", "structure", "classification", "relation"} - set(counts)
    if missing_tasks:
        print(f"[FAIL] no rows for task type(s): {sorted(missing_tasks)}")
        return 1
    print("[PASS] all four task types present")

    if not expect_path:
        return 0

    expected = json.loads(Path(expect_path).read_text())
    got = [_canonical(r) for r in sorted(rows, key=_sort_key)]
    want = [_canonical(r) for r in sorted(expected, key=_sort_key)]

    if got == want:
        print(f"[PASS] {len(got)} rows identical to {expect_path}")
        return 0

    print(f"[FAIL] rows differ: service={len(got)} notebook={len(want)}")
    for index, (a, b) in enumerate(zip(got, want)):
        if a != b:
            print(f"\nfirst difference at index {index}:")
            print(f"  service:  {json.dumps(a, default=str)[:400]}")
            print(f"  notebook: {json.dumps(b, default=str)[:400]}")
            break
    return 1


def main() -> int:
    global BASE_URL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsl-only", action="store_true", help="check config normalization only")
    parser.add_argument("--run", action="store_true", help="POST the notebook's samples")
    parser.add_argument("--expect", help="notebook rows JSON to diff against")
    parser.add_argument("--dump", help="write the service's rows here")
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    BASE_URL = args.base_url

    if args.dsl_only or not args.run:
        code = check_dsl()
        if code or not args.run:
            return code
    return run(args.expect, args.dump)


if __name__ == "__main__":
    sys.exit(main())
