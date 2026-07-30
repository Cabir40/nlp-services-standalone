"""JSL license bootstrap + Spark session singleton.

Trimmed from custom_nlp_service/app/runtime.py. The license and stale-JVM-recovery logic is
carried over as-is (it is hard-won); the TPJ pipeline import bootstrap, the config-DB license
fallback, the ES helpers and the DSL normalizers are all dropped -- this service needs none of
them.
"""

from __future__ import annotations

import json
import os
from threading import Lock
from typing import Dict, Optional

from py4j.protocol import Py4JError, Py4JNetworkError
from pyspark import SparkContext
from pyspark.sql import SparkSession

from logging_setup import logger
from settings import get_settings

try:
    import sparknlp_jsl
except ImportError:  # pragma: no cover - absent during static checks outside the container
    sparknlp_jsl = None


_LICENSE_LOCK = Lock()
_SPARK_LOCK = Lock()

# Version pins in the license file describe the JSL release, not the runtime; exporting them
# would override the versions the image was actually built with.
_LICENSE_IGNORED_KEYS = {"JSL_VERSION", "PUBLIC_VERSION", "OCR_VERSION", "SPARK_OCR_SECRET"}


def _ensure_license_aliases() -> tuple[Optional[str], Optional[str]]:
    secret = (
        os.environ.get("SPARK_NLP_SECRET")
        or os.environ.get("SPARK_NLP_JSL_SECRET")
        or os.environ.get("SECRET")
    )
    if secret:
        os.environ["SPARK_NLP_SECRET"] = secret
        os.environ["SECRET"] = secret

    license_value = os.environ.get("SPARK_NLP_LICENSE") or os.environ.get("JSL_NLP_LICENSE")
    if license_value:
        os.environ["SPARK_NLP_LICENSE"] = license_value
        os.environ["JSL_NLP_LICENSE"] = license_value

    return secret, license_value


def _set_license_env_vars(license_keys: Dict[str, object]) -> None:
    for key, value in license_keys.items():
        normalized = str(key).upper()
        if normalized in _LICENSE_IGNORED_KEYS:
            continue
        os.environ[normalized] = str(value)
    _ensure_license_aliases()


def ensure_license() -> None:
    with _LICENSE_LOCK:
        license_path = get_settings().license_file_path
        if license_path.exists():
            with license_path.open("r", encoding="utf-8") as handle:
                _set_license_env_vars(json.load(handle))
            logger.info("Loaded JSL license from %s", license_path)

        secret, license_value = _ensure_license_aliases()
        if secret and license_value:
            return

        raise RuntimeError(
            f"JSL license could not be initialized (looked in {license_path}). Ensure both a "
            "runtime secret (SPARK_NLP_SECRET or SECRET) and a license token "
            "(SPARK_NLP_LICENSE or JSL_NLP_LICENSE) are available."
        )


def get_spark() -> SparkSession:
    session = _get_active_session()
    if session is not None:
        return session

    with _SPARK_LOCK:
        session = _get_active_session()
        if session is not None:
            return session

        ensure_license()
        settings = get_settings()
        hardware = settings.spark_hardware
        secret = os.environ.get("SPARK_NLP_SECRET") or os.environ.get("SECRET")

        logger.info("Starting Spark runtime (hardware=%s)", hardware)
        session = None
        if sparknlp_jsl is not None and secret:
            session = sparknlp_jsl.start(
                secret=secret,
                gpu=(hardware == "gpu"),
                params=settings.spark_params(),
            )

        session = session or SparkSession.getActiveSession()
        if session is None:
            raise RuntimeError("Spark session could not be initialized.")

        session.sparkContext.setLogLevel("ERROR")
        logger.info("Spark runtime ready (applicationId=%s)", session.sparkContext.applicationId)
        return session


def spark_is_initialized() -> bool:
    return _get_active_session() is not None


def _reset_stale_spark_state() -> None:
    """Clear stale pyspark globals after the JVM side disappears."""
    for attr_name in ("_activeSession", "_instantiatedSession"):
        if hasattr(SparkSession, attr_name):
            setattr(SparkSession, attr_name, None)
    if hasattr(SparkContext, "_active_spark_context"):
        SparkContext._active_spark_context = None


def _get_active_session() -> Optional[SparkSession]:
    try:
        candidate = SparkSession.getActiveSession()
    except (AttributeError, ConnectionRefusedError, Py4JError, Py4JNetworkError) as exc:
        logger.warning("Discarding stale Spark session reference after JVM failure: %s", exc)
        _reset_stale_spark_state()
        return None

    if candidate is None:
        candidate = getattr(SparkSession, "_instantiatedSession", None)
    if candidate is None:
        return None

    try:
        candidate.sparkContext.applicationId
        return candidate
    except (AttributeError, ConnectionRefusedError, Py4JError, Py4JNetworkError) as exc:
        logger.warning("Discarding stale Spark session reference after JVM failure: %s", exc)
        _reset_stale_spark_state()
        return None
