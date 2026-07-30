"""Environment-driven config.

custom_nlp_service resolves every value env -> service_config.json -> TPJ config DB -> default.
This service is standalone, so there is one source: the environment. Properties read os.environ
on each access rather than snapshotting it, because runtime.ensure_license() exports the license
keys into the environment after this module is first imported.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_MODEL_ID = "zeroshot_multitask_base"
SENTENCE_DETECTOR_MODEL = "sentence_detector_dl_healthcare_v2_wip"


def _env(name: str, default: Any = None) -> Any:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    # --- model -------------------------------------------------------------------------------
    @property
    def model_id(self) -> str:
        """The single multitask model this container serves. See multitask_model.get_annotator."""
        return str(_env("MULTITASK_MODEL_ID", DEFAULT_MODEL_ID))

    @property
    def sentence_detector_model(self) -> str:
        return str(_env("SENTENCE_DETECTOR_MODEL", SENTENCE_DETECTOR_MODEL))

    @property
    def pretrained_cache_folder(self) -> str:
        """Where spark-nlp reads/writes pretrained models.

        Set explicitly because the default resolves from the JVM's user.home, which is /root
        under JDK 8 even though the image sets HOME=/app -- a mismatch that silently
        re-downloads ~800MB per restart if the volume is mounted at the wrong path.
        """
        return str(_env("PRETRAINED_CACHE_FOLDER", "/root/cache_pretrained"))

    # --- runtime -----------------------------------------------------------------------------
    @property
    def license_file_path(self) -> Path:
        return Path(str(_env("LICENSE_FILE_PATH", "/tmp/license/license.json")))

    @property
    def public_url(self) -> Optional[str]:
        return _env("PUBLIC_URL")

    @property
    def queue_poll_seconds(self) -> int:
        return _env_int("QUEUE_POLL_SECONDS", 1)

    @property
    def session_history_limit(self) -> int:
        """How many finished sessions stay readable before the oldest is dropped.

        Sessions live in memory only, so this bounds RAM: roughly this many sessions' worth of
        result rows. Raise it for more history, lower it if runs are large.
        """
        return _env_int("SESSION_HISTORY_LIMIT", 50)

    def route_url(self, path: str) -> str:
        base = self.public_url
        return f"{base.rstrip('/')}{path}" if base else path

    # --- elasticsearch (optional: only for document_ids input / result mirroring) -------------
    @property
    def es_enabled(self) -> bool:
        return _env_bool("ES_ENABLED", False)

    @property
    def es_url(self) -> str:
        return str(_env("ES_URL", "http://pj-nosql:9200"))

    @property
    def es_basic_auth(self) -> Optional[tuple]:
        user, password = _env("ES_USER"), _env("ES_PASSWORD")
        return (str(user), str(password)) if user and password else None

    @property
    def input_index(self) -> str:
        return str(_env("INPUT_INDEX", "raw_extractions"))

    @property
    def results_index(self) -> str:
        return str(_env("RESULTS_INDEX", "custom_nlp_results"))

    @property
    def results_es_write(self) -> bool:
        return _env_bool("RESULTS_ES_WRITE", False)

    # --- spark -------------------------------------------------------------------------------
    @property
    def spark_hardware(self) -> str:
        return str(_env("SPARK_HARDWARE", "cpu")).lower()

    def spark_params(self) -> Dict[str, str]:
        """Adapted from custom_nlp_service's settings.spark_params().

        Dropped vs the reference: spark.jars.packages (the elasticsearch-spark connector, which
        forces a Maven resolve on every JVM start -- ES I/O here goes through the Python client)
        and all spark.es.* keys that configured it.
        """
        return {
            "spark.app.name": str(_env("SPARK_APP_NAME", "multitask_zeroshot_standalone")),
            "spark.master": str(_env("SPARK_MASTER", "local[*]")),
            "spark.driver.memory": str(_env("SPARK_DRIVER_MEMORY", "8g")),
            "spark.driver.cores": str(_env("SPARK_DRIVER_CORES", "2")),
            "spark.driver.maxResultSize": str(_env("SPARK_DRIVER_MAX_RESULT_SIZE", "4096m")),
            "spark.executor.memory": str(_env("SPARK_EXECUTOR_MEMORY", "8g")),
            "spark.executor.cores": str(_env("SPARK_EXECUTOR_CORES", "2")),
            "spark.executor.memoryOverhead": str(_env("SPARK_EXECUTOR_MEMORY_OVERHEAD", "3g")),
            "spark.task.cpus": str(_env("SPARK_TASK_CPUS", "2")),
            "spark.default.parallelism": str(_env("SPARK_DEFAULT_PARALLELISM", "8")),
            "spark.sql.shuffle.partitions": str(_env("SPARK_SQL_SHUFFLE_PARTITIONS", "8")),
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.kryoserializer.buffer": str(_env("SPARK_KRYO_BUFFER", "256k")),
            "spark.kryoserializer.buffer.max": str(_env("SPARK_KRYO_BUFFER_MAX", "2000m")),
            "spark.executor.heartbeatInterval": "60s",
            "spark.speculation": "false",
            "spark.log.level": "ERROR",
            "spark.extraListeners": "com.johnsnowlabs.license.LicenseLifeCycleManager",
            "spark.jsl.settings.pretrained.cache_folder": self.pretrained_cache_folder,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
