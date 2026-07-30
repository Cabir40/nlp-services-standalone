"""Stdlib logger. Replaces pj_lib.logger, which this service deliberately does not depend on."""

from __future__ import annotations

import logging
import os
import sys

_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("nlp_multitask")
logger.setLevel(_LEVEL)
