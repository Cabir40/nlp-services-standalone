"""Container healthcheck. Stdlib only -- this runs before/independent of the app's deps."""

import json
import sys
import urllib.request

URL = "http://localhost:8510/health"

try:
    with urllib.request.urlopen(URL, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception as exc:  # noqa: BLE001 - any failure is an unhealthy container
    print(f"health check failed: {exc}")
    sys.exit(1)

status = payload.get("status")
print(f"status={status} model_loaded={payload.get('model_loaded')}")

# "warming" is healthy: the model load takes minutes on a cold cache.
sys.exit(0 if status in ("ok", "warming") else 1)
