# src/observability/mlflow_utils.py
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

def init_mlflow(experiment: str = "market-sentiment-analyzer", tracking_uri: Optional[str] = None) -> None:
    import mlflow  # local import avoids name shadowing issues
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    # Best-effort autolog for LangChain; ignore if not available
    try:
        import mlflow.langchain  # noqa: F401
        mlflow.langchain.autolog()
    except Exception:
        pass

@contextmanager
def span(name: str, params: Optional[Dict[str, Any]] = None):
    import mlflow  # local import
    start = time.time()
    with mlflow.start_run(nested=True, run_name=name):
        if params:
            mlflow.log_params({f"{name}.{k}": v for k, v in params.items()})
        try:
            yield
        finally:
            duration_ms = int((time.time() - start) * 1000)
            mlflow.log_metric(f"{name}.duration_ms", duration_ms)

def log_text(path: str, text: str) -> None:
    import mlflow  # local import
    mlflow.log_text(text, artifact_file=path)

def log_dict(path: str, obj: Dict[str, Any]) -> None:
    import mlflow  # local import
    mlflow.log_dict(obj, artifact_file=path)
