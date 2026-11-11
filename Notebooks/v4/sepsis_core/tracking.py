"""Fail-safe MLflow logging. Logging can never stop a run, and the v3 "Run ... is already active" bug is fixed:
any run left open (by a crash, an interrupted cell or a timeout inside start_run) is ended before a new top-level run starts
and again after any logging failure."""
import os, contextlib
import numpy as np

try:
    import mlflow
except Exception:                      # tracking is optional everywhere
    mlflow = None

TRACKING = {"enabled": False, "remote": False, "uri": None, "fails": 0}


def _secret(name):
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v: return v
    except Exception:
        pass
    return os.environ.get(name)


def end_active_runs():
    """End every run mlflow still considers active. Safe to call at any time."""
    if mlflow is None: return 0
    n = 0
    try:
        while mlflow.active_run() is not None and n < 20:
            mlflow.end_run(); n += 1
    except Exception as e:
        print(f"[mlflow] could not end an active run ({type(e).__name__}: {str(e)[:100]})")
    return n


def setup_tracking(experiment, use_dagshub=True, local_dir="."):
    """DagsHub if the Colab secrets DAGSHUB_USER / DAGSHUB_REPO / DAGSHUB_TOKEN exist, otherwise a local ./mlruns store. Never raises."""
    if mlflow is None:
        print("[mlflow] not installed -> tracking off (checkpoints and the manifest are still written)"); return TRACKING
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30"); os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
    end_active_runs()
    if use_dagshub:
        try:
            user, repo, token = _secret("DAGSHUB_USER"), _secret("DAGSHUB_REPO"), _secret("DAGSHUB_TOKEN")
            owner = _secret("DAGSHUB_REPO_OWNER") or user
            if not (owner and repo and token): raise RuntimeError("DagsHub secrets not set")
            os.environ["MLFLOW_TRACKING_USERNAME"] = user; os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            mlflow.set_tracking_uri(f"https://dagshub.com/{owner}/{repo}.mlflow"); mlflow.set_experiment(experiment)
            TRACKING.update(enabled=True, remote=True, uri=mlflow.get_tracking_uri(), fails=0)
            print("[mlflow] connected to DagsHub:", TRACKING["uri"]); return TRACKING
        except Exception as e:
            print(f"[mlflow] DagsHub unavailable ({type(e).__name__}: {str(e)[:160]}) -> LOCAL store")
    try:
        uri = "file://" + os.path.join(os.path.abspath(local_dir), "mlruns")
        mlflow.set_tracking_uri(uri); mlflow.set_experiment(experiment)
        TRACKING.update(enabled=True, remote=False, uri=uri, fails=0); print("[mlflow] LOCAL tracking at", uri)
    except Exception as e:
        TRACKING.update(enabled=False); print(f"[mlflow] tracking off ({type(e).__name__}: {str(e)[:160]})")
    return TRACKING


def _guard(fn, *a, **k):
    if not TRACKING["enabled"] or mlflow is None: return None
    try:
        r = fn(*a, **k); TRACKING["fails"] = 0; return r
    except Exception as e:
        TRACKING["fails"] += 1
        print(f"[mlflow] {getattr(fn, '__name__', 'call')} failed ({type(e).__name__}: {str(e)[:120]}) [{TRACKING['fails']}/3]")
        if getattr(fn, "__name__", "") == "start_run": end_active_runs()         # a half-created run must not block the next one
        if TRACKING["fails"] >= 3:
            TRACKING["enabled"] = False; print("[mlflow] tracking DISABLED for the rest of the run (checkpoints still saved).")
        return None


def m_params(d):
    items = [(str(k), str(v)[:500]) for k, v in d.items()]
    for i in range(0, len(items), 90): _guard(mlflow.log_params, dict(items[i:i + 90])) if mlflow else None
def m_metrics(d, step=None):
    clean = {}
    for k, v in d.items():
        try:
            if v is not None and np.isfinite(float(v)): clean[k] = float(v)
        except (TypeError, ValueError): pass
    if clean and mlflow: _guard(mlflow.log_metrics, clean, step=step)
def m_tags(d):
    if mlflow: _guard(mlflow.set_tags, {k: str(v) for k, v in d.items()})
def m_artifact(p, artifact_path=None):
    if mlflow and os.path.exists(p): _guard(mlflow.log_artifact, p, artifact_path)
def m_dict(d, name):
    if mlflow: _guard(mlflow.log_dict, d, name)


@contextlib.contextmanager
def safe_run(run_name=None, nested=False, tags=None, run_id=None):
    """`with safe_run("name"):` -> a run that always closes. A top-level run first ends whatever was left active."""
    run = None
    if TRACKING["enabled"] and mlflow is not None:
        if not nested: end_active_runs()
        run = _guard(mlflow.start_run, run_id=run_id, run_name=(None if run_id else run_name), nested=nested,
                     tags={k: str(v) for k, v in (tags or {}).items()})
    try:
        yield run
    except BaseException:
        if run is not None: _guard(mlflow.end_run, "FAILED")
        raise
    else:
        if run is not None: _guard(mlflow.end_run, "FINISHED")
