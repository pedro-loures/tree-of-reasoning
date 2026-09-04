"""FastAPI server for the interactive tree dashboard."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.dashboard.payload import merge_runs
from src.dashboard.persist import (
    default_save_dir,
    list_saved_runs,
    load_run_file,
    save_run,
)
from src.dashboard.progress import JobManager
from src.dashboard.rescore import rescore_run
from src.dashboard.service import DashboardConfig, DashboardService

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
TEMPLATE_DIR = WORKSPACE_ROOT / "tree-testing" / "templates"
TEMPLATE_PATH = TEMPLATE_DIR / "interactive_dashboard.html"
D3_PATH = WORKSPACE_ROOT / "tree-testing" / "output" / "tau001_viz" / "d3.min.js"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model_id: str
    tau: float = Field(default=0.01, gt=0.0, le=1.0)
    expected_answers: str | None = None
    answer_mode: str = Field(default="or")


class LoadSavedRequest(BaseModel):
    filename: str | None = None
    tree_key: str | None = None


class RescoreRequest(BaseModel):
    expected_answers: str | None = None
    answer_mode: str = Field(default="or")


class DashboardState:
    def __init__(self) -> None:
        self._runs: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_run(self, run: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._runs.insert(0, run)
            return merge_runs(self._runs)

    def get_run(self, tree_key: str) -> dict[str, Any] | None:
        with self._lock:
            for run in self._runs:
                if run["tree_key"] == tree_key:
                    return run
            return None

    def import_run(self, run: dict[str, Any], *, replace: bool = False) -> dict[str, Any]:
        with self._lock:
            tree_key = run["tree_key"]
            for index, existing in enumerate(self._runs):
                if existing["tree_key"] == tree_key:
                    if replace:
                        self._runs[index] = run
                    return merge_runs(self._runs)
            self._runs.insert(0, run)
            return merge_runs(self._runs)

    def update_run(self, tree_key: str, run: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            for index, existing in enumerate(self._runs):
                if existing["tree_key"] == tree_key:
                    self._runs[index] = run
                    return merge_runs(self._runs)
            raise KeyError(tree_key)

    def remove_run(self, tree_key: str) -> dict[str, Any]:
        with self._lock:
            self._runs = [run for run in self._runs if run["tree_key"] != tree_key]
            return merge_runs(self._runs)

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self._runs = []
            return merge_runs(self._runs)

    def payload(self) -> dict[str, Any]:
        with self._lock:
            return merge_runs(self._runs)

    def all_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._runs)


def create_app(
    config_path: Path | None = None,
    template_path: Path | None = None,
    save_dir: Path | None = None,
    *,
    enable_tqdm: bool = True,
) -> FastAPI:
    config_path = config_path or (REPO_ROOT / "configs" / "experiment.yaml")
    template_path = template_path or TEMPLATE_PATH
    sessions_dir = save_dir or default_save_dir()

    dashboard_config = DashboardConfig(
        config_path=config_path,
        repo_root=REPO_ROOT,
    )
    service = DashboardService(dashboard_config)
    state = DashboardState()
    jobs = JobManager()
    gpu_lock = threading.Lock()

    app = FastAPI(title="Interactive Tree Dashboard", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        if not template_path.exists():
            raise HTTPException(status_code=500, detail=f"Template not found: {template_path}")
        return FileResponse(template_path)

    @app.get("/api/models")
    def list_models() -> dict[str, Any]:
        return {"models": service.list_models()}

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return state.payload()

    @app.get("/api/progress")
    def get_progress() -> dict[str, Any]:
        return jobs.progress_dict()

    @app.post("/api/generate")
    def generate_tree(request: GenerateRequest) -> dict[str, Any]:
        if jobs.active_job() is not None and jobs.active_job().is_running():
            raise HTTPException(status_code=409, detail="A tree generation is already in progress")

        def run_job(progress) -> dict[str, Any]:
            with gpu_lock:
                return service.generate_tree(
                    prompt=request.prompt,
                    model_id=request.model_id,
                    tau=request.tau,
                    expected_answers=request.expected_answers,
                    answer_mode=request.answer_mode,
                    progress=progress,
                )

        def finalize(job) -> None:
            if job.thread is not None:
                job.thread.join()
            snapshot = job.progress.snapshot()
            if snapshot.status == "completed" and job.result is not None:
                payload = state.add_run(job.result)
                jobs.attach_payload(payload)

        try:
            job = jobs.start(run_job, enable_tqdm=enable_tqdm)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        threading.Thread(target=finalize, args=(job,), daemon=True).start()
        return {"job_id": job.job_id, "status": "started"}

    @app.delete("/api/runs/{tree_key}")
    def delete_run(tree_key: str) -> dict[str, Any]:
        return state.remove_run(tree_key)

    @app.post("/api/runs/{tree_key}/rescore")
    def rescore_tree_run(tree_key: str, request: RescoreRequest) -> dict[str, Any]:
        run = state.get_run(tree_key)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Tree not found: {tree_key}")
        try:
            updated = rescore_run(
                run,
                expected_answers=request.expected_answers,
                answer_mode=request.answer_mode,
            )
            payload = state.update_run(tree_key, updated)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Tree not found: {tree_key}") from None
        return payload

    @app.post("/api/runs/{tree_key}/save")
    def save_tree_run(tree_key: str) -> dict[str, Any]:
        run = state.get_run(tree_key)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Tree not found: {tree_key}")
        path = save_run(run, sessions_dir)
        return {
            "tree_key": tree_key,
            "saved_path": str(path),
            "filename": path.name,
            "message": "Tree saved to disk.",
        }

    @app.post("/api/save-all")
    def save_all_runs() -> dict[str, Any]:
        runs = state.all_runs()
        if not runs:
            raise HTTPException(status_code=404, detail="No trees in session to save")
        saved = []
        for run in runs:
            path = save_run(run, sessions_dir)
            saved.append({"tree_key": run["tree_key"], "filename": path.name, "saved_path": str(path)})
        return {"saved": saved, "count": len(saved)}

    @app.get("/api/saved")
    def list_saved_trees() -> dict[str, Any]:
        return {"saved": list_saved_runs(sessions_dir), "save_dir": str(sessions_dir)}

    @app.post("/api/saved/load")
    def load_saved_tree(request: LoadSavedRequest) -> dict[str, Any]:
        if not request.filename and not request.tree_key:
            raise HTTPException(status_code=400, detail="Provide filename or tree_key")

        entries = list_saved_runs(sessions_dir)
        match = None
        if request.filename:
            match = next((entry for entry in entries if entry["filename"] == request.filename), None)
        elif request.tree_key:
            match = next((entry for entry in entries if entry["tree_key"] == request.tree_key), None)
        if match is None:
            raise HTTPException(status_code=404, detail="Saved tree not found")

        run = load_run_file(Path(match["path"]))
        payload = state.import_run(run)
        return {
            "tree_key": run["tree_key"],
            "filename": match["filename"],
            "payload": payload,
            "message": "Saved tree loaded into session.",
        }

    @app.post("/api/clear")
    def clear_runs() -> dict[str, Any]:
        return state.clear()

    if D3_PATH.exists():
        app.mount("/static", StaticFiles(directory=str(D3_PATH.parent)), name="static")

    app.mount("/assets", StaticFiles(directory=str(TEMPLATE_DIR)), name="assets")

    return app
