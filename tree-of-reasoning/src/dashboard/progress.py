"""Progress tracking for dashboard tree generation (tqdm + API polling)."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from tqdm import tqdm


@dataclass
class ProgressSnapshot:
    job_id: str
    status: str
    stage: str
    message: str
    current: int
    total: int | None
    percent: float | None
    nodes: int | None = None
    leaves: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "current": self.current,
            "total": self.total,
            "percent": self.percent,
            "nodes": self.nodes,
            "leaves": self.leaves,
            "error": self.error,
            "result": self.result,
        }


class ProgressTracker:
    """Thread-safe progress state with an optional terminal tqdm bar."""

    def __init__(self, job_id: str, *, enable_tqdm: bool = True) -> None:
        self.job_id = job_id
        self._lock = threading.Lock()
        self._enable_tqdm = enable_tqdm
        self._status = "running"
        self._stage = "starting"
        self._message = "Starting…"
        self._current = 0
        self._total: int | None = None
        self._nodes: int | None = None
        self._leaves: int | None = None
        self._error: str | None = None
        self._bar: tqdm | None = None

    def snapshot(self) -> ProgressSnapshot:
        with self._lock:
            percent = None
            if self._total and self._total > 0:
                percent = round(100.0 * min(self._current, self._total) / self._total, 1)
            return ProgressSnapshot(
                job_id=self.job_id,
                status=self._status,
                stage=self._stage,
                message=self._message,
                current=self._current,
                total=self._total,
                percent=percent,
                nodes=self._nodes,
                leaves=self._leaves,
                error=self._error,
            )

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def start_stage(
        self,
        stage: str,
        message: str,
        *,
        total: int | None = None,
        current: int = 0,
    ) -> None:
        with self._lock:
            self._stage = stage
            self._message = message
            self._total = total
            self._current = current
            self._close_bar()
            if self._enable_tqdm:
                self._bar = tqdm(
                    total=total,
                    initial=current,
                    desc=stage,
                    unit="step",
                    dynamic_ncols=True,
                    leave=True,
                )
                self._bar.set_postfix_str(message, refresh=False)

    def update(
        self,
        *,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        nodes: int | None = None,
        leaves: int | None = None,
        advance: int = 0,
    ) -> None:
        with self._lock:
            if message is not None:
                self._message = message
            if total is not None:
                self._total = total
            if nodes is not None:
                self._nodes = nodes
            if leaves is not None:
                self._leaves = leaves
            if current is not None:
                delta = current - self._current
                self._current = current
            else:
                delta = advance
                self._current += advance

            if self._bar is not None:
                if delta > 0:
                    self._bar.update(delta)
                if self._total is not None and self._bar.total != self._total:
                    self._bar.total = self._total
                    self._bar.refresh()
                postfix_parts = [self._message]
                if self._nodes is not None:
                    postfix_parts.append(f"nodes={self._nodes}")
                if self._leaves is not None:
                    postfix_parts.append(f"leaves={self._leaves}")
                self._bar.set_postfix_str(" · ".join(postfix_parts), refresh=False)

    def finish(self, message: str = "Done") -> None:
        with self._lock:
            self._status = "completed"
            self._message = message
            if self._total is not None:
                self._current = self._total
            self._close_bar()

    def fail(self, error: str) -> None:
        with self._lock:
            self._status = "failed"
            self._error = error
            self._message = error
            self._close_bar()


@dataclass
class GenerationJob:
    job_id: str
    progress: ProgressTracker
    thread: threading.Thread | None = None
    result: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None

    def is_running(self) -> bool:
        return self.progress.snapshot().status == "running"


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: GenerationJob | None = None

    def active_job(self) -> GenerationJob | None:
        with self._lock:
            return self._active

    def start(
        self,
        target: Callable[[ProgressTracker], dict[str, Any]],
        *,
        enable_tqdm: bool = True,
    ) -> GenerationJob:
        with self._lock:
            if self._active is not None and self._active.is_running():
                raise RuntimeError("A tree generation is already in progress")

            job_id = uuid.uuid4().hex[:12]
            progress = ProgressTracker(job_id, enable_tqdm=enable_tqdm)
            job = GenerationJob(job_id=job_id, progress=progress)
            self._active = job

        def runner() -> None:
            try:
                job.result = target(progress)
                progress.finish("Tree ready")
            except Exception as exc:
                progress.fail(str(exc))

        job.thread = threading.Thread(target=runner, daemon=True)
        job.thread.start()
        return job

    def progress_dict(self) -> dict[str, Any]:
        with self._lock:
            job = self._active
            if job is None:
                return {
                    "active": False,
                    "status": "idle",
                    "stage": "",
                    "message": "",
                    "current": 0,
                    "total": None,
                    "percent": None,
                }
            snapshot = job.progress.snapshot()
            data = snapshot.to_dict()
            data["active"] = snapshot.status == "running"
            if snapshot.status == "completed" and job.payload is not None:
                data["payload"] = job.payload
            return data

    def attach_payload(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._active is not None:
                self._active.payload = payload

    def clear_if_done(self) -> None:
        with self._lock:
            if self._active is not None and not self._active.is_running():
                if self._active.progress.snapshot().status in {"completed", "failed"}:
                    pass
