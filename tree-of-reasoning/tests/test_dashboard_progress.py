"""Tests for dashboard progress tracking."""

from __future__ import annotations

from src.dashboard.progress import JobManager, ProgressTracker


def test_progress_tracker_updates_snapshot() -> None:
    tracker = ProgressTracker("job-1", enable_tqdm=False)
    tracker.start_stage("build_tree", "Building", total=10)
    tracker.update(advance=3, message="batch 3", nodes=42, leaves=7)
    snapshot = tracker.snapshot()
    assert snapshot.status == "running"
    assert snapshot.stage == "build_tree"
    assert snapshot.current == 3
    assert snapshot.total == 10
    assert snapshot.percent == 30.0
    assert snapshot.nodes == 42
    assert snapshot.leaves == 7


def test_job_manager_runs_to_completion() -> None:
    manager = JobManager()

    def work(progress: ProgressTracker) -> dict[str, str]:
        progress.start_stage("analyze", "Done", total=1)
        progress.update(advance=1)
        return {"tree_key": "interactive:test"}

    job = manager.start(work, enable_tqdm=False)
    assert job.thread is not None
    job.thread.join()
    data = manager.progress_dict()
    assert data["status"] == "completed"
    assert data["active"] is False
    assert job.result == {"tree_key": "interactive:test"}
