"""Thin orchestration layer over the experiment runner."""

from __future__ import annotations

from pathlib import Path

from src.experiment.runner import run_experiment


class ExecutionPipeline:
    """Run experiments and write tau-tree results to JSONL."""

    def __init__(
        self,
        config_path: Path,
        results_dir: Path,
        repo_root: Path | None = None,
    ) -> None:
        self.config_path = config_path
        self.results_dir = results_dir
        self.repo_root = repo_root

    def run(
        self,
        model_ids: list[str] | None = None,
        prefix_lengths: list[int] | None = None,
        seeds: list[int] | None = None,
        country_ids: list[str] | None = None,
    ) -> list[Path]:
        return run_experiment(
            config_path=self.config_path,
            results_dir=self.results_dir,
            model_ids=model_ids,
            prefix_lengths=prefix_lengths,
            seeds=seeds,
            country_ids=country_ids,
            repo_root=self.repo_root,
        )
