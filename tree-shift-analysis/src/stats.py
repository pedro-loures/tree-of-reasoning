"""Correlation and partial-correlation statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.features import layer_columns


def _spearman(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return {"rho": None, "pvalue": None, "n": len(valid)}
    rho, pvalue = stats.spearmanr(valid["x"], valid["y"])
    return {"rho": float(rho), "pvalue": float(pvalue), "n": int(len(valid))}


def _partial_spearman(x: pd.Series, y: pd.Series, control: pd.Series) -> dict[str, float | int | None]:
    frame = pd.DataFrame({"x": x, "y": y, "control": control}).dropna()
    if len(frame) < 4:
        return {"rho": None, "pvalue": None, "n": len(frame)}
    x_rank = frame["x"].rank()
    y_rank = frame["y"].rank()
    control_rank = frame["control"].rank()
    residual_x = x_rank - control_rank
    residual_y = y_rank - control_rank
    rho, pvalue = stats.spearmanr(residual_x, residual_y)
    return {"rho": float(rho), "pvalue": float(pvalue), "n": int(len(frame))}


def _depth_bin_label(depth: int, depth_bins: list[list[int]]) -> str:
    for low, high in depth_bins:
        if low <= depth <= high:
            return f"{low}-{high}"
    return "other"


def compute_correlations(
    df: pd.DataFrame,
    depth_bins: list[list[int]] | None = None,
) -> dict[str, Any]:
    if df.empty:
        return {
            "overall": {},
            "by_layer": {},
            "by_model": {},
            "by_prefix_length": {},
            "by_depth_bin": {},
            "by_country": {},
            "by_region": {},
        }

    depth_bins = depth_bins or [[1, 5], [6, 10], [11, 15], [16, 20], [21, 999]]
    layer_cols = layer_columns(df)
    targets = {
        "breadth": "breadth",
        "top1_prob": "top1_prob",
        "topk_entropy": "topk_entropy",
    }

    overall: dict[str, Any] = {}
    by_layer: dict[str, Any] = {}
    by_model: dict[str, Any] = {}
    by_prefix_length: dict[str, Any] = {}
    by_depth_bin: dict[str, Any] = {}
    by_country: dict[str, Any] = {}
    by_region: dict[str, Any] = {}

    for layer_col in layer_cols:
        layer_name = layer_col.replace("cos_dist_", "")
        by_layer[layer_name] = {}
        for target_name, target_col in targets.items():
            by_layer[layer_name][target_name] = _spearman(df[layer_col], df[target_col])
            by_layer[layer_name][f"{target_name}_partial_depth"] = _partial_spearman(
                df[layer_col], df[target_col], df["depth"]
            )

    stacked = pd.DataFrame(
        {
            "cos_dist": pd.concat([df[col] for col in layer_cols], ignore_index=True),
            "layer": np.repeat([col.replace("cos_dist_l", "") for col in layer_cols], len(df)),
        }
    )
    # rebuild aligned target columns for stacked rows
    for target_name, target_col in targets.items():
        stacked[target_name] = pd.concat([df[target_col]] * len(layer_cols), ignore_index=True)
        overall[target_name] = _spearman(stacked["cos_dist"], stacked[target_name])
        overall[f"{target_name}_partial_depth"] = _partial_spearman(
            stacked["cos_dist"], stacked[target_name], pd.concat([df["depth"]] * len(layer_cols), ignore_index=True)
        )

    for model_id, group in df.groupby("model_id"):
        by_model[model_id] = {}
        for layer_col in layer_cols:
            layer_name = layer_col.replace("cos_dist_", "")
            by_model[model_id][layer_name] = _spearman(group[layer_col], group["breadth"])

    legacy = df[df["instruction_variant"] == "legacy"]
    for prefix_length, group in legacy.groupby("prefix_length"):
        by_prefix_length[str(prefix_length)] = {}
        for layer_col in layer_cols:
            layer_name = layer_col.replace("cos_dist_", "")
            by_prefix_length[str(prefix_length)][layer_name] = _spearman(group[layer_col], group["breadth"])

    frame = df.copy()
    frame["depth_bin"] = frame["depth"].map(lambda depth: _depth_bin_label(int(depth), depth_bins))
    for depth_bin, group in frame.groupby("depth_bin"):
        by_depth_bin[depth_bin] = {}
        for layer_col in layer_cols:
            layer_name = layer_col.replace("cos_dist_", "")
            by_depth_bin[depth_bin][layer_name] = _spearman(group[layer_col], group["breadth"])

    if "country_id" in df.columns:
        for country_id, group in df.dropna(subset=["country_id"]).groupby("country_id"):
            by_country[str(country_id)] = {}
            for layer_col in layer_cols:
                layer_name = layer_col.replace("cos_dist_", "")
                by_country[str(country_id)][layer_name] = _spearman(group[layer_col], group["breadth"])

    if "region_id" in df.columns:
        for region_id, group in df.dropna(subset=["region_id"]).groupby("region_id"):
            by_region[str(region_id)] = {}
            for layer_col in layer_cols:
                layer_name = layer_col.replace("cos_dist_", "")
                by_region[str(region_id)][layer_name] = _spearman(group[layer_col], group["breadth"])

    layer_summary = {
        layer_col.replace("cos_dist_l", ""): by_layer[layer_col.replace("cos_dist_", "")]["breadth"]["rho"]
        for layer_col in layer_cols
    }

    return {
        "overall": overall,
        "by_layer": by_layer,
        "by_model": by_model,
        "by_prefix_length": by_prefix_length,
        "by_depth_bin": by_depth_bin,
        "by_country": by_country,
        "by_region": by_region,
        "layer_breadth_rho": layer_summary,
    }
