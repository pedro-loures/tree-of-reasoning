"""Local tau-star expansion for subtrees rooted at internal nodes."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

from src.models.hf_runner import HfRunner
from src.tree.tau_builder import TauTreeNode, TauTreeResult, _children_from_logprobs


@dataclass
class TauStarSearchResult:
    tau_star: float
    leaf_count: int
    probes: int
    hit_tau_floor: bool


@dataclass
class SubtreeExpansionResult:
    new_nodes: list[TauTreeNode]
    new_leaf_ids: list[str]
    tau_star: float
    leaves_before: int
    leaves_after: int
    binary_search_probes: int
    hit_tau_floor: bool


def leaf_descendants(node_id: str, nodes_by_id: dict[str, TauTreeNode]) -> list[str]:
    node = nodes_by_id[node_id]
    if not node.child_ids:
        return [node_id]
    leaves: list[str] = []
    for child_id in node.child_ids:
        leaves.extend(leaf_descendants(child_id, nodes_by_id))
    return leaves


def _subtree_node_ids(node_id: str, nodes_by_id: dict[str, TauTreeNode]) -> set[str]:
    ids = {node_id}
    for child_id in nodes_by_id[node_id].child_ids:
        ids.update(_subtree_node_ids(child_id, nodes_by_id))
    return ids


def _next_node_counter(tree: TauTreeResult) -> int:
    max_counter = 0
    for node in tree.nodes:
        match = re.match(r"d\d+_(\d+)$", node.id)
        if match:
            max_counter = max(max_counter, int(match.group(1)))
    return max_counter + 1


def _make_child_node(
    parent: TauTreeNode,
    token_id: int,
    path_prob: float,
    logprob: float,
    token_text: str,
    node_counter: int,
) -> TauTreeNode:
    return TauTreeNode(
        id=f"d{parent.depth + 1}_{node_counter}",
        depth=parent.depth + 1,
        prefix_text=parent.prefix_text + token_text,
        log_path_prob=parent.log_path_prob + logprob,
        path_prob=path_prob,
        parent_id=parent.id,
    )


class _ForwardCache:
    def __init__(self, hf_runner: HfRunner, batch_size: int) -> None:
        self.hf_runner = hf_runner
        self.batch_size = batch_size
        self._cache: dict[str, torch.Tensor] = {}

    def logprobs_batch(self, prefix_texts: list[str]) -> list[torch.Tensor]:
        missing = [prefix for prefix in prefix_texts if prefix not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            for prefix, features in zip(
                batch,
                self.hf_runner.forward_batch(batch, capture_layers=[]),
            ):
                self._cache[prefix] = features.logprobs
        return [self._cache[prefix] for prefix in prefix_texts]


def count_leaves_at_tau(
    hf_runner: HfRunner,
    anchor: TauTreeNode,
    tau: float,
    max_depth: int,
    *,
    numerical_floor: float = 1e-12,
    batch_size: int = 4,
    forward_cache: _ForwardCache | None = None,
) -> int:
    """Count leaf descendants when expanding locally from anchor at tau."""
    log_tau = math.log(tau)
    cache = forward_cache or _ForwardCache(hf_runner, batch_size)
    frontier: list[TauTreeNode] = [anchor]
    leaf_count = 0

    while frontier:
        batch_nodes = frontier
        frontier = []
        prefixes = [node.prefix_text for node in batch_nodes]

        for start in range(0, len(prefixes), batch_size):
            batch = batch_nodes[start : start + batch_size]
            batch_prefixes = prefixes[start : start + batch_size]
            logprob_batch = cache.logprobs_batch(batch_prefixes)

            for parent, logprobs in zip(batch, logprob_batch):
                if parent.depth >= max_depth:
                    leaf_count += 1
                    continue

                children = _children_from_logprobs(
                    parent.log_path_prob,
                    logprobs,
                    log_tau,
                    numerical_floor,
                )
                if not children:
                    leaf_count += 1
                    continue

                for token_id, path_prob in children:
                    token_text = hf_runner.decode_token_id(token_id)
                    logprob = float(logprobs[token_id].item())
                    child = _make_child_node(
                        parent,
                        token_id,
                        path_prob,
                        logprob,
                        token_text,
                        node_counter=0,
                    )
                    frontier.append(child)

    return leaf_count


def binary_search_tau_star(
    hf_runner: HfRunner,
    anchor: TauTreeNode,
    tau_original: float,
    tau_floor: float,
    target_leaves: int,
    max_depth: int,
    *,
    tau_search_epsilon: float = 1e-4,
    numerical_floor: float = 1e-12,
    batch_size: int = 4,
) -> TauStarSearchResult:
    """Find the highest tau in [tau_floor, tau_original] with >= target_leaves."""
    cache = _ForwardCache(hf_runner, batch_size)
    probes = 0

    count_at_original = count_leaves_at_tau(
        hf_runner,
        anchor,
        tau_original,
        max_depth,
        numerical_floor=numerical_floor,
        batch_size=batch_size,
        forward_cache=cache,
    )
    probes += 1
    if count_at_original >= target_leaves:
        return TauStarSearchResult(
            tau_star=tau_original,
            leaf_count=count_at_original,
            probes=probes,
            hit_tau_floor=False,
        )

    count_at_floor = count_leaves_at_tau(
        hf_runner,
        anchor,
        tau_floor,
        max_depth,
        numerical_floor=numerical_floor,
        batch_size=batch_size,
        forward_cache=cache,
    )
    probes += 1
    if count_at_floor < target_leaves:
        return TauStarSearchResult(
            tau_star=tau_floor,
            leaf_count=count_at_floor,
            probes=probes,
            hit_tau_floor=True,
        )

    lo = tau_floor
    hi = tau_original
    best_tau = tau_floor
    best_count = count_at_floor

    while hi - lo > tau_search_epsilon:
        mid = (lo + hi) / 2.0
        count = count_leaves_at_tau(
            hf_runner,
            anchor,
            mid,
            max_depth,
            numerical_floor=numerical_floor,
            batch_size=batch_size,
            forward_cache=cache,
        )
        probes += 1
        if count >= target_leaves:
            best_tau = mid
            best_count = count
            lo = mid
        else:
            hi = mid

    return TauStarSearchResult(
        tau_star=best_tau,
        leaf_count=best_count,
        probes=probes,
        hit_tau_floor=best_tau <= tau_floor + tau_search_epsilon,
    )


def expand_subtree_at_tau(
    hf_runner: HfRunner,
    tree: TauTreeResult,
    anchor_id: str,
    tau: float,
    max_depth: int,
    *,
    numerical_floor: float = 1e-12,
    batch_size: int = 4,
    forward_cache: _ForwardCache | None = None,
) -> SubtreeExpansionResult:
    """Expand the existing subtree at anchor with additional nodes at tau."""
    nodes_by_id = {node.id: node for node in tree.nodes}
    anchor = nodes_by_id[anchor_id]
    leaves_before = len(leaf_descendants(anchor_id, nodes_by_id))
    subtree_ids = _subtree_node_ids(anchor_id, nodes_by_id)

    log_tau = math.log(tau)
    cache = forward_cache or _ForwardCache(hf_runner, batch_size)
    node_counter = _next_node_counter(tree)
    new_nodes: list[TauTreeNode] = []

    token_to_child: dict[tuple[str, int], str] = {}
    for node_id in subtree_ids:
        node = nodes_by_id[node_id]
        if node.parent_id is None:
            continue
        parent = nodes_by_id[node.parent_id]
        for token_id, child_id in zip(parent.child_token_ids, parent.child_ids):
            token_to_child[(parent.id, token_id)] = child_id

    frontier = [nodes_by_id[node_id] for node_id in sorted(subtree_ids, key=lambda nid: nodes_by_id[nid].depth)]
    seen_frontier = set(subtree_ids)

    while frontier:
        batch_nodes = frontier
        frontier = []
        prefixes = [node.prefix_text for node in batch_nodes]

        for start in range(0, len(prefixes), batch_size):
            batch = batch_nodes[start : start + batch_size]
            batch_prefixes = prefixes[start : start + batch_size]
            logprob_batch = cache.logprobs_batch(batch_prefixes)

            for parent, logprobs in zip(batch, logprob_batch):
                if parent.depth >= max_depth:
                    continue

                children = _children_from_logprobs(
                    parent.log_path_prob,
                    logprobs,
                    log_tau,
                    numerical_floor,
                )

                for token_id, path_prob in children:
                    key = (parent.id, token_id)
                    if key in token_to_child:
                        child_id = token_to_child[key]
                        if child_id not in seen_frontier:
                            seen_frontier.add(child_id)
                            frontier.append(nodes_by_id[child_id])
                        continue

                    token_text = hf_runner.decode_token_id(token_id)
                    logprob = float(logprobs[token_id].item())
                    child = _make_child_node(
                        parent,
                        token_id,
                        path_prob,
                        logprob,
                        token_text,
                        node_counter,
                    )
                    node_counter += 1
                    new_nodes.append(child)
                    nodes_by_id[child.id] = child
                    tree.nodes.append(child)
                    token_to_child[key] = child.id
                    subtree_ids.add(child.id)
                    seen_frontier.add(child.id)

                    parent.child_token_ids.append(token_id)
                    parent.child_ids.append(child.id)
                    parent.child_tokens.append(token_text)
                    parent.breadth = len(parent.child_ids)

                    if child.depth < max_depth:
                        frontier.append(child)

    nodes_by_id = {node.id: node for node in tree.nodes}
    leaves_after = len(leaf_descendants(anchor_id, nodes_by_id))
    new_leaf_ids = sorted(node.id for node in new_nodes if not node.child_ids)

    return SubtreeExpansionResult(
        new_nodes=new_nodes,
        new_leaf_ids=new_leaf_ids,
        tau_star=tau,
        leaves_before=leaves_before,
        leaves_after=leaves_after,
        binary_search_probes=0,
        hit_tau_floor=False,
    )


def expand_anchor_to_target(
    hf_runner: HfRunner,
    tree: TauTreeResult,
    anchor_id: str,
    *,
    tau_original: float,
    tau_floor: float,
    tau_search_epsilon: float,
    target_leaves: int,
    max_depth: int,
    numerical_floor: float = 1e-12,
    batch_size: int = 4,
) -> tuple[SubtreeExpansionResult, TauStarSearchResult]:
    nodes_by_id = {node.id: node for node in tree.nodes}
    leaves_before = len(leaf_descendants(anchor_id, nodes_by_id))

    if leaves_before >= target_leaves:
        search = TauStarSearchResult(
            tau_star=tau_original,
            leaf_count=leaves_before,
            probes=0,
            hit_tau_floor=False,
        )
        return (
            SubtreeExpansionResult(
                new_nodes=[],
                new_leaf_ids=[],
                tau_star=tau_original,
                leaves_before=leaves_before,
                leaves_after=leaves_before,
                binary_search_probes=0,
                hit_tau_floor=False,
            ),
            search,
        )

    anchor = nodes_by_id[anchor_id]
    search = binary_search_tau_star(
        hf_runner,
        anchor,
        tau_original,
        tau_floor,
        target_leaves,
        max_depth,
        tau_search_epsilon=tau_search_epsilon,
        numerical_floor=numerical_floor,
        batch_size=batch_size,
    )
    cache = _ForwardCache(hf_runner, batch_size)
    expansion = expand_subtree_at_tau(
        hf_runner,
        tree,
        anchor_id,
        search.tau_star,
        max_depth,
        numerical_floor=numerical_floor,
        batch_size=batch_size,
        forward_cache=cache,
    )
    expansion.binary_search_probes = search.probes
    expansion.hit_tau_floor = search.hit_tau_floor
    expansion.tau_star = search.tau_star
    return expansion, search
