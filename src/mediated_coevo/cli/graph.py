"""Task graph materialization for experiment CLIs."""

from __future__ import annotations

from pathlib import Path

from mediated_coevo.analysis.task_similarity import (
    build_task_graph_precompute,
    write_task_graph_artifacts,
)
from mediated_coevo.benchmarks import SkillFlowRepository
from mediated_coevo.core.config import Config

TASK_SIMILARITY_GRAPH_NAMES = frozenset({"task_similarity", "precomputed_similarity"})
DEFAULT_TASK_GRAPH_EDGE_THRESHOLD = 0.05


def materialize_task_graph_for_diffusion(
    *,
    config: Config,
    experiment_dir: Path,
    benchmark_repo: SkillFlowRepository,
) -> None:
    """Write task graph artifacts when graph-aware diffusion is enabled."""
    if (
        not config.diffusion.enabled
        or config.diffusion.graph not in TASK_SIMILARITY_GRAPH_NAMES
    ):
        return

    output_dir = experiment_dir / "task-graph"
    if output_dir.exists():
        return

    precompute = build_task_graph_precompute(
        benchmark_repo.default_local_cache_dir(),
        edge_score_threshold=DEFAULT_TASK_GRAPH_EDGE_THRESHOLD,
    )
    write_task_graph_artifacts(precompute, output_dir)
