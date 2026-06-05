from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mediated_coevo.analysis.task_similarity import (
    build_task_graph_precompute,
    percentile_threshold,
    write_task_graph_artifacts,
)
from mediated_coevo.main import app


def test_percentile_threshold_uses_nearest_rank() -> None:
    assert percentile_threshold([0.0, 0.1, 0.2, 0.3, 0.4], 0.20) == 0.0
    assert percentile_threshold([0.0, 0.1, 0.2, 0.3, 0.4], 0.50) == 0.2

    with pytest.raises(ValueError, match="percentile"):
        percentile_threshold([0.0], 0.0)


def test_task_graph_precompute_scores_directed_skillflow_edges_and_cuts_below_p20(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    _write_task(
        tasks_root,
        "family-a/task-one",
        family="family-a",
        category="data-analysis",
        difficulty="easy",
        tags=["csv", "analysis"],
        environment_files=["input.csv"],
        instruction="Clean CSV data and write report.json.",
    )
    _write_task(
        tasks_root,
        "family-a/task-two",
        family="family-a",
        category="data-analysis",
        difficulty="medium",
        tags=["csv", "analysis"],
        environment_files=["output.csv"],
        instruction="Transform CSV data and write report.json.",
    )
    _write_task(
        tasks_root,
        "family-a/task-three",
        family="family-a",
        category="document-research",
        difficulty="hard",
        tags=["pdf", "citation"],
        environment_files=["article.pdf"],
        instruction="Review PDF citations and write answer.json.",
    )
    _write_task(
        tasks_root,
        "family-b/task-x",
        family="family-b",
        category="data-analysis",
        difficulty="medium",
        tags=["csv"],
        environment_files=["table.csv"],
        instruction="Analyze CSV input and write report.json.",
    )
    _write_ranking(tasks_root, "family-a", ["task-one", "task-two", "task-three"])
    _write_ranking(tasks_root, "family-b", ["task-x"])

    precompute = build_task_graph_precompute(tasks_root)

    assert precompute.task_count == 4
    assert precompute.pair_count == 3
    assert precompute.kept_edge_count + precompute.cut_edge_count == 3
    assert precompute.components_after_cut == [
        ["family-a/task-one", "family-a/task-three", "family-a/task-two"],
        ["family-b/task-x"],
    ]

    pairs = {
        (pair.source, pair.target): pair for pair in precompute.pairwise_similarity
    }
    forward_pair = pairs[("family-a/task-one", "family-a/task-two")]
    assert ("family-a/task-two", "family-a/task-one") not in pairs
    assert ("family-a/task-one", "family-b/task-x") not in pairs
    assert ("family-b/task-x", "family-a/task-one") not in pairs
    assert forward_pair.score == pytest.approx(1.0)
    assert forward_pair.metadata["edge_kind"] == "same_family_forward"
    assert forward_pair.metadata["rank_gap"] == 1
    assert forward_pair.metadata["rank_affinity"] == pytest.approx(1.0)
    assert forward_pair.components["category"] == 1.0
    assert forward_pair.shared["io_shape"] == ["csv", "json"]

    skipped_pair = pairs[("family-a/task-one", "family-a/task-three")]
    assert skipped_pair.metadata["rank_affinity"] == pytest.approx(0.0)
    assert all(pair.metadata["same_family"] for pair in precompute.pairwise_similarity)

    for pair in precompute.pairwise_similarity:
        if pair.score < precompute.p20_threshold:
            assert pair.kept_after_p20_cut is False
        else:
            assert pair.kept_after_p20_cut is True
        assert pair.kept_after_threshold_cut == pair.kept_after_p20_cut


def test_task_graph_precompute_can_use_absolute_score_threshold(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    _write_task(
        tasks_root,
        "family-a/task-one",
        family="family-a",
        category="data-analysis",
        difficulty="easy",
        tags=["csv", "analysis"],
        environment_files=["input.csv"],
        instruction="Clean CSV data and write report.json.",
    )
    _write_task(
        tasks_root,
        "family-a/task-two",
        family="family-a",
        category="data-analysis",
        difficulty="easy",
        tags=["csv", "analysis"],
        environment_files=["output.csv"],
        instruction="Transform CSV data and write report.json.",
    )
    _write_task(
        tasks_root,
        "family-a/task-three",
        family="family-a",
        category="research",
        difficulty="medium",
        tags=["citation", "bibtex"],
        instruction="Find fake BibTeX citations and write answer.json.",
    )
    _write_ranking(tasks_root, "family-a", ["task-one", "task-two", "task-three"])

    precompute = build_task_graph_precompute(tasks_root, edge_score_threshold=0.7)

    assert precompute.threshold_kind == "absolute_score"
    assert precompute.edge_score_threshold == 0.7
    assert precompute.active_threshold == 0.7
    assert precompute.kept_edge_count == sum(
        pair.score >= 0.7 for pair in precompute.pairwise_similarity
    )

    for pair in precompute.pairwise_similarity:
        assert pair.kept_after_threshold_cut == (pair.score >= 0.7)


def test_write_task_graph_artifacts(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    _write_task(
        tasks_root,
        "family-a/task-a",
        family="family-a",
        category="data",
        difficulty="easy",
        tags=["json"],
        instruction="Write output.json.",
    )
    _write_task(
        tasks_root,
        "family-a/task-b",
        family="family-a",
        category="data",
        difficulty="easy",
        tags=["json"],
        instruction="Write report.json.",
    )
    _write_ranking(tasks_root, "family-a", ["task-a", "task-b"])

    output_dir = tmp_path / "graph"
    precompute = build_task_graph_precompute(tasks_root)
    write_task_graph_artifacts(precompute, output_dir)

    profiles = json.loads((output_dir / "task_profiles.json").read_text())
    pairwise = json.loads((output_dir / "pairwise_similarity.json").read_text())
    summary = json.loads((output_dir / "graph_summary.json").read_text())

    assert profiles["task_count"] == 2
    assert sorted(profiles["profiles"]) == ["family-a/task-a", "family-a/task-b"]
    assert pairwise["graph_kind"] == "skillflow_ranked_similarity"
    assert pairwise["pair_count"] == 1
    assert pairwise["active_threshold"] == pairwise["p20_threshold"]
    assert pairwise["threshold_kind"] == "p20"
    assert pairwise["pairs"][0]["source"] == "family-a/task-a"
    assert pairwise["pairs"][0]["metadata"]["directed"] is True
    assert pairwise["pairs"][0]["metadata"]["same_family"] is True
    assert summary["graph_kind"] == "skillflow_ranked_similarity"
    assert summary["task_count"] == 2
    assert "profiles" not in summary
    assert "pairwise_similarity" not in summary


def test_create_graph_cli_writes_thresholded_artifacts(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    output_dir = tmp_path / "graph"
    _write_task(
        tasks_root,
        "family-a/task-a",
        family="family-a",
        category="Compilation & Build",
        difficulty="easy",
        tags=["build", "ci"],
        instruction="Fix a build failure and write a patch diff.",
    )
    _write_task(
        tasks_root,
        "family-a/task-b",
        family="family-a",
        category="Compilation & Build",
        difficulty="easy",
        tags=["build", "debugging"],
        instruction="Debug a CI build and write a patch diff.",
    )
    _write_task(
        tasks_root,
        "family-b/task-c",
        family="family-b",
        category="research",
        difficulty="medium",
        tags=["citation"],
        instruction="Write answer.json with fake citations.",
    )
    _write_ranking(tasks_root, "family-a", ["task-a", "task-b"])
    _write_ranking(tasks_root, "family-b", ["task-c"])

    result = CliRunner().invoke(
        app,
        [
            "create-graph",
            "--tasks-root",
            str(tasks_root),
            "--output-dir",
            str(output_dir),
            "--threshold",
            "0.05",
        ],
    )

    assert result.exit_code == 0
    assert "Kept edges" in result.output

    summary = json.loads((output_dir / "graph_summary.json").read_text())
    pairwise = json.loads((output_dir / "pairwise_similarity.json").read_text())

    assert summary["task_count"] == 3
    assert summary["pair_count"] == 1
    assert summary["threshold_kind"] == "absolute_score"
    assert summary["active_threshold"] == 0.05
    assert summary["edge_score_threshold"] == 0.05
    assert summary["components_after_cut"] == [
        ["family-a/task-a", "family-a/task-b"],
        ["family-b/task-c"],
    ]
    assert pairwise["active_threshold"] == 0.05
    assert len(pairwise["pairs"]) == 1


def _write_task(
    tasks_root: Path,
    task_id: str,
    *,
    family: str | None = None,
    category: str,
    difficulty: str,
    tags: list[str],
    instruction: str,
    skills: list[str] | None = None,
    environment_files: list[str] | None = None,
) -> None:
    task_dir = tasks_root / task_id
    environment_dir = task_dir / "environment"
    task_dir.mkdir(parents=True)
    environment_dir.mkdir()
    (task_dir / "instruction.md").write_text(instruction)
    family_lines = [f'family = "{family}"'] if family is not None else []
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                'version = "1.0"',
                "",
                "[metadata]",
                *family_lines,
                f'difficulty = "{difficulty}"',
                f'category = "{category}"',
                f"tags = {_toml_list(tags)}",
                "",
            ]
        )
    )
    for skill in skills or []:
        skill_dir = environment_dir / "skills" / skill
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {skill}\n")
    for filename in environment_files or []:
        (environment_dir / filename).write_text("")


def _write_ranking(tasks_root: Path, family_id: str, task_ids: list[str]) -> None:
    family_dir = tasks_root / family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    (family_dir / "ALL_TASK_DIFFICULTY_RANKING.json").write_text(json.dumps(task_ids))


def _toml_list(values: list[str]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"
