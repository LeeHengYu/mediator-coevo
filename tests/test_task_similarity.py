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


def test_task_graph_precompute_scores_all_pairs_and_cuts_below_p20(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    _write_task(
        tasks_root,
        "java-build",
        category="Compilation & Build",
        difficulty="easy",
        tags=["Java", "maven", "software", "build", "ci", "debugging"],
        skills=["maven-build-lifecycle"],
        instruction="Fix Java Maven build errors and write a patch diff.",
    )
    _write_task(
        tasks_root,
        "python-build",
        category="Compilation & Build",
        difficulty="easy",
        tags=["Python", "software", "build", "ci", "debugging"],
        instruction="Fix Python CI build errors and write a patch diff.",
    )
    _write_task(
        tasks_root,
        "finance-xlsx",
        category="finance-economics",
        difficulty="medium",
        tags=["excel", "financial-analysis"],
        skills=["xlsx"],
        environment_files=["data.xlsx"],
        instruction="Recover missing spreadsheet values in an xlsx workbook.",
    )
    _write_task(
        tasks_root,
        "citation-check",
        category="research",
        difficulty="medium",
        tags=["citation", "bibtex"],
        skills=["citation-management"],
        environment_files=["test.bib"],
        instruction="Find fake BibTeX citations and write answer.json.",
    )

    precompute = build_task_graph_precompute(tasks_root)

    assert precompute.task_count == 4
    assert precompute.pair_count == 6
    assert precompute.kept_edge_count + precompute.cut_edge_count == 6

    pairs = {
        (pair.source, pair.target): pair for pair in precompute.pairwise_similarity
    }
    build_pair = pairs[("java-build", "python-build")]
    assert build_pair.score == max(pair.score for pair in pairs.values())
    assert build_pair.kept_after_p20_cut is True
    assert build_pair.kept_after_threshold_cut is True
    assert build_pair.components["category"] == 1.0
    assert "build-debugging" in build_pair.shared["capabilities"]

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
        "java-build",
        category="Compilation & Build",
        difficulty="easy",
        tags=["Java", "maven", "software", "build", "ci", "debugging"],
        instruction="Fix Java Maven build errors and write a patch diff.",
    )
    _write_task(
        tasks_root,
        "python-build",
        category="Compilation & Build",
        difficulty="easy",
        tags=["Python", "software", "build", "ci", "debugging"],
        instruction="Fix Python CI build errors and write a patch diff.",
    )
    _write_task(
        tasks_root,
        "citation-check",
        category="research",
        difficulty="medium",
        tags=["citation", "bibtex"],
        instruction="Find fake BibTeX citations and write answer.json.",
    )

    precompute = build_task_graph_precompute(tasks_root, edge_score_threshold=0.05)

    assert precompute.threshold_kind == "absolute_score"
    assert precompute.edge_score_threshold == 0.05
    assert precompute.active_threshold == 0.05
    assert precompute.kept_edge_count == sum(
        pair.score >= 0.05 for pair in precompute.pairwise_similarity
    )

    for pair in precompute.pairwise_similarity:
        assert pair.kept_after_threshold_cut == (pair.score >= 0.05)


def test_write_task_graph_artifacts(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    _write_task(
        tasks_root,
        "task-a",
        category="data",
        difficulty="easy",
        tags=["json"],
        instruction="Write output.json.",
    )
    _write_task(
        tasks_root,
        "task-b",
        category="data",
        difficulty="easy",
        tags=["json"],
        instruction="Write report.json.",
    )

    output_dir = tmp_path / "graph"
    precompute = build_task_graph_precompute(tasks_root)
    write_task_graph_artifacts(precompute, output_dir)

    profiles = json.loads((output_dir / "task_profiles.json").read_text())
    pairwise = json.loads((output_dir / "pairwise_similarity.json").read_text())
    summary = json.loads((output_dir / "graph_summary.json").read_text())

    assert profiles["task_count"] == 2
    assert sorted(profiles["profiles"]) == ["task-a", "task-b"]
    assert pairwise["pair_count"] == 1
    assert pairwise["active_threshold"] == pairwise["p20_threshold"]
    assert pairwise["threshold_kind"] == "p20"
    assert pairwise["pairs"][0]["source"] == "task-a"
    assert summary["task_count"] == 2
    assert "profiles" not in summary
    assert "pairwise_similarity" not in summary


def test_create_graph_cli_writes_thresholded_artifacts(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    output_dir = tmp_path / "graph"
    _write_task(
        tasks_root,
        "task-a",
        category="Compilation & Build",
        difficulty="easy",
        tags=["build", "ci"],
        instruction="Fix a build failure and write a patch diff.",
    )
    _write_task(
        tasks_root,
        "task-b",
        category="Compilation & Build",
        difficulty="easy",
        tags=["build", "debugging"],
        instruction="Debug a CI build and write a patch diff.",
    )
    _write_task(
        tasks_root,
        "task-c",
        category="research",
        difficulty="medium",
        tags=["citation"],
        instruction="Write answer.json with fake citations.",
    )

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
    assert summary["pair_count"] == 3
    assert summary["threshold_kind"] == "absolute_score"
    assert summary["active_threshold"] == 0.05
    assert summary["edge_score_threshold"] == 0.05
    assert pairwise["active_threshold"] == 0.05
    assert len(pairwise["pairs"]) == 3


def _write_task(
    tasks_root: Path,
    task_id: str,
    *,
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
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                'version = "1.0"',
                "",
                "[metadata]",
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


def _toml_list(values: list[str]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"
