"""LifelongAgentBench maintenance operations."""

from __future__ import annotations

from pathlib import Path

import typer

from mediated_coevo.benchmarks.lifelong_agent_bench import (
    OS_BASE_IMAGE,
    SUPPORTED_FAMILY,
    LifelongAgentBenchImportError,
    load_lifelong_agent_bench_rows,
    materialize_lifelong_agent_bench,
)
from mediated_coevo.cli.output import console
from mediated_coevo.core.config import DEFAULT_SKILLFLOW_HARBOR_BASE_IMAGE

BENCHMARK_PATH = Path("benchmarks/lifelong_agent_bench")
TASK_INDEX_PATH = Path("docs/lifelong_agent_bench_tasks.txt")


def benchmark_root(project_root: Path) -> Path:
    return project_root / BENCHMARK_PATH


def base_image_commands(
    project_root: Path,
) -> list[list[str]]:
    skillflow_build_script = (
        project_root / "docker" / "harbor-cli-base" / "build.sh"
    )
    context = benchmark_root(project_root) / "docker" / "os-base"
    dockerfile = context / "Dockerfile"
    if not skillflow_build_script.is_file() or not dockerfile.is_file():
        missing = skillflow_build_script if not skillflow_build_script.is_file() else dockerfile
        console.print(f"[bold red]ERROR:[/] missing base-image input: {missing}")
        raise typer.Exit(code=1)
    return [
        [
            "bash",
            str(skillflow_build_script),
            DEFAULT_SKILLFLOW_HARBOR_BASE_IMAGE,
        ],
        [
            "docker",
            "build",
            "--progress=plain",
            "-f",
            str(dockerfile),
            "-t",
            OS_BASE_IMAGE,
            str(context),
        ],
    ]


def sync_tasks(
    *,
    project_root: Path,
    family: str,
    tasks: list[str] | None,
    output_dir: Path | None,
) -> None:
    if tasks:
        raise typer.BadParameter(
            "LifelongAgentBench sync selects one --family, not --tasks"
        )
    if family != SUPPORTED_FAMILY:
        raise typer.BadParameter(
            f"LifelongAgentBench family {family!r} is fidelity-gated; "
            f"sync currently supports {SUPPORTED_FAMILY!r}"
        )
    root = benchmark_root(project_root)
    source = root / "data" / family / "train.jsonl"
    destination = output_dir or root / "tasks"
    try:
        task_dirs = materialize_lifelong_agent_bench(
            family=family,
            rows=load_lifelong_agent_bench_rows(source),
            tasks_root=destination,
        )
    except LifelongAgentBenchImportError as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc
    slug_file = project_root / TASK_INDEX_PATH
    slug_file.parent.mkdir(parents=True, exist_ok=True)
    slug_file.write_text(
        "".join(f"{family}/{path.name}\n" for path in sorted(task_dirs))
    )
    console.print(f"[bold]Synced {len(task_dirs)} {family} tasks to:[/] {destination}")


def list_task_ids(project_root: Path, family: str | None = None) -> list[str]:
    slug_file = project_root / TASK_INDEX_PATH
    if not slug_file.is_file():
        if family is None:
            return []
        raise typer.BadParameter(
            f"local task slug file is missing: {slug_file}; run sync first"
        )
    prefix = f"{family}/" if family else ""
    return [
        slug
        for slug in slug_file.read_text().splitlines()
        if slug.startswith(prefix)
    ]
