from __future__ import annotations

import inspect

import pytest
from typer.testing import CliRunner

from mediated_coevo.cli.hl import hl_agent
from mediated_coevo.main import app


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ([], "heuristic-learning experiment runner"),
        (["run"], "one task or a selected benchmark task stream"),
        (["sync"], "inferring known sources"),
        (["list"], "SkillFlow and local benchmark indexes"),
        (["build-base-image"], "shared SkillFlow and OS benchmark base images"),
        (["base-artifacts"], "reusable HL artifact stores"),
        (["sequence"], "Run seeded single-family iterations"),
        (["sequence"], "including warmup"),
        (["sequence"], "Preloaded prefix"),
        (["hl-agent"], "independent offline agent"),
        (["hl-agent"], "execute and analyze"),
    ],
)
def test_cli_help_describes_benchmark_workflow(
    command: list[str],
    expected: str,
) -> None:
    result = CliRunner().invoke(app, [*command, "--help"])

    assert result.exit_code == 0
    assert expected in " ".join(result.output.split())


def test_hl_cli_uses_only_managed_episode_inputs() -> None:
    assert "source_sequence" not in inspect.signature(hl_agent).parameters
