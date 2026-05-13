from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from mediated_coevo.core.config import Config
from mediated_coevo.benchmarks import swebench
from mediated_coevo import main as main_module
from mediated_coevo.main import app
from mediated_coevo.models.trace import ExecutionTrace


def test_parse_swebench_instance_ids_defaults_to_smoke_instance():
    assert swebench.parse_swebench_instance_ids(None) == [
        swebench.DEFAULT_SWEBENCH_INSTANCE_ID
    ]


def test_parse_swebench_instance_ids_accepts_repeatable_and_comma_values():
    assert swebench.parse_swebench_instance_ids(
        ["sympy__sympy-20590, django__django-11099", "sympy__sympy-20590"]
    ) == ["sympy__sympy-20590", "django__django-11099"]


def test_list_swebench_instances_uses_dataset_backed_ids(monkeypatch):
    def load_dataset(
        dataset_name: str,
        split: str,
        instance_ids: list[str] | None = None,
    ) -> list[dict]:
        assert dataset_name == "SWE-bench/SWE-bench_Lite"
        assert split == "test"
        assert instance_ids is None
        return [
            {
                "instance_id": "sympy__sympy-20590",
                "repo": "sympy/sympy",
                "version": "1.7",
                "created_at": "2020-01-01",
            },
            {
                "instance_id": "django__django-11099",
                "repo": "django/django",
                "version": "2.2",
            },
        ]

    monkeypatch.setattr(swebench, "load_swebench_dataset", load_dataset)

    instances = swebench.list_swebench_instances(
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="test",
        limit=10,
        repo_filter="sympy",
    )

    assert [instance.instance_id for instance in instances] == ["sympy__sympy-20590"]
    assert instances[0].repo == "sympy/sympy"


def test_resolve_swebench_instance_ids_validates_explicit_ids(monkeypatch):
    captured: dict[str, object] = {}

    def load_dataset(
        dataset_name: str,
        split: str,
        instance_ids: list[str] | None = None,
    ) -> list[dict]:
        captured["dataset_name"] = dataset_name
        captured["split"] = split
        captured["instance_ids"] = instance_ids
        return [{"instance_id": "django__django-11910"}]

    monkeypatch.setattr(swebench, "load_swebench_dataset", load_dataset)

    instance_ids = swebench.resolve_swebench_instance_ids(
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="test",
        raw_instance_ids=["django__django-11910"],
    )

    assert instance_ids == ["django__django-11910"]
    assert captured["instance_ids"] == ["django__django-11910"]


def test_resolve_swebench_instance_ids_fails_when_explicit_id_missing(monkeypatch):
    monkeypatch.setattr(
        swebench,
        "load_swebench_dataset",
        lambda *_, **__: [{"instance_id": "django__django-11910"}],
    )

    try:
        swebench.resolve_swebench_instance_ids(
            dataset_name="SWE-bench/SWE-bench_Lite",
            split="test",
            raw_instance_ids=["django__django-11910", "missing__repo-1"],
        )
    except ValueError as exc:
        assert "missing__repo-1" in str(exc)
    else:
        raise AssertionError("expected missing explicit SWE-bench ID to fail")


def test_resolve_swebench_instance_ids_can_discover_limited_slice(monkeypatch):
    monkeypatch.setattr(
        swebench,
        "list_swebench_instances",
        lambda **_: [
            swebench.SWEbenchInstanceInfo(instance_id="django__django-11910"),
            swebench.SWEbenchInstanceInfo(instance_id="django__django-11099"),
        ],
    )

    assert swebench.resolve_swebench_instance_ids(
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="test",
        raw_instance_ids=None,
        limit=2,
        repo_filter="django",
    ) == ["django__django-11910", "django__django-11099"]


def test_build_swebench_harness_command_uses_modal_and_instance_ids():
    command = swebench.build_swebench_harness_command(
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="test",
        instance_ids=["sympy__sympy-20590"],
        predictions_path="gold",
        run_id="smoke-run",
        timeout=1800,
        max_workers=1,
        report_dir=".",
    )

    assert command[:3] == [sys.executable, "-m", "swebench.harness.run_evaluation"]
    assert command[-3:] == ["sympy__sympy-20590", "--modal", "true"]
    assert command[command.index("--report_dir") + 1] == "."
    assert command.index("--report_dir") < command.index("--instance_ids")
    assert "--instance_ids" in command


def test_build_swebench_traces_maps_resolved_report_to_reward(tmp_path):
    _write_report(tmp_path, "smoke-run", "gold", "sympy__sympy-20590", True)
    stdout = "x" * 9000
    stderr = "y" * 9000
    harness_run = swebench.SWEbenchHarnessRun(
        command=["python", "-m", "swebench.harness.run_evaluation"],
        returncode=0,
        stdout=stdout,
        stderr=stderr,
        duration_sec=2.5,
    )

    trace = swebench.build_swebench_traces(
        instance_ids=["sympy__sympy-20590"],
        run_id="smoke-run",
        project_root=tmp_path,
        harness_run=harness_run,
    )[0]

    assert trace.task_id == "sympy__sympy-20590"
    assert trace.reward == 1.0
    assert trace.status == "ok"
    assert trace.test_results == {"resolved": True}
    assert trace.stdout == stdout
    assert trace.stderr == stderr


def test_build_swebench_traces_reads_reports_from_raw_output_root(tmp_path):
    raw_output_root = tmp_path / "raw"
    _write_report(raw_output_root, "smoke-run", "gold", "sympy__sympy-20590", True)
    harness_run = swebench.SWEbenchHarnessRun(
        command=["python", "-m", "swebench.harness.run_evaluation"],
        returncode=0,
        stdout="ok",
        stderr="",
        duration_sec=2.5,
    )

    trace = swebench.build_swebench_traces(
        instance_ids=["sympy__sympy-20590"],
        run_id="smoke-run",
        project_root=tmp_path / "project",
        harness_run=harness_run,
        raw_output_root=raw_output_root,
    )[0]

    assert trace.reward == 1.0
    assert trace.harbor_paths["swebench_report"].startswith(str(raw_output_root))


def test_build_swebench_traces_uses_aggregate_report_for_empty_instance_report(
    tmp_path,
):
    instance_id = "django__django-11910"
    run_id = "smoke-run"
    model = "openrouter__google__gemini-2.5-flash"
    instance_report_path = _write_empty_report(tmp_path, run_id, model, instance_id)
    aggregate_report_path = _write_aggregate_report(
        tmp_path,
        run_id,
        model,
        {
            "total_instances": 1,
            "submitted_instances": 1,
            "completed_instances": 1,
            "resolved_instances": 0,
            "unresolved_instances": 0,
            "empty_patch_instances": 0,
            "error_instances": 1,
            "resolved_ids": [],
            "unresolved_ids": [],
            "empty_patch_ids": [],
            "error_ids": [instance_id],
            "incomplete_ids": [],
            "submitted_ids": [instance_id],
            "schema_version": 2,
        },
    )
    harness_run = swebench.SWEbenchHarnessRun(
        command=["python", "-m", "swebench.harness.run_evaluation"],
        returncode=0,
        stdout="patch apply failed",
        stderr="",
        duration_sec=2.5,
    )

    trace = swebench.build_swebench_traces(
        instance_ids=[instance_id],
        run_id=run_id,
        project_root=tmp_path,
        harness_run=harness_run,
    )[0]

    assert trace.reward == 0.0
    assert trace.status == "ok"
    assert trace.test_results is not None
    assert trace.test_results["aggregate_outcome"] == "error"
    assert "empty report.json" in trace.test_results["instance_report_error"]
    assert trace.harbor_paths["swebench_report"] == str(instance_report_path)
    assert trace.harbor_paths["swebench_aggregate_report"] == str(aggregate_report_path)


def test_build_swebench_traces_uses_aggregate_report_when_instance_report_missing(
    tmp_path,
):
    instance_id = "django__django-11910"
    run_id = "smoke-run"
    model = "gold"
    aggregate_report_path = _write_aggregate_report(
        tmp_path,
        run_id,
        model,
        {
            "resolved_ids": [],
            "unresolved_ids": [instance_id],
            "empty_patch_ids": [],
            "error_ids": [],
            "incomplete_ids": [],
            "submitted_ids": [instance_id],
            "schema_version": 2,
        },
    )
    harness_run = swebench.SWEbenchHarnessRun(
        command=["python", "-m", "swebench.harness.run_evaluation"],
        returncode=0,
        stdout="ok",
        stderr="",
        duration_sec=2.5,
    )

    trace = swebench.build_swebench_traces(
        instance_ids=[instance_id],
        run_id=run_id,
        project_root=tmp_path,
        harness_run=harness_run,
    )[0]

    assert trace.reward == 0.0
    assert trace.status == "ok"
    assert trace.test_results is not None
    assert trace.test_results["aggregate_outcome"] == "unresolved"
    assert "swebench_report" not in trace.harbor_paths
    assert trace.harbor_paths["swebench_aggregate_report"] == str(aggregate_report_path)


def test_build_swebench_traces_maps_unresolved_report_to_zero(tmp_path):
    _write_report(tmp_path, "smoke-run", "gold", "sympy__sympy-20590", False)
    harness_run = swebench.SWEbenchHarnessRun(
        command=["python", "-m", "swebench.harness.run_evaluation"],
        returncode=0,
        stdout="ok",
        stderr="",
        duration_sec=2.5,
    )

    trace = swebench.build_swebench_traces(
        instance_ids=["sympy__sympy-20590"],
        run_id="smoke-run",
        project_root=tmp_path,
        harness_run=harness_run,
    )[0]

    assert trace.reward == 0.0
    assert trace.status == "ok"


def test_build_swebench_traces_marks_missing_report_as_env_failure(tmp_path):
    stdout = "x" * 9000
    stderr = "y" * 9000
    harness_run = swebench.SWEbenchHarnessRun(
        command=["python", "-m", "swebench.harness.run_evaluation"],
        returncode=1,
        stdout=stdout,
        stderr=stderr,
        duration_sec=1.0,
    )

    trace = swebench.build_swebench_traces(
        instance_ids=["sympy__sympy-20590"],
        run_id="smoke-run",
        project_root=tmp_path,
        harness_run=harness_run,
    )[0]

    assert trace.reward is None
    assert trace.status == "env_failure"
    assert trace.error_kind == "harness_failed"
    assert trace.stdout == stdout
    assert trace.stderr == stderr


def test_write_swebench_eval_outputs_writes_traces_and_summary(tmp_path):
    traces = [
        ExecutionTrace(
            task_id="sympy__sympy-20590",
            iteration=0,
            reward=1.0,
            status="ok",
            run_id="smoke-run",
        )
    ]

    traces_path, summary_path = swebench.write_swebench_eval_outputs(
        traces=traces,
        output_dir=tmp_path / "out",
        run_id="smoke-run",
    )

    assert traces_path.exists()
    assert summary_path.exists()
    assert json.loads(traces_path.read_text())["reward"] == 1.0
    assert json.loads(summary_path.read_text())["mean_reward"] == 1.0


def test_swebench_list_instances_cli_prints_dataset_ids(monkeypatch):
    monkeypatch.setattr(
        swebench,
        "list_swebench_instances",
        lambda **_: [
            swebench.SWEbenchInstanceInfo(
                instance_id="sympy__sympy-20590",
                repo="sympy/sympy",
                version="1.7",
                created_at="2020-01-01",
            )
        ],
    )

    result = CliRunner().invoke(app, ["swebench", "list-instances", "--limit", "1"])

    assert result.exit_code == 0
    assert "sympy__sympy-20590" in result.stdout
    assert "sympy/sympy" in result.stdout


def test_swebench_smoke_cli_runs_harness_and_writes_outputs(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def run_harness(
        *,
        command: list[str],
        cwd: Path,
        stream_output: bool = False,
    ) -> swebench.SWEbenchHarnessRun:
        captured["command"] = command
        captured["cwd"] = cwd
        captured["stream_output"] = stream_output
        return swebench.SWEbenchHarnessRun(
            command=command,
            returncode=0,
            stdout="ok",
            stderr="",
            duration_sec=1.0,
        )

    monkeypatch.setattr(
        swebench,
        "resolve_swebench_instance_ids",
        lambda **_: ["sympy__sympy-20590", "django__django-11099"],
    )
    monkeypatch.setattr(swebench, "ensure_swebench_available", lambda: None)
    monkeypatch.setattr(swebench, "validate_modal_credentials", lambda: None)
    monkeypatch.setattr(swebench, "run_swebench_harness", run_harness)
    monkeypatch.setattr(
        swebench,
        "build_swebench_traces",
        lambda **_: [
            ExecutionTrace(
                task_id="sympy__sympy-20590",
                iteration=0,
                reward=1.0,
                status="ok",
                run_id="smoke-run",
            )
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "swebench",
            "smoke",
            "--instance-id",
            "sympy__sympy-20590,django__django-11099",
            "--run-id",
            "smoke-run",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert "SWE-bench eval outputs" in result.stdout
    command = captured["command"]
    assert isinstance(command, list)
    assert captured["stream_output"] is True
    assert captured["cwd"] == tmp_path / "out" / "smoke-run" / "raw"
    assert command[command.index("--report_dir") + 1] == "."
    assert command[-4:] == [
        "sympy__sympy-20590",
        "django__django-11099",
        "--modal",
        "true",
    ]
    assert (tmp_path / "out" / "smoke-run" / "traces.jsonl").exists()
    assert (tmp_path / "out" / "smoke-run" / "raw").exists()


def test_swebench_run_cli_requires_evolution_and_eval_selection():
    result = CliRunner().invoke(app, ["swebench", "run"])

    assert result.exit_code != 0
    assert "provide --evolve-instance-id or --evolve-limit" in result.stdout


def test_swebench_run_cli_fails_when_evolve_and_eval_overlap(monkeypatch):
    def resolve_ids(**kwargs) -> list[str]:
        raw_instance_ids = kwargs["raw_instance_ids"]
        assert isinstance(raw_instance_ids, list)
        return swebench.parse_swebench_instance_ids(raw_instance_ids)

    monkeypatch.setattr(swebench, "resolve_swebench_instance_ids", resolve_ids)

    result = CliRunner().invoke(
        app,
        [
            "swebench",
            "run",
            "--evolve-instance-id",
            "sympy__sympy-20590",
            "--eval-instance-id",
            "sympy__sympy-20590",
        ],
    )

    assert result.exit_code != 0
    assert "must be disjoint" in result.stdout


def test_swebench_run_cli_delegates_integrated_experiment(monkeypatch):
    captured: dict[str, object] = {}

    def resolve_ids(**kwargs) -> list[str]:
        raw_instance_ids = kwargs["raw_instance_ids"]
        assert isinstance(raw_instance_ids, list)
        return swebench.parse_swebench_instance_ids(raw_instance_ids)

    def run_experiment(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(swebench, "resolve_swebench_instance_ids", resolve_ids)
    monkeypatch.setattr(
        main_module,
        "_run_swebench_experiment",
        run_experiment,
    )

    result = CliRunner().invoke(
        app,
        [
            "swebench",
            "run",
            "--evolve-instance-id",
            "django__django-11910",
            "--eval-instance-id",
            "django__django-11099",
            "--iterations",
            "2",
            "--coevo-interval",
            "2",
            "--advisor-buffer-max",
            "1",
            "--no-skill-validation",
        ],
    )

    assert result.exit_code == 0
    assert captured["evolve_instance_ids"] == ["django__django-11910"]
    assert captured["eval_instance_ids"] == ["django__django-11099"]
    assert captured["iterations"] == 2
    assert captured["coevo_interval"] == 2
    assert captured["advisor_buffer_max"] == 1
    assert captured["skill_validation_enabled"] is False


def test_common_experiment_override_flags_are_shared_by_benchmark_clis():
    runner = CliRunner()

    cases = (
        (["run"], "provide --tasks or --task-set"),
        (["matrix"], "provide --tasks or --task-set"),
        (["swebench", "run"], "provide --evolve-instance-id or --evolve-limit"),
    )
    for base_command, expected_error in cases:
        command = [
            *base_command,
            "--coevo-interval",
            "2",
            "--advisor-buffer-max",
            "1",
            "--skill-validation",
        ]
        result = runner.invoke(app, command)

        assert expected_error in result.output
        assert "No such option" not in result.output

        no_validation_result = runner.invoke(
            app,
            [*command[:-1], "--no-skill-validation"],
        )
        assert expected_error in no_validation_result.output
        assert "No such option" not in no_validation_result.output


def test_swebench_runtime_backend_is_persisted(tmp_path, monkeypatch):
    for skill_name in ("executor", "planner", "mediator"):
        skill_dir = tmp_path / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {skill_name}\n")

    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
        }
    )
    config.paths.skills_dir = "skills"
    config.paths.data_dir = "data"
    config.executor_runtime.backend = "skillsbench"

    main_module._apply_experiment_settings(
        config,
        iterations=5,
        seed=7,
        coevo_interval=2,
        advisor_buffer_max=1,
        skill_validation_enabled=False,
    )
    main_module._force_swebench_runtime(config)
    experiment_dir, _ = main_module._prepare_swebench_experiment_root(
        config=config,
        seed=7,
        run_id="swebench-test",
    )

    saved = tomllib.loads((experiment_dir / "config.toml").read_text())
    assert saved["executor_runtime"]["backend"] == "swebench"
    assert saved["experiment"]["coevo_interval"] == 2
    assert saved["experiment"]["advisor_buffer_max"] == 1
    assert saved["experiment"]["skill_validation"]["enabled"] is False


def test_swebench_experiment_does_not_check_harbor_availability(
    tmp_path,
    monkeypatch,
):
    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
        }
    )

    class _FakeOrchestrator:
        async def run_experiment(
            self,
            task_ids: list[str],
            iterations: int,
        ) -> list:
            return []

    async def run_frozen_eval(**kwargs) -> list[ExecutionTrace]:
        return [
            ExecutionTrace(
                task_id="django__django-11099",
                iteration=0,
                reward=0.0,
                status="ok",
            )
        ]

    def fail_harbor_check(_: Config) -> None:
        raise AssertionError("SWE-bench should not check Harbor availability")

    monkeypatch.setattr(main_module, "load_config", lambda _: config)
    monkeypatch.setattr(main_module, "_validate_or_raise_bad_parameter", lambda _: None)
    monkeypatch.setattr(main_module, "_ensure_harbor_available", fail_harbor_check)
    monkeypatch.setattr(swebench, "validate_modal_credentials", lambda: None)
    monkeypatch.setattr(
        main_module,
        "_prepare_swebench_experiment_root",
        lambda **_: (tmp_path / "experiment", tmp_path / "skills"),
    )
    monkeypatch.setattr(
        main_module,
        "_build_swebench_orchestrator",
        lambda **_: _FakeOrchestrator(),
    )
    monkeypatch.setattr(main_module, "_run_swebench_frozen_eval", run_frozen_eval)
    monkeypatch.setattr(
        main_module, "_write_and_print_result_summary", lambda **_: None
    )
    monkeypatch.setattr(
        main_module,
        "_write_swebench_phase_outputs",
        lambda **_: (
            tmp_path / "traces.jsonl",
            tmp_path / "predictions.jsonl",
            tmp_path / "summary.json",
            {},
        ),
    )
    monkeypatch.setattr(main_module, "write_score_summary", lambda *_, **__: None)
    monkeypatch.setattr(main_module, "_print_result_summary", lambda **_: None)

    main_module._run_swebench_experiment(
        evolve_instance_ids=["django__django-11910"],
        eval_instance_ids=["django__django-11099"],
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="test",
        iterations=1,
        seed=7,
        condition_name="learned_mediator",
        skill_updates=config.experiment.skill_updates,
        config_dir=tmp_path / "config",
        timeout=30,
        max_workers=1,
        run_id="swebench-test",
    )


def _write_report(
    root: Path,
    run_id: str,
    model: str,
    instance_id: str,
    resolved: bool,
) -> None:
    report_dir = root / "logs" / "run_evaluation" / run_id / model / instance_id
    report_dir.mkdir(parents=True)
    report = {instance_id: {"resolved": resolved}}
    (report_dir / "report.json").write_text(json.dumps(report))


def _write_empty_report(
    root: Path,
    run_id: str,
    model: str,
    instance_id: str,
) -> Path:
    report_dir = root / "logs" / "run_evaluation" / run_id / model / instance_id
    report_dir.mkdir(parents=True)
    report_path = report_dir / "report.json"
    report_path.write_text("")
    return report_path


def _write_aggregate_report(
    root: Path,
    run_id: str,
    model: str,
    report: dict,
) -> Path:
    report_path = root / f"{model}.{run_id}.json"
    report_path.write_text(json.dumps(report))
    return report_path
