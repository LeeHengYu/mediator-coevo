from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from mediated_coevo.agents.swebench_patch_generator import _generation_messages
from mediated_coevo.benchmarks import swebench
from mediated_coevo.benchmarks import swebench_harness_wrapper


def test_build_swebench_harness_command_uses_project_wrapper():
    command = swebench.build_swebench_harness_command(
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="test",
        instance_ids=["sympy__sympy-20590", "django__django-11910"],
        predictions_path="/tmp/predictions.jsonl",
        run_id="run-123",
        timeout=900,
        max_workers=2,
        report_dir="/tmp/reports",
    )

    assert command == [
        sys.executable,
        "-m",
        swebench.SWEBENCH_HARNESS_WRAPPER_MODULE,
        "--dataset_name",
        "SWE-bench/SWE-bench_Lite",
        "--split",
        "test",
        "--predictions_path",
        "/tmp/predictions.jsonl",
        "--run_id",
        "run-123",
        "--timeout",
        "900",
        "--max_workers",
        "2",
        "--report_dir",
        "/tmp/reports",
        "--instance_ids",
        "sympy__sympy-20590",
        "django__django-11910",
        "--modal",
        "true",
    ]


def test_swebench_generation_messages_use_executor_envelope(tmp_path):
    task = swebench.SWEbenchTask(
        task_id="django__django-11910",
        instance={},
        instruction="Fix the regression.",
        task_config={},
        repo="django/django",
        base_commit="base-commit",
    )

    messages = _generation_messages(
        task=task,
        workspace=tmp_path,
        planner_instruction="Planner instruction",
        executor_skill_text="Use targeted reproduction before editing.",
        injected_skill_name="executor",
    )

    user_content = messages[1]["content"]
    assert "# Task Instruction" in user_content
    assert "# Executor Policy" in user_content
    assert "# Task Resources" in user_content
    assert "# Verifier Contract" in user_content
    assert "# Active executor Skill" not in user_content
    assert "Use targeted reproduction before editing." in user_content
    assert "SWE-bench harness" in user_content


def test_harness_wrapper_removes_stale_sympy_branch_mapping(monkeypatch):
    repo_branch_map = {
        "sympy/sympy": {
            "cffd4e0f86fefd4802349a9f9b19ed70934ea354": "1.7",
            "other-base": "master",
        },
        "django/django": {"django-base": "main"},
    }
    constants_module = types.ModuleType("swebench.harness.constants")
    constants_module.REPO_BASE_COMMIT_BRANCH = repo_branch_map
    harness_module = types.ModuleType("swebench.harness")
    harness_module.__path__ = []
    harness_module.constants = constants_module
    swebench_module = types.ModuleType("swebench")
    swebench_module.__path__ = []
    swebench_module.harness = harness_module

    monkeypatch.setitem(sys.modules, "swebench", swebench_module)
    monkeypatch.setitem(sys.modules, "swebench.harness", harness_module)
    monkeypatch.setitem(
        sys.modules,
        "swebench.harness.constants",
        constants_module,
    )

    delegated_modules: list[tuple[str, str | None]] = []

    def fake_run_module(module_name: str, run_name: str | None = None) -> None:
        delegated_modules.append((module_name, run_name))
        assert repo_branch_map["sympy/sympy"] == {"other-base": "master"}
        assert repo_branch_map["django/django"] == {"django-base": "main"}

    clone_patch_calls = []
    monkeypatch.setattr(swebench_harness_wrapper.runpy, "run_module", fake_run_module)
    monkeypatch.setattr(
        swebench_harness_wrapper,
        "_patch_stale_sympy_clone_scope",
        lambda: clone_patch_calls.append("called"),
    )

    swebench_harness_wrapper.main()

    assert clone_patch_calls == ["called"]
    assert delegated_modules == [
        (swebench_harness_wrapper.OFFICIAL_SWEBENCH_RUNNER, "__main__")
    ]


def test_harness_wrapper_removes_single_branch_for_stale_sympy_clone(monkeypatch):
    def fake_make_repo_script_list_py(
        specs,
        repo,
        repo_directory,
        base_commit,
        env_name,
    ) -> list[str]:
        return [
            f"git clone -o origin --single-branch https://github.com/{repo} "
            f"{repo_directory}",
            f"git reset --hard {base_commit}",
        ]

    python_module = types.ModuleType("swebench.harness.test_spec.python")
    python_module.make_repo_script_list_py = fake_make_repo_script_list_py
    create_scripts_module = types.ModuleType("swebench.harness.test_spec.create_scripts")
    create_scripts_module.make_repo_script_list_py = fake_make_repo_script_list_py
    test_spec_module = types.ModuleType("swebench.harness.test_spec")
    test_spec_module.__path__ = []
    test_spec_module.python = python_module
    test_spec_module.create_scripts = create_scripts_module
    harness_module = types.ModuleType("swebench.harness")
    harness_module.__path__ = []
    harness_module.test_spec = test_spec_module
    swebench_module = types.ModuleType("swebench")
    swebench_module.__path__ = []
    swebench_module.harness = harness_module

    monkeypatch.setitem(sys.modules, "swebench", swebench_module)
    monkeypatch.setitem(sys.modules, "swebench.harness", harness_module)
    monkeypatch.setitem(sys.modules, "swebench.harness.test_spec", test_spec_module)
    monkeypatch.setitem(
        sys.modules,
        "swebench.harness.test_spec.python",
        python_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "swebench.harness.test_spec.create_scripts",
        create_scripts_module,
    )

    swebench_harness_wrapper._patch_stale_sympy_clone_scope()

    stale_script = create_scripts_module.make_repo_script_list_py(
        {},
        swebench_harness_wrapper.STALE_SYMPY_REPO,
        "/testbed",
        swebench_harness_wrapper.STALE_SYMPY_BASE_COMMIT,
        "testbed",
    )
    unrelated_script = create_scripts_module.make_repo_script_list_py(
        {},
        "django/django",
        "/testbed",
        "base-commit",
        "testbed",
    )

    assert stale_script[0] == "git clone -o origin https://github.com/sympy/sympy /testbed"
    assert "--single-branch" in unrelated_script[0]


def test_prepare_patch_generation_workspace_retries_transient_clone(
    monkeypatch,
    tmp_path,
):
    task = swebench.SWEbenchTask(
        task_id="sympy__sympy-20590",
        instance={},
        instruction="fix it",
        task_config={},
        repo="sympy/sympy",
        base_commit="base-commit",
    )
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        command = args[0]
        calls.append(command)
        if command[:2] == ["git", "clone"] and len(calls) == 1:
            return subprocess.CompletedProcess(
                command,
                returncode=128,
                stdout="",
                stderr="curl 56 early EOF",
            )
        if command[:2] == ["git", "clone"]:
            Path(command[-1]).mkdir(parents=True)
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    cleaned_paths: list[Path] = []

    def fake_rmtree(path: Path, ignore_errors: bool = False) -> None:
        cleaned_paths.append(path)
        assert ignore_errors is True

    monkeypatch.setattr(swebench.subprocess, "run", fake_run)
    monkeypatch.setattr(swebench.shutil, "rmtree", fake_rmtree)

    repo = swebench.SWEbenchRepository(dataset_name="dataset", split="test")
    workspace = repo.prepare_patch_generation_workspace(
        task=task,
        destination_root=tmp_path,
        planner_instruction="planner instruction",
        injected_skill_text="Executor policy",
        injected_skill_name="executor",
    )

    clone_calls = [call for call in calls if call[:2] == ["git", "clone"]]
    checkout_calls = [call for call in calls if call[:2] == ["git", "checkout"]]
    assert len(clone_calls) == 2
    assert checkout_calls == [["git", "checkout", "base-commit"]]
    assert cleaned_paths == [Path(clone_calls[0][-1])]
    assert workspace == Path(clone_calls[1][-1])
    assert "# Executor Policy" in (workspace / "instruction.md").read_text()
    assert not (workspace / "environment" / "skills" / "executor").exists()


def test_prepare_patch_generation_workspace_does_not_retry_non_transient_clone(
    monkeypatch,
    tmp_path,
):
    task = swebench.SWEbenchTask(
        task_id="django__django-11910",
        instance={},
        instruction="fix it",
        task_config={},
        repo="django/django",
        base_commit="base-commit",
    )
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        command = args[0]
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            returncode=128,
            stdout="",
            stderr="fatal: repository not found",
        )

    cleaned_paths: list[Path] = []
    monkeypatch.setattr(swebench.subprocess, "run", fake_run)
    monkeypatch.setattr(
        swebench.shutil,
        "rmtree",
        lambda path, ignore_errors=False: cleaned_paths.append(path),
    )

    repo = swebench.SWEbenchRepository(dataset_name="dataset", split="test")

    with pytest.raises(RuntimeError, match="repository not found"):
        repo.prepare_patch_generation_workspace(
            task=task,
            destination_root=tmp_path,
            planner_instruction="planner instruction",
            injected_skill_text=None,
            injected_skill_name="executor",
        )

    clone_calls = [call for call in calls if call[:2] == ["git", "clone"]]
    assert clone_calls == [calls[0]]
    assert cleaned_paths == []


def test_aggregate_error_with_modal_setup_logs_is_env_failure(tmp_path):
    run_id = "run-123"
    _write_aggregate_report(
        tmp_path,
        run_id,
        {
            "resolved_ids": [],
            "unresolved_ids": [],
            "empty_patch_ids": [],
            "error_ids": ["sympy__sympy-20590"],
            "incomplete_ids": [],
        },
    )
    harness_run = swebench.SWEbenchHarnessRun(
        command=["python", "-m", "harness"],
        returncode=1,
        stdout="Remote branch 1.7 not found in upstream origin",
        stderr="",
        duration_sec=3.0,
    )

    trace = swebench.build_swebench_traces(
        instance_ids=["sympy__sympy-20590"],
        run_id=run_id,
        project_root=tmp_path,
        harness_run=harness_run,
        raw_output_root=tmp_path,
    )[0]

    assert trace.status == "env_failure"
    assert trace.reward is None
    assert trace.error_kind == swebench.SWEBENCH_MODAL_IMAGE_SETUP_ERROR_KIND
    assert "remote branch not found" in trace.error_detail


def test_aggregate_unresolved_remains_scored_ok(tmp_path):
    run_id = "run-456"
    _write_aggregate_report(
        tmp_path,
        run_id,
        {
            "resolved_ids": [],
            "unresolved_ids": ["django__django-11910"],
            "empty_patch_ids": [],
            "error_ids": [],
            "incomplete_ids": [],
        },
    )
    harness_run = swebench.SWEbenchHarnessRun(
        command=["python", "-m", "harness"],
        returncode=0,
        stdout="",
        stderr="",
        duration_sec=1.0,
    )

    trace = swebench.build_swebench_traces(
        instance_ids=["django__django-11910"],
        run_id=run_id,
        project_root=tmp_path,
        harness_run=harness_run,
        raw_output_root=tmp_path,
    )[0]

    assert trace.status == "ok"
    assert trace.reward == 0.0


def _write_aggregate_report(
    report_root: Path,
    run_id: str,
    report: dict[str, list[str]],
) -> Path:
    report_path = report_root / f"report.{run_id}.json"
    report_path.write_text(json.dumps(report))
    return report_path
