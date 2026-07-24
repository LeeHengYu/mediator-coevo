"""Tests for ArtifactStore idempotency guard."""

import json

import pytest

from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.stores.artifact_store import ArtifactStore


def test_store_trace_first_write_succeeds(tmp_path):
    store = ArtifactStore(base_dir=tmp_path)
    trace = ExecutionTrace(task_id="task1", iteration=0)
    assert store.store_trace(trace).exists()


def test_store_trace_rejects_double_write(tmp_path):
    store = ArtifactStore(base_dir=tmp_path)
    trace = ExecutionTrace(task_id="task1", iteration=0)
    store.store_trace(trace)
    with pytest.raises(FileExistsError):
        store.store_trace(trace)


def test_store_trace_overwrite_true_replaces(tmp_path):
    store = ArtifactStore(base_dir=tmp_path)
    store.store_trace(ExecutionTrace(task_id="task1", iteration=0, reward=0.5))
    store.store_trace(
        ExecutionTrace(task_id="task1", iteration=0, reward=0.9),
        overwrite=True,
    )
    assert store.load_trace("task1", 0).reward == pytest.approx(0.9)


def test_store_report_first_write_succeeds(tmp_path):
    store = ArtifactStore(base_dir=tmp_path)
    report = MediatorReport(task_id="task1", iteration=0)
    assert store.store_report(report).exists()


def test_store_report_rejects_double_write(tmp_path):
    """Same report object passed twice — simulates old Mediator+Orchestrator regression."""
    store = ArtifactStore(base_dir=tmp_path)
    report = MediatorReport(task_id="task1", iteration=0)
    store.store_report(report)
    with pytest.raises(FileExistsError):
        store.store_report(report)


def test_store_report_overwrite_true_replaces(tmp_path):
    store = ArtifactStore(base_dir=tmp_path)
    report = MediatorReport(task_id="task1", iteration=0, content="v1")
    store.store_report(report)
    report.content = "v2"
    store.store_report(report, overwrite=True)
    assert store.query_reports(task_id="task1")[0].content == "v2"


def test_query_traces_filters_by_exact_parsed_task_id(tmp_path):
    store = ArtifactStore(base_dir=tmp_path)
    store.store_trace(ExecutionTrace(task_id="task", iteration=0, reward=0.1))
    store.store_trace(ExecutionTrace(task_id="task-A", iteration=1, reward=0.9))

    traces = store.query_traces(task_id="task", recent=10)

    assert [trace.task_id for trace in traces] == ["task"]


def test_query_reports_filters_by_exact_parsed_task_id(tmp_path):
    store = ArtifactStore(base_dir=tmp_path)
    store.store_report(MediatorReport(task_id="task", iteration=0, content="expected"))
    store.store_report(MediatorReport(task_id="task-A", iteration=1, content="leak"))

    reports = store.query_reports(task_id="task", recent=10)

    assert [report.content for report in reports] == ["expected"]


def test_query_traces_applies_recent_after_exact_filtering(tmp_path):
    store = ArtifactStore(base_dir=tmp_path)
    store.store_trace(ExecutionTrace(task_id="task", iteration=0, reward=0.1))
    store.store_trace(ExecutionTrace(task_id="task-A", iteration=3, reward=0.9))
    store.store_trace(ExecutionTrace(task_id="task", iteration=2, reward=0.3))
    store.store_trace(ExecutionTrace(task_id="task-B", iteration=4, reward=0.8))
    store.store_trace(ExecutionTrace(task_id="task", iteration=1, reward=0.2))

    traces = store.query_traces(task_id="task", recent=2)

    assert [trace.iteration for trace in traces] == [2, 1]


def test_query_traces_applies_iteration_cutoff_before_recent(tmp_path):
    store = ArtifactStore(base_dir=tmp_path)
    store.store_trace(ExecutionTrace(task_id="task", iteration=0, reward=0.1))
    store.store_trace(ExecutionTrace(task_id="task", iteration=1, reward=0.2))
    store.store_trace(ExecutionTrace(task_id="task", iteration=2, reward=0.3))

    traces = store.query_traces(task_id="task", recent=2, before_iteration=2)

    assert [trace.iteration for trace in traces] == [1, 0]


def test_query_traces_skips_malformed_json_after_filtering(tmp_path, caplog):
    store = ArtifactStore(base_dir=tmp_path)
    store.store_trace(ExecutionTrace(task_id="task", iteration=0, reward=0.1))
    malformed = tmp_path / "traces" / "task_iter9999.json"
    malformed.write_text(json.dumps({"task_id": "task", "iteration": "bad"}))

    traces = store.query_traces(task_id="task", recent=10)

    assert [trace.iteration for trace in traces] == [0]
    assert "Failed to load ExecutionTrace" in caplog.text
