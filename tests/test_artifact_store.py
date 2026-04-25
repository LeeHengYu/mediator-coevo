"""Tests for ArtifactStore idempotency guard."""
import pytest
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.models.report import MediatorReport


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
    store.store_trace(ExecutionTrace(task_id="task1", iteration=0, reward=0.9), overwrite=True)
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
