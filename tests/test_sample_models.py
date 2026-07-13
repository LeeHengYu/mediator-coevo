from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mediated_coevo.artifacts.models import ArtifactBankUpdate
from mediated_coevo.diffusion.models import (
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
)
from mediated_coevo.execution.models import (
    TaskExecutionResult,
    TaskProfile,
    empty_context_pack,
)
from mediated_coevo.experiment.sample_models import (
    AgentCallRecord,
    ArchiveEntry,
    ArchiveManifest,
    ExternalArchiveRef,
    OrchestrationArm,
    RuntimeProvenance,
    SampleSpec,
    SequenceRewards,
    SequenceSpec,
    WarmupBundle,
    WarmupExecution,
    WarmupTaskRecord,
    calculate_sequence_rewards,
    sanitize_failure_message,
)
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace


def _task(task_id: str, *, marker: int = 0) -> TaskProfile:
    return TaskProfile(
        task_id=task_id,
        instruction=f"execute {task_id}",
        task_config={"marker": marker, "nested": {"b": 2, "a": 1}},
    )


def _sequence(
    *,
    warmup_count: int = 1,
    reward_weights: tuple[float, ...] | None = None,
) -> SequenceSpec:
    return SequenceSpec(
        sequence_id="sequence-1",
        tasks=(_task("task-a"), _task("task-a", marker=1), _task("task-b")),
        warmup_count=warmup_count,
        policy_seed=17,
        reward_weights=reward_weights,
        task_set_id="static-family-v1",
    )


def _artifact(
    artifact_id: str,
    *,
    run_id: str = "warmup-run-1",
    task_id: str = "task-a",
    position: int = 0,
) -> DiffusionArtifact:
    return DiffusionArtifact(
        artifact_id=artifact_id,
        source_task_id=task_id,
        source_iteration=position,
        source_run_id=run_id,
        artifact_type=DiffusionArtifactType.RUN_OUTCOME,
        risk_level=DiffusionRiskLevel.LOW,
        content=f"outcome for {task_id}",
        verifier_reward=0.0,
    )


def _provenance() -> RuntimeProvenance:
    started = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    return RuntimeProvenance(
        implementation_revision="abc123",
        implementation_dirty=False,
        config_hash="1" * 64,
        graph_implementation_hash="2" * 64,
        policy_implementation_hash="3" * 64,
        harness_hash=None,
        model_mapping={"graph": "fake/graph", "policy": "fake/policy"},
        executor_backend="fake",
        executor_agent="fake-agent",
        python_version="3.13.5",
        package_version="0.1.0",
        started_at=started,
        finished_at=started,
    )


def _warmup_record() -> WarmupTaskRecord:
    artifact = _artifact("artifact-0")
    execution = TaskExecutionResult(
        run_id="warmup-run-1",
        position=0,
        task_id="task-a",
        record=IterationRecord(iteration=0, task_id="task-a", reward=0.0),
        archive_paths=("warmup/warmup-run-1/jobs/position-0000.json",),
        metadata={"phase": "warmup"},
    )
    update = ArtifactBankUpdate(
        run_id="warmup-run-1",
        position=0,
        task_id="task-a",
        before_artifact_ids=(),
        added_artifacts=(artifact,),
        after_artifact_ids=(artifact.artifact_id,),
    )
    return WarmupTaskRecord(
        run_id="warmup-run-1",
        sequence_id="sequence-1",
        position=0,
        task=_task("task-a"),
        artifact_ids_before=(),
        context=empty_context_pack(),
        execution=execution,
        bank_update=update,
    )


def test_sequence_spec_freezes_normalized_profiles_and_allows_duplicate_ids():
    first_config = {"z": [3, 2, 1], "a": {"right": 2, "left": 1}}
    first = TaskProfile(
        task_id="task-a",
        instruction="first occurrence",
        task_config=first_config,
    )
    spec = SequenceSpec(
        sequence_id="sequence-1",
        tasks=(first, first.model_copy(update={"instruction": "second occurrence"})),
        warmup_count=1,
        policy_seed=7,
    )
    first_config["new"] = "must not leak"

    assert spec.task_ids == ("task-a", "task-a")
    assert spec.tasks[0].task_config == {
        "a": {"left": 1, "right": 2},
        "z": (3, 2, 1),
    }
    assert "new" not in spec.tasks[0].task_config
    with pytest.raises(TypeError):
        spec.tasks[0].task_config["mutate"] = True


def test_sequence_spec_revalidates_preconstructed_task_profile_instances():
    valid = _task("task-a")
    invalid = valid.model_copy(update={"task_id": " task-a "})

    with pytest.raises(ValidationError, match="task_id"):
        SequenceSpec(
            sequence_id="sequence-1",
            tasks=(invalid, _task("task-b")),
            warmup_count=1,
            policy_seed=7,
        )

    mutable_config = valid.model_copy(
        update={"task_config": {"nested": {"items": [1, 2]}}}
    )
    normalized = SequenceSpec(
        sequence_id="sequence-1",
        tasks=(mutable_config, _task("task-b")),
        warmup_count=1,
        policy_seed=7,
    )
    assert normalized.tasks[0].task_config == {"nested": {"items": (1, 2)}}
    with pytest.raises(TypeError):
        normalized.tasks[0].task_config["nested"] = {}


@pytest.mark.parametrize(
    "updates",
    [
        {"warmup_count": 3},
        {"reward_weights": (1.0,)},
        {"reward_weights": (-1.0, 1.0)},
        {"reward_weights": (0.0, 0.0)},
    ],
)
def test_sequence_spec_rejects_no_suffix_and_invalid_predeclared_weights(updates):
    payload = _sequence().model_dump()
    payload.update(updates)

    with pytest.raises(ValidationError):
        SequenceSpec.model_validate(payload)


def test_sample_spec_separates_sequence_identity_from_arm_run_identity():
    sequence = _sequence()
    spec = SampleSpec(
        sample_id="sample-full-1",
        sequence=sequence,
        arm=OrchestrationArm.FULL_ORCHESTRATION,
        warmup_bundle_id="a" * 64,
    )

    assert spec.sample_id != spec.sequence.sequence_id
    assert spec.run_id == "sample-full-1"
    assert "arm" not in sequence.model_dump()
    assert "sample_id" not in sequence.model_dump()
    assert "split" not in sequence.model_dump()

    with pytest.raises(ValidationError, match="warmup_bundle_id"):
        SampleSpec(
            sample_id="sample-full-2",
            sequence=sequence,
            arm=OrchestrationArm.FULL_ORCHESTRATION,
        )


@pytest.mark.parametrize("sample_id", ("../escape", "nested/run", "..", "bad\\run"))
def test_sample_spec_rejects_path_like_run_identity(sample_id):
    with pytest.raises(ValidationError, match="path component"):
        SampleSpec(
            sample_id=sample_id,
            sequence=_sequence(warmup_count=0),
            arm=OrchestrationArm.EXECUTION_ONLY,
        )


def test_zero_warmup_sequence_does_not_require_or_accept_a_bundle_reference():
    sequence = _sequence(warmup_count=0)

    spec = SampleSpec(
        sample_id="sample-zero-warmup",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    assert spec.warmup_bundle_id is None

    with pytest.raises(ValidationError, match="warmup_bundle_id"):
        SampleSpec(
            sample_id="sample-zero-warmup-2",
            sequence=sequence,
            arm=OrchestrationArm.EXECUTION_ONLY,
            warmup_bundle_id="a" * 64,
        )


def test_rewards_are_suffix_only_preserve_zero_and_require_complete_coverage():
    sequence = _sequence(reward_weights=(1.0, 3.0))
    rewards = calculate_sequence_rewards(
        sequence=sequence,
        task_rewards=(0.0, 1.0),
        completed_positions=(1, 2),
    )

    assert rewards.positions == (1, 2)
    assert rewards.task_ids == ("task-a", "task-b")
    assert rewards.task_rewards == (0.0, 1.0)
    assert rewards.unweighted_sum == 1.0
    assert rewards.unweighted_mean == 0.5
    assert rewards.weighted_sum == 3.0
    assert rewards.weighted_mean == 0.75
    assert rewards.valid_for_reporting is True

    missing = calculate_sequence_rewards(
        sequence=sequence,
        task_rewards=(0.0, None),
        completed_positions=(1, 2),
    )
    assert missing.task_rewards == (0.0, None)
    assert missing.rewards_complete is False
    assert missing.valid_for_reporting is False
    assert missing.unweighted_sum is None
    assert missing.unweighted_mean is None
    assert missing.weighted_sum is None
    assert missing.weighted_mean is None

    incomplete = calculate_sequence_rewards(
        sequence=sequence,
        task_rewards=(0.0, None),
        completed_positions=(1,),
    )
    assert incomplete.all_tasks_completed is False
    assert incomplete.valid_for_reporting is False

    tampered = rewards.model_dump()
    tampered["scored_count"] = 1
    with pytest.raises(ValidationError, match="coverage counts"):
        type(rewards).model_validate(tampered)

    tampered = rewards.model_dump()
    tampered["weighted_mean"] = 99.0
    with pytest.raises(ValidationError, match="aggregates"):
        type(rewards).model_validate(tampered)


def test_sequence_rewards_rejects_a_reward_for_an_uncompleted_position():
    with pytest.raises(ValidationError, match="uncompleted position"):
        SequenceRewards(
            positions=(1, 2),
            task_ids=("task-a", "task-b"),
            task_rewards=(0.0, 1.0),
            weights=(1.0, 1.0),
            completed_positions=(1,),
            expected_count=2,
            scored_count=2,
            missing_count=0,
            all_tasks_completed=False,
            rewards_complete=True,
            valid_for_reporting=False,
            unweighted_sum=None,
            unweighted_mean=None,
            weighted_sum=None,
            weighted_mean=None,
        )


def test_archive_entries_are_portable_content_addressed_and_unique():
    entry = ArchiveEntry(
        relative_path="warmup/warmup-run-1/artifacts/a.json",
        kind="diffusion_artifact",
        sha256="b" * 64,
        byte_size=12,
    )
    manifest = ArchiveManifest(entries=(entry,))
    assert manifest.entries[0].relative_path.startswith("warmup/")

    for bad_path in ("/tmp/a.json", "../a.json", "warmup/../a.json"):
        with pytest.raises(ValidationError, match="relative"):
            ArchiveEntry(
                relative_path=bad_path,
                kind="artifact",
                sha256="b" * 64,
                byte_size=1,
            )

    with pytest.raises(ValidationError, match="unique"):
        ArchiveManifest(entries=(entry, entry))


def test_warmup_bundle_is_arm_neutral_and_rejects_semantic_hash_mismatch():
    record = _warmup_record()
    manifest = ArchiveManifest()
    bundle = WarmupBundle.create(
        sequence_id="sequence-1",
        warmup_run_id="warmup-run-1",
        warmup_count=1,
        task_records=(record,),
        final_artifact_bank=record.bank_update.added_artifacts,
        archive_manifest=manifest,
        provenance=_provenance(),
    )

    payload = bundle.model_dump(mode="json")
    serialized_record = payload["task_records"][0]
    assert "arm" not in serialized_record
    assert "sample_id" not in serialized_record
    assert payload["bundle_id"] == bundle.semantic_hash()

    payload["task_records"][0]["task"]["instruction"] = "tampered"
    with pytest.raises(ValidationError, match="bundle_id"):
        WarmupBundle.model_validate(payload)


def test_warmup_outputs_bind_final_artifact_objects_to_transitions():
    record = _warmup_record()
    substituted = record.bank_update.added_artifacts[0].model_copy(
        update={"content": "same ID, different artifact"}
    )

    with pytest.raises(ValidationError, match="artifact objects"):
        WarmupBundle.create(
            sequence_id="sequence-1",
            warmup_run_id="warmup-run-1",
            warmup_count=1,
            task_records=(record,),
            final_artifact_bank=(substituted,),
            archive_manifest=ArchiveManifest(),
            provenance=_provenance(),
        )

    with pytest.raises(ValidationError, match="artifact objects"):
        WarmupExecution(
            sequence_id="sequence-1",
            warmup_run_id="warmup-run-1",
            task_records=(record,),
            final_artifact_bank=(substituted,),
        )


def test_failure_message_redacts_bearer_and_quoted_credentials():
    sanitized = sanitize_failure_message(
        'Authorization: Bearer bearer-secret '
        '"api_key": "json-secret" password=password-secret'
    )

    assert "bearer-secret" not in sanitized
    assert "json-secret" not in sanitized
    assert "password-secret" not in sanitized
    assert sanitized.count("[redacted]") == 3


@pytest.mark.parametrize(
    "task_config",
    (
        {"env": ["OPENAI_API_KEY=super-secret"]},
        {"runtime": {"headers": ["Authorization: Bearer bearer-secret"]}},
        {"env": ["AWS_ACCESS_KEY_ID=AKIASECRET"]},
    ),
)
def test_task_profile_rejects_credentials_embedded_in_string_values(task_config):
    with pytest.raises(ValidationError, match="credentials"):
        TaskProfile(task_id="task-a", instruction="execute", task_config=task_config)


def test_task_profile_allows_benign_url_query_configuration():
    profile = TaskProfile(
        task_id="family/task-a",
        instruction="execute",
        task_config={"url": "https://example.com/search?q=public#section"},
    )

    assert profile.task_config["url"].endswith("#section")


@pytest.mark.parametrize("value", ("a/./b", "a//b", "a\\b", "a\x00b"))
def test_portable_archive_paths_reject_noncanonical_values(value):
    with pytest.raises(ValidationError, match="relative POSIX path"):
        ArchiveEntry(
            relative_path=value,
            kind="run_file",
            sha256="1" * 64,
            byte_size=0,
        )

    execution = _warmup_record().execution
    with pytest.raises(ValidationError, match="archive_paths"):
        TaskExecutionResult.model_validate(
            {
                **execution.model_dump(mode="python"),
                "archive_paths": (value,),
            }
        )


def test_external_archive_refs_require_sanitized_absolute_uris():
    with pytest.raises(ValidationError, match="absolute|scheme"):
        ExternalArchiveRef(kind="harbor_job", uri="relative/job")
    with pytest.raises(ValidationError, match="credentials|query"):
        ExternalArchiveRef(
            kind="harbor_job",
            uri="https://user:password@example.com/job?token=secret",
        )

    reference = ExternalArchiveRef(
        kind="harbor_job",
        uri="https://example.com/job",
        provenance={"OPENAI_API_KEY": "must-not-persist"},
    )
    assert "must-not-persist" not in reference.model_dump_json()


@pytest.mark.parametrize(
    "uri",
    ("/tmp/host-secret-value/job", "s3://bucket/host-secret-value/job"),
)
def test_external_archive_refs_reject_host_credentials_in_locators(
    monkeypatch,
    uri,
):
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret-value")

    with pytest.raises(ValidationError, match="credentials"):
        ExternalArchiveRef(kind="harbor_job", uri=uri)


def test_new_json_payload_contracts_are_deeply_immutable_and_redacted():
    call = AgentCallRecord(
        input_payload={"nested": {"value": 1}},
        output_payload={"authorization": "Bearer secret"},
    )
    reference = ExternalArchiveRef(
        kind="remote",
        uri="s3://bucket/key",
        provenance={"nested": {"value": 1}},
    )
    provenance = _provenance()

    for mapping in (
        call.input_payload,
        call.input_payload["nested"],
        reference.provenance,
        reference.provenance["nested"],
        provenance.model_mapping,
    ):
        with pytest.raises(TypeError, match="immutable"):
            mapping["new"] = "value"
    assert "secret" not in call.model_dump_json()


def test_warmup_record_rejects_nested_treatment_arm_metadata():
    record = _warmup_record()
    payload = record.model_dump(mode="python")
    payload["execution"]["metadata"] = {
        "phase": "warmup",
        "orchestration": {"arm": "full_orchestration"},
    }

    with pytest.raises(ValidationError, match="arm-neutral"):
        WarmupTaskRecord.model_validate(payload)


@pytest.mark.parametrize(
    "metadata",
    (
        {"phase": "orchestrated"},
        {"phase": "warmup", "Arm": "full_orchestration"},
        {"phase": "warmup", "treatment_arm": "full_orchestration"},
        {"phase": "warmup", "baseline_preset": "random_policy"},
    ),
)
def test_warmup_record_requires_canonical_arm_neutral_execution_metadata(metadata):
    record = _warmup_record()
    payload = record.model_dump(mode="python")
    payload["execution"]["metadata"] = metadata

    with pytest.raises(ValidationError, match="phase|arm-neutral"):
        WarmupTaskRecord.model_validate(payload)


def test_task_execution_result_redacts_arbitrary_executor_record(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret-value")
    result = TaskExecutionResult(
        run_id="run-1",
        position=0,
        task_id="task-a",
        record=IterationRecord(
            iteration=0,
            task_id="task-a",
            execution_trace=ExecutionTrace(
                task_id="task-a",
                iteration=0,
                status="ok",
                stdout="diagnostic host-secret-value",
                error_detail={"detail": "Bearer raw-bearer-secret"},
            ),
        ),
    )

    encoded = result.model_dump_json()
    assert "host-secret-value" not in encoded
    assert "raw-bearer-secret" not in encoded


def test_task_execution_result_requires_trace_paths_in_archive_paths():
    with pytest.raises(ValidationError, match="declared in archive_paths"):
        TaskExecutionResult(
            run_id="run-1",
            position=0,
            task_id="task-a",
            record=IterationRecord(
                iteration=0,
                task_id="task-a",
                execution_trace=ExecutionTrace(
                    task_id="task-a",
                    iteration=0,
                    status="ok",
                    harbor_paths={"job": "jobs/task-a"},
                ),
            ),
        )


def test_failure_message_redacts_standalone_bearer_and_credential_url():
    sanitized = sanitize_failure_message(
        "Bearer standalone-secret "
        "https://user:password@example.com/path?token=query-secret"
    )

    assert "standalone-secret" not in sanitized
    assert "password" not in sanitized
    assert "query-secret" not in sanitized


def test_every_public_model_rejects_extra_fields_and_carries_schema_version_one():
    sequence = _sequence()
    assert sequence.schema_version == 1

    payload = sequence.model_dump()
    payload["split"] = "test"
    with pytest.raises(ValidationError, match="extra"):
        SequenceSpec.model_validate(payload)
