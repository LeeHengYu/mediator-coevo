from mediated_coevo.cli.config import _run_config_overrides
from mediated_coevo.core.config import Config
from mediated_coevo.experiment.baselines import (
    BASELINE_PRESET_NAMES,
    BASELINE_PRESETS,
    get_baseline_preset,
)
from tests.config_helpers import (
    budgets_config,
    diffusion_config,
    experiment_config,
    models_config,
)


def _config() -> Config:
    return Config(
        models=models_config(),
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )


def test_matrix_has_four_fixed_skill_diffusion_rows():
    assert BASELINE_PRESET_NAMES == [
        "diffusion_none",
        "capped_broadcast",
        "random_k",
        "top_k_similarity",
    ]
    assert [preset.diffusion_policy for preset in BASELINE_PRESETS] == [
        "none",
        "capped_broadcast",
        "random_k",
        "top_k_similarity",
    ]
    assert all(
        preset.condition_name == "learned_mediator" for preset in BASELINE_PRESETS
    )


def test_matrix_preset_builds_only_diffusion_treatment():
    row = get_baseline_preset("top_k_similarity").build_config(_config(), seed=17)

    assert row.experiment.seed == 17
    assert row.experiment.baseline_preset == "top_k_similarity"
    assert row.diffusion.enabled is True
    assert row.diffusion.policy == "top_k_similarity"
    assert row.diffusion.graph == "task_similarity"


def test_run_overrides_have_no_skill_mutation_controls():
    overrides = _run_config_overrides(
        iterations=4,
        seed=123,
        condition="learned_mediator",
        diffusion_enabled=True,
        diffusion_policy="random_k",
        diffusion_graph="none",
        diffusion_max_artifacts=5,
        diffusion_top_k_neighbors=2,
        harbor_agent_setup_timeout_multiplier=2.5,
    )

    assert overrides["experiment"] == {
        "num_iterations": 4,
        "seed": 123,
        "condition_name": "learned_mediator",
    }
    assert overrides["diffusion"]["policy"] == "random_k"
