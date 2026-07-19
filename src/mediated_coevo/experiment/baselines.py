"""Fixed-skill diffusion matrix presets."""

from __future__ import annotations

from dataclasses import dataclass

from mediated_coevo.core.config import Config, DiffusionPolicyName
from mediated_coevo.experiment.conditions import ConditionName


@dataclass(frozen=True)
class BaselinePreset:
    """One row in the learned-mediator diffusion matrix."""

    name: str
    condition_name: ConditionName
    diffusion_enabled: bool
    diffusion_policy: DiffusionPolicyName
    diffusion_graph: str = "none"

    def build_config(self, base_config: Config, *, seed: int) -> Config:
        """Return a row-local config copy with this preset applied."""
        row_config = base_config.model_copy(deep=True)
        row_config.experiment.seed = seed
        row_config.experiment.condition_name = self.condition_name
        row_config.experiment.baseline_preset = self.name
        row_config.diffusion.enabled = self.diffusion_enabled
        row_config.diffusion.policy = self.diffusion_policy
        row_config.diffusion.graph = self.diffusion_graph
        return row_config


def _learned_mediator_diffusion_preset(
    name: str,
    *,
    diffusion_policy: DiffusionPolicyName,
) -> BaselinePreset:
    return BaselinePreset(
        name=name,
        condition_name="learned_mediator",
        diffusion_enabled=diffusion_policy != "none",
        diffusion_policy=diffusion_policy,
        diffusion_graph=(
            "task_similarity" if diffusion_policy == "top_k_similarity" else "none"
        ),
    )


BASELINE_PRESETS: tuple[BaselinePreset, ...] = (
    _learned_mediator_diffusion_preset(
        "diffusion_none",
        diffusion_policy="none",
    ),
    _learned_mediator_diffusion_preset(
        "capped_broadcast",
        diffusion_policy="capped_broadcast",
    ),
    _learned_mediator_diffusion_preset(
        "random_k",
        diffusion_policy="random_k",
    ),
    _learned_mediator_diffusion_preset(
        "top_k_similarity",
        diffusion_policy="top_k_similarity",
    ),
)

BASELINE_PRESETS_BY_NAME = {preset.name: preset for preset in BASELINE_PRESETS}
BASELINE_PRESET_NAMES = [preset.name for preset in BASELINE_PRESETS]


def get_baseline_preset(preset_name: str) -> BaselinePreset:
    """Return a named matrix preset or raise a user-facing ValueError."""
    try:
        return BASELINE_PRESETS_BY_NAME[preset_name]
    except KeyError as exc:
        allowed = ", ".join(BASELINE_PRESET_NAMES)
        raise ValueError(
            f"invalid baseline preset {preset_name!r}; expected one of: {allowed}"
        ) from exc
