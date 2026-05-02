"""Baseline matrix presets and skill-update policy parsing."""

from __future__ import annotations

from dataclasses import dataclass

from mediated_coevo.conditions import ConditionName
from mediated_coevo.config import Config, SkillUpdateConfig


SKILL_UPDATE_TOKENS = frozenset({"none", "executor", "planner", "mediator", "all"})


class SkillUpdateParseError(ValueError):
    """Raised when a CLI skill-update policy cannot be parsed."""


@dataclass(frozen=True)
class BaselinePreset:
    """One row in the baseline matrix."""

    name: str
    condition_name: ConditionName
    skill_updates: SkillUpdateConfig

    def build_config(self, base_config: Config, *, seed: int) -> Config:
        """Return a row-local config copy with this preset applied."""
        row_config = base_config.model_copy(deep=True)
        row_config.experiment.seed = seed
        row_config.experiment.condition_name = self.condition_name
        row_config.experiment.skill_updates = self.skill_updates.model_copy(deep=True)
        row_config.experiment.baseline_preset = self.name
        return row_config


def skill_updates_config(*enabled: str) -> SkillUpdateConfig:
    """Build a policy object with exactly the named roles enabled."""
    enabled_set = set(enabled)
    return SkillUpdateConfig(
        executor="executor" in enabled_set,
        planner="planner" in enabled_set,
        mediator="mediator" in enabled_set,
    )


BASELINE_PRESETS: tuple[BaselinePreset, ...] = (
    BaselinePreset("no_feedback", "no_feedback", skill_updates_config()),
    BaselinePreset("full_trace_same_task", "full_traces", skill_updates_config()),
    BaselinePreset(
        "static_mediator_same_task",
        "static_mediator",
        skill_updates_config(),
    ),
    BaselinePreset(
        "learned_mediator_same_task",
        "learned_mediator",
        skill_updates_config("mediator"),
    ),
    BaselinePreset(
        "planner_only_skill_evolution",
        "learned_mediator",
        skill_updates_config("planner"),
    ),
    BaselinePreset(
        "mediator_only_protocol_evolution",
        "learned_mediator",
        skill_updates_config("mediator"),
    ),
    BaselinePreset(
        "full_coevolution",
        "learned_mediator",
        skill_updates_config("executor", "planner", "mediator"),
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


def parse_skill_updates(raw_value: str) -> SkillUpdateConfig:
    """Parse comma-separated skill-update permissions from CLI input."""
    tokens = [part.strip().lower() for part in raw_value.split(",") if part.strip()]
    if not tokens:
        raise SkillUpdateParseError(
            "expected one of: none, executor, planner, mediator, all"
        )

    unknown = sorted(set(tokens) - SKILL_UPDATE_TOKENS)
    if unknown:
        allowed = ", ".join(sorted(SKILL_UPDATE_TOKENS))
        raise SkillUpdateParseError(
            f"invalid skill update value(s): {', '.join(unknown)}; "
            f"expected comma-separated values from: {allowed}"
        )

    token_set = set(tokens)

    if "none" in token_set and len(token_set) > 1:
        raise SkillUpdateParseError(
            "'none' cannot be combined with other skill update values"
        )
    if "all" in token_set and len(token_set) > 1:
        raise SkillUpdateParseError(
            "'all' cannot be combined with other skill update values"
        )

    if "none" in token_set:
        return skill_updates_config()
    if "all" in token_set:
        return skill_updates_config("executor", "planner", "mediator")
    return skill_updates_config(*token_set)
