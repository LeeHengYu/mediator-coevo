"""Explicit component composition for each experimental arm."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class OrchestrationArm(str, Enum):
    """Supported orchestration treatments and matched ablations."""

    EXECUTION_ONLY = "execution_only"
    GRAPH_ONLY = "graph_only"
    DIFFUSION_ONLY = "diffusion_only"
    FULL_ORCHESTRATION = "full_orchestration"


PolicyComponent = Literal["none", "random_uniform", "diffusion"]


@dataclass(frozen=True, slots=True)
class ArmPlan:
    """Frozen component plan invoked for one suffix task occurrence."""

    arm: OrchestrationArm
    graph_agent_enabled: bool
    diffusion_agent_enabled: bool
    policy_component: PolicyComponent
    pack_context: bool


_ARM_PLANS = {
    OrchestrationArm.EXECUTION_ONLY: ArmPlan(
        arm=OrchestrationArm.EXECUTION_ONLY,
        graph_agent_enabled=False,
        diffusion_agent_enabled=False,
        policy_component="none",
        pack_context=False,
    ),
    OrchestrationArm.GRAPH_ONLY: ArmPlan(
        arm=OrchestrationArm.GRAPH_ONLY,
        graph_agent_enabled=True,
        diffusion_agent_enabled=False,
        policy_component="random_uniform",
        pack_context=True,
    ),
    OrchestrationArm.DIFFUSION_ONLY: ArmPlan(
        arm=OrchestrationArm.DIFFUSION_ONLY,
        graph_agent_enabled=False,
        diffusion_agent_enabled=True,
        policy_component="diffusion",
        pack_context=True,
    ),
    OrchestrationArm.FULL_ORCHESTRATION: ArmPlan(
        arm=OrchestrationArm.FULL_ORCHESTRATION,
        graph_agent_enabled=True,
        diffusion_agent_enabled=True,
        policy_component="diffusion",
        pack_context=True,
    ),
}


def plan_for_arm(arm: OrchestrationArm) -> ArmPlan:
    """Return the immutable component plan for an arm."""
    return _ARM_PLANS[arm]


def arm_for_flags(
    *, graph_agent_enabled: bool, diffusion_agent_enabled: bool
) -> OrchestrationArm:
    """Resolve the stable treatment label for two independent agent flags."""
    return {
        (False, False): OrchestrationArm.EXECUTION_ONLY,
        (True, False): OrchestrationArm.GRAPH_ONLY,
        (False, True): OrchestrationArm.DIFFUSION_ONLY,
        (True, True): OrchestrationArm.FULL_ORCHESTRATION,
    }[(graph_agent_enabled, diffusion_agent_enabled)]
