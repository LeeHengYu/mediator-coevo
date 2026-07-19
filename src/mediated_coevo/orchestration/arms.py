"""Explicit component composition for each experimental arm."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator


class OrchestrationArm(str, Enum):
    """Supported orchestration treatments and matched ablations."""

    EXECUTION_ONLY = "execution_only"
    GRAPH_ONLY = "graph_only"
    DIFFUSION_ONLY = "diffusion_only"
    FULL_ORCHESTRATION = "full_orchestration"


PolicyComponent = Literal["none", "random_uniform", "diffusion"]


class ArmPlan(BaseModel):
    """Frozen component plan invoked for one suffix task occurrence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    arm: OrchestrationArm
    graph_agent_enabled: bool
    diffusion_agent_enabled: bool
    policy_component: PolicyComponent
    pack_context: bool

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        """Reject compositions that blur the four fixed treatments."""
        if self.arm is OrchestrationArm.EXECUTION_ONLY:
            expected = (False, False, "none", False)
        elif self.arm is OrchestrationArm.GRAPH_ONLY:
            expected = (True, False, "random_uniform", True)
        elif self.arm is OrchestrationArm.DIFFUSION_ONLY:
            expected = (False, True, "diffusion", True)
        else:
            expected = (True, True, "diffusion", True)
        actual = (
            self.graph_agent_enabled,
            self.diffusion_agent_enabled,
            self.policy_component,
            self.pack_context,
        )
        if actual != expected:
            raise ValueError(f"invalid component plan for {self.arm.value}")
        return self


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
