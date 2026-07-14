"""Explicit component composition for each experimental arm."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator


class OrchestrationArm(str, Enum):
    """Supported orchestration treatments and matched ablations."""

    EXECUTION_ONLY = "execution_only"
    RANDOM_POLICY = "random_policy"
    NO_GRAPH = "no_graph"
    FULL_ORCHESTRATION = "full_orchestration"


PolicyComponent = Literal["none", "random_uniform", "diffusion"]


class ArmPlan(BaseModel):
    """Frozen component plan invoked for one suffix task occurrence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    arm: OrchestrationArm
    use_graph_agent: bool
    policy_component: PolicyComponent
    pack_context: bool

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        """Reject compositions that blur the four fixed treatments."""
        if self.arm is OrchestrationArm.EXECUTION_ONLY:
            expected = (False, "none", False)
        elif self.arm is OrchestrationArm.RANDOM_POLICY:
            expected = (False, "random_uniform", True)
        elif self.arm is OrchestrationArm.NO_GRAPH:
            expected = (False, "diffusion", True)
        else:
            expected = (True, "diffusion", True)
        actual = (self.use_graph_agent, self.policy_component, self.pack_context)
        if actual != expected:
            raise ValueError(f"invalid component plan for {self.arm.value}")
        return self


_ARM_PLANS = {
    OrchestrationArm.EXECUTION_ONLY: ArmPlan(
        arm=OrchestrationArm.EXECUTION_ONLY,
        use_graph_agent=False,
        policy_component="none",
        pack_context=False,
    ),
    OrchestrationArm.RANDOM_POLICY: ArmPlan(
        arm=OrchestrationArm.RANDOM_POLICY,
        use_graph_agent=False,
        policy_component="random_uniform",
        pack_context=True,
    ),
    OrchestrationArm.NO_GRAPH: ArmPlan(
        arm=OrchestrationArm.NO_GRAPH,
        use_graph_agent=False,
        policy_component="diffusion",
        pack_context=True,
    ),
    OrchestrationArm.FULL_ORCHESTRATION: ArmPlan(
        arm=OrchestrationArm.FULL_ORCHESTRATION,
        use_graph_agent=True,
        policy_component="diffusion",
        pack_context=True,
    ),
}


def plan_for_arm(arm: OrchestrationArm) -> ArmPlan:
    """Return the immutable component plan for an arm."""
    return _ARM_PLANS[arm]
