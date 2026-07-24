"""Static task-package templates for LifelongAgentBench OS tasks."""

from string import Template

DOCKERFILE_TEMPLATE = Template(
    """FROM $base_image

COPY $init_name /opt/lab/$init_name
WORKDIR /root
RUN $invocation
"""
)

VERIFIER_TEMPLATE = Template(
    """#!/usr/bin/env bash
set -u

SCRIPT_DIR="$$(cd "$$(dirname "$${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$${TASK_ROOT:-/root}"
VERIFIER_DIR="$${VERIFIER_DIR:-/logs/verifier}"
mkdir -p "$${VERIFIER_DIR}"
cd "$${TASK_ROOT}"

set +e
$invocation
evaluation_status=$$?
set -e

if [ "$${evaluation_status}" -eq 0 ]; then
  reward=1
else
  reward=0
fi
printf "%s\\n" "$${reward}" > "$${VERIFIER_DIR}/reward.txt"
exit 0
"""
)

TASK_TOML_TEMPLATE = Template(
    """schema_version = "1.2"
artifacts = []

[task]
name = "lifelong-agent-bench/$task_id"
description = "LifelongAgentBench OS interaction task."
authors = []
keywords = ["lifelong-agent-bench", "os-interaction"]

[metadata]
author_name = "LifelongAgentBench"
author_email = "unknown@example.com"
difficulty = "unknown"
category = "os_interaction"
family = "os_interaction"
benchmark = "lifelong_agent_bench"
sample_index = "$sample_index"
tags = ["lifelong-agent-bench", "os-interaction"]
expected_reward_range = [0.0, 1.0]

[verifier]
timeout_sec = 120.0

[verifier.env]

[agent]
timeout_sec = 600.0

[environment]
docker_image = "harbor-prebuilt:lifelong-agent-bench-os-$sample_index"
build_timeout_sec = 600.0
os = "linux"
cpus = 1
memory_mb = 2048
storage_mb = 4096
gpus = 0
allow_internet = true
mcp_servers = []

[environment.env]

[solution.env]
"""
)
