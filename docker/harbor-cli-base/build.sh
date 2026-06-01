#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image_tag="${1:-skillflow/harbor-cli-base:ubuntu24.04}"

docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE:-ubuntu:24.04}" \
    --build-arg HERMES_AGENT_VERSION="${HERMES_AGENT_VERSION:-latest}" \
    -t "$image_tag" \
    "$script_dir"

printf 'Built %s\n' "$image_tag"
printf 'Downstream Dockerfiles can use: FROM %s\n' "$image_tag"
