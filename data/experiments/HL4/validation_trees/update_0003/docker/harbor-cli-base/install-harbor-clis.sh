#!/usr/bin/env bash
set -euo pipefail

: "${HERMES_AGENT_VERSION:=latest}"

export HOME=/root
export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
export PATH="/root/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export HERMES_HOME="${HERMES_HOME:-/tmp/hermes}"

install_system_packages() {
    apt-get update
    apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
        xz-utils
    rm -rf /var/lib/apt/lists/*
}

install_hermes_agent() {
    local branch_args=()

    if [ -n "$HERMES_AGENT_VERSION" ] && [ "$HERMES_AGENT_VERSION" != "latest" ]; then
        branch_args=(--branch "$HERMES_AGENT_VERSION")
    fi

    curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh \
        | bash -s -- --skip-setup "${branch_args[@]}"

    mkdir -p "$HERMES_HOME" "$HERMES_HOME/sessions" "$HERMES_HOME/skills" "$HERMES_HOME/memories"
}

cleanup_caches() {
    rm -rf "$HOME/.cache/pip"
}

verify_installations() {
    hermes version
}

install_system_packages
install_hermes_agent
cleanup_caches
verify_installations
