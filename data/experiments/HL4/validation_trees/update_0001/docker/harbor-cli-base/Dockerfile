# Build locally with ./build.sh
# Downstream images can simply use: FROM skillflow/harbor-cli-base:ubuntu24.04
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG HERMES_AGENT_VERSION=latest

ENV HOME=/root \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HERMES_HOME=/tmp/hermes \
    HERMES_AGENT_VERSION=${HERMES_AGENT_VERSION}

COPY install-harbor-clis.sh /tmp/install-harbor-clis.sh

RUN chmod +x /tmp/install-harbor-clis.sh \
    && /tmp/install-harbor-clis.sh \
    && rm -f /tmp/install-harbor-clis.sh

WORKDIR /workspace
