FROM skillflow/harbor-cli-base:ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 git vim curl wget unzip zip tree gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
