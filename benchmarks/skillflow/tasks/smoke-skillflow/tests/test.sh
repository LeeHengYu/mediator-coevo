#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier

if [ -f answer.txt ] && [ "$(tr -d '\r' < answer.txt)" = "ok" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
