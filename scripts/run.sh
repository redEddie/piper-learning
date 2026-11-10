#!/usr/bin/env bash
set -euo pipefail
# Avoid system CUDA libraries shadowing the pinned CUDA wheels.
exec env -u LD_LIBRARY_PATH "$@"
