#!/usr/bin/env bash
# Raise the Metal GPU working-set ceiling (LLM-1.2) so large resident weights aren't
# throttled. Size on TOTAL params x quant bits + KV headroom, never active params.
#
# Usage: sudo ./set-wired-limit.sh <megabytes>
#   e.g. a ~26 GB Q6 model + ~12 GB KV/context headroom -> ~40960
#
# This resets on reboot. Re-run after restart, or add a LaunchDaemon to persist.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sudo $0 <wired_limit_mb>" >&2
  exit 1
fi

LIMIT_MB="$1"
if ! [[ "$LIMIT_MB" =~ ^[0-9]+$ ]]; then
  echo "error: argument must be an integer number of megabytes" >&2
  exit 1
fi

echo "Setting iogpu.wired_limit_mb=${LIMIT_MB} (was: $(sysctl -n iogpu.wired_limit_mb))"
sysctl -w "iogpu.wired_limit_mb=${LIMIT_MB}"
echo "Done. Verify with: sysctl iogpu.wired_limit_mb"
