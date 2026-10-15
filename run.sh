#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-simulate}"
if (( $# > 0 )); then
  shift
fi

case "$mode" in
  simulate)
    PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" \
      python3 "$repo_dir/examples/network_net.py" --simulate "$@"
    ;;
  emit)
    PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" \
      python3 "$repo_dir/examples/network_net.py" "$@"
    ;;
  compile)
    if (( $# < 1 )); then
      echo "Usage: $0 compile OUTPUT [network example options]" >&2
      exit 2
    fi
    output="$1"
    shift
    if [[ ! -x "$repo_dir/build/tools/lpn-compile/lpn-compile" ]]; then
      echo "Build lpn-compile first with ./build.sh" >&2
      exit 1
    fi
    model_file="$(mktemp --suffix=.mlir)"
    trap 'rm -f -- "$model_file"' EXIT
    PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" \
      python3 "$repo_dir/examples/network_net.py" "$@" > "$model_file"
    "$repo_dir/build/tools/lpn-compile/lpn-compile" "$model_file" -o "$output"
    ;;
  *)
    echo "Usage: $0 [simulate|emit|compile] [network example options]" >&2
    exit 2
    ;;
esac
