#!/usr/bin/env bash
# Run `uv` with this app's reproducible index selection. When sourced, this wrapper prepares the
# same environment for all package-manager commands in the workspace.
#
# Why the pinning is here and not left to the environment: uv's user config and the
# UV_DEFAULT_INDEX / UV_INDEX_URL environment outrank pyproject's `[[tool.uv.index]]`, so without
# forcing the public default here a stray env would rewrite the lock's provenance. UV_EXTRA_INDEX_URL
# is blanked, not set: a general extra index re-opens the dependency-confusion surface an explicit
# source binding closes. AGENT is dropped because the workflows SDK reads that unprefixed name at
# import time, so a shell that exports it breaks every `uv run` that imports the SDK.
#
# Add or remove a dependency through this wrapper too: `bash tools/uv.sh add --package <member> <dep>`.
#
# Executed (`bash tools/uv.sh <args>`) -> strict mode, then exec uv. Sourced (by tools/install.sh, or
# by hand from bash or zsh) -> only export the environment and leave the caller's shell options alone:
# `set -e` in an interactive shell would close it on the next failing command. zsh has no BASH_SOURCE,
# so a sourced run there reads as "not executed" too.
_uv_sh_executed=0
if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
  _uv_sh_executed=1
  set -euo pipefail
fi

PUBLIC_INDEX="https://pypi.org/simple"
export UV_DEFAULT_INDEX="$PUBLIC_INDEX"
export UV_INDEX_URL="$PUBLIC_INDEX"
export PIP_INDEX_URL="$PUBLIC_INDEX"
export UV_EXTRA_INDEX_URL=""
unset AGENT

# No package-index authentication is required for this audience.

# Executed directly -> run uv with this environment. Sourced -> only export the environment.
if [ "$_uv_sh_executed" = 1 ]; then
  unset _uv_sh_executed
  case "${1:-}" in
    add | remove)
      # `uv add` / `uv remove` persist every index named on the command line or in the environment
      # into the pyproject they edit. With UV_DEFAULT_INDEX exported they wrote a nameless
      # `[[tool.uv.index]] ... default = true` into the member, which tests/test_lock_provenance.py
      # rejects. The root pyproject already pins `pypi` as the default index, so drop the override.
      unset UV_DEFAULT_INDEX UV_INDEX_URL
      ;;
  esac
  exec uv "$@"
fi
unset _uv_sh_executed
