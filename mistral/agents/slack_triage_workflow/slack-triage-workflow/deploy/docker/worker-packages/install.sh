#!/bin/sh
# Installs the Debian packages capabilities need in the worker image. A capability whose worker code
# needs a system tool lists it in deploy/docker/worker-packages/<capability>.txt (one package per
# line, `#` comments); Dockerfile.worker runs this with the directory bind-mounted, so the image
# installs exactly what the selected capabilities ask for and nothing when none does.
set -eu

dir=$(dirname "$0")
packages=$(
  for list in "$dir"/*.txt; do
    if [ -e "$list" ]; then sed 's/#.*//; s/\r$//' "$list"; fi
  done | xargs
)
[ -n "$packages" ] || exit 0

apt-get update
# shellcheck disable=SC2086 # one word per package
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $packages
# No `rm -rf /var/lib/apt/lists/*`: Dockerfile.worker mounts that directory as a build cache, so
# nothing in it lands in the image layer, and wiping it would only throw the cache away.
