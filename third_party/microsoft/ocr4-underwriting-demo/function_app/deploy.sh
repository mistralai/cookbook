#!/usr/bin/env bash
# Publish the pipeline code to the deployed Function App.
#
# The shared pipeline lives in ../src; Azure Functions only packages this folder,
# so we vendor those modules next to function_app.py for the publish, then remove
# them. (function_app.py imports them by module name, which resolves either way.)
set -euo pipefail
cd "$(dirname "$0")"

APP="${1:-mistral-ocr-foundry-mistralai-func}"
RG="${2:-rg-mistral-ocr-example}"
MODULES=(config observability ocr_client agents workflow pipeline)

for m in "${MODULES[@]}"; do cp "../src/$m.py" "$m.py"; done
cleanup() { for m in "${MODULES[@]}"; do rm -f "$m.py"; done; }
trap cleanup EXIT

if command -v func >/dev/null 2>&1; then
  func azure functionapp publish "$APP" --python
else
  echo "Azure Functions Core Tools (func) not found."
  echo "  Install: brew tap azure/functions && brew install azure-functions-core-tools@4"
  echo "  Fallback (zip deploy, remote build):"
  echo "    mkdir -p ../.build && zip -r ../.build/app.zip . -x '*.pyc' '__pycache__/*'"
  echo "    az functionapp deploy -g $RG -n $APP --src-path ../.build/app.zip --type zip --build-remote true"
  exit 1
fi
