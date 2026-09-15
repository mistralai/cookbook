#!/usr/bin/env bash
# One command, no arguments: get the demo running.
#
#   scripts/up.sh
#
# It handles everything: checks prerequisites, signs you in if needed, finds or deploys a
# cloud stack, writes the local .env from that stack, starts the backend API and the UI as two
# local processes, and opens your browser. The apps run on your machine; the models, hosted
# agents, and portal traces are in the cloud. The UI (port 8000) talks to the backend API
# (port 8001); neither is exposed publicly. Stop both with Ctrl-C. Tear down with:
# scripts/down.sh
#
# Overrides (rarely needed):
#   scripts/up.sh --rg <name>     use this existing resource group as the backend
#   scripts/up.sh --deploy        force a fresh `azd up` even if a stack exists
#   scripts/up.sh --new-name      deploy under a fresh Foundry account name (implies --deploy);
#                                 use when a soft-deleted, unpurgeable name blocks a same-RG redeploy
#   scripts/up.sh --no-open       don't open the browser
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"

# ---- output helpers ----
_tty() { [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; }

# Step progress: every stage announces itself as "[n/N] ...", so the user always knows which
# step is running and how many remain, even during the long deploy.
STEP=0
TOTAL=5
HANDLED=0   # set to 1 by fail(), so the EXIT trap knows the error was already explained
step() { STEP=$((STEP + 1)); printf '\n[%s/%s] %s\n' "$STEP" "$TOTAL" "$*"; }
say()  { printf '    %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

# fail prints a clear reason + a hint, then exits. Every hard stop goes through here so the
# user learns which step failed and what to do, never a bare stack trace.
fail() {
  HANDLED=1
  printf '\nERROR: %s\n' "$1" >&2
  [ -n "${2:-}" ] && printf '  %s\n' "$2" >&2
  exit 1
}

cat_say() {
  local msg="$1" c="" r=""
  if _tty; then c=$'\033[36m'; r=$'\033[0m'; fi
  printf '%s\n' "${c} /\\_/\\   ${msg}${r}"
  printf '%s\n' "${c}( o.o )${r}"
  printf '%s\n' "${c} > ^ <${r}"
}

# Only speak up for UNEXPECTED aborts. If fail() already explained it (HANDLED=1), stay quiet
# so the user sees one clear message, not two.
trap 'code=$?; if [ $code -ne 0 ] && [ "$HANDLED" -eq 0 ]; then printf "\nup.sh stopped unexpectedly (exit %s). Nothing is left running; safe to re-run. Clean up any partial stack with scripts/down.sh.\n" "$code" >&2; fi' EXIT

# run_or_fail <timeout-seconds> <human-name> <cmd...>: run a command with a timeout so a
# network or interactive stall can never hang the script silently (the 15-min bug). A timeout
# exits 124; we translate that into a clear message instead of a mystery freeze.
run_or_fail() {
  local secs="$1" name="$2"; shift 2
  local rc=0
  timeout "$secs" "$@" || rc=$?
  if [ "$rc" -eq 124 ]; then
    fail "'$name' did not finish within ${secs}s (it may be waiting on input or the network)." \
         "Run it manually to see the prompt: $*"
  elif [ "$rc" -ne 0 ]; then
    fail "'$name' failed (exit $rc)." "Command: $*"
  fi
}

# warn_if_soft_deleted: a soft-deleted Foundry account or its backing Azure ML workspace shadow
# from a previous teardown still holds the deterministic name azd recreates (aif<hash-of-rg>), and
# Cognitive account names are globally unique, so a fresh same-name deploy fails partway through
# with "Soft-deleted workspace exists". Detect it up front and stop with a clear instruction rather
# than waiting minutes for azd up to fail. Best-effort: a check that cannot run (permission, API)
# returns quietly and lets the deploy proceed.
warn_if_soft_deleted() {
  local sub accts wss
  sub="$(az account show --query id -o tsv 2>/dev/null)" || return 0
  accts="$(az cognitiveservices account list-deleted --query "[?starts_with(name,'aif')].name" -o tsv 2>/dev/null || true)"
  wss="$(az rest --method get \
           --url "https://management.azure.com/subscriptions/${sub}/providers/Microsoft.MachineLearningServices/deletedWorkspaces?api-version=2024-10-01" \
           --query "value[?starts_with(name,'aif')].name" -o tsv 2>/dev/null || true)"
  if [ -n "$accts" ] || [ -n "$wss" ]; then
    fail "A soft-deleted AI account or workspace from a previous deploy is still present, so a same-name redeploy will fail (\"Soft-deleted workspace exists\")." \
         "Deploy under a fresh name: scripts/up.sh --new-name   (or wait a few minutes for the purge to finish, then re-run scripts/up.sh)."
  fi
}

# ---- args ----
RG=""
FORCE_DEPLOY=0
NEW_NAME=0
OPEN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --rg) RG="${2:-}"; [ -n "$RG" ] || fail "--rg needs a resource group name."; shift 2 ;;
    --deploy) FORCE_DEPLOY=1; shift ;;
    --new-name) NEW_NAME=1; FORCE_DEPLOY=1; shift ;;
    --no-open) OPEN=0; shift ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) fail "unknown argument: $1" "See: scripts/up.sh --help" ;;
  esac
done

cat_say "starting up..."

# ---- Step 1: Preflight ----
step "Checking prerequisites"
for tool in az uv curl; do
  command -v "$tool" >/dev/null 2>&1 || fail "'$tool' is not installed or not on PATH." \
    "Install it and re-run. (az: Azure CLI, uv: https://docs.astral.sh/uv, curl: usually preinstalled)"
done
command -v timeout >/dev/null 2>&1 || fail "'timeout' is not available (part of coreutils)." \
  "On macOS: brew install coreutils, or use --rg <existing-rg> to skip deploy."
say "az, uv, curl, timeout present."

# ---- Step 2: Auth ----
step "Signing in to Azure (if needed)"
if az account show >/dev/null 2>&1; then
  say "Already signed in."
else
  say "Opening Azure sign-in..."
  az login >/dev/null || fail "az login failed or was cancelled." "Re-run and complete the browser sign-in."
fi

# ---- Step 3: Backend (reuse or deploy), then write .env ----
step "Preparing the cloud backend"
if [ -n "$RG" ]; then
  run_or_fail 30 "resource group check" az group show -n "$RG"
  say "Using existing resource group: $RG"
  uv run python scripts/sync_env.py --rg "$RG" --force \
    || fail "Could not read stack parameters from '$RG'." "Check the group exists and you have access."

else
  export AZURE_ENV_NAME="mistral-demo"
  DEMO_RG="rg-mistral-demo"

  # Does a usable demo stack already exist? Probe with az (fast, never interactive), guarded
  # by a timeout. Never use `azd env get-values` as a probe: it can prompt and hang.
  reuse=0
  if [ "$FORCE_DEPLOY" -eq 0 ] && timeout 30 az group show -n "$DEMO_RG" >/dev/null 2>&1; then
    acct="$(timeout 30 az cognitiveservices account list -g "$DEMO_RG" --query '[0].name' -o tsv 2>/dev/null || true)"
    [ -n "$acct" ] && reuse=1
  fi

  if [ "$reuse" -eq 1 ]; then
    say "Reusing the deployed demo stack ($DEMO_RG)"
    uv run python scripts/sync_env.py --rg "$DEMO_RG" --force \
      || fail "Could not read parameters from the existing demo stack." "Try: scripts/up.sh --deploy"
  else
    command -v azd >/dev/null 2>&1 || fail "'azd' (Azure Developer CLI) is not installed." \
      "Install from https://aka.ms/azd, or point at an existing stack: scripts/up.sh --rg <name>"
    say "Deploying the demo stack with azd (a few minutes)"
    if ! timeout 15 azd auth login --check-status >/dev/null 2>&1; then
      say "Signing in to azd..."
      azd auth login || fail "azd sign-in failed or was cancelled." "Re-run and complete the sign-in."
    fi
    # Non-interactive env setup (a prompt here is what hung the script before).
    azd env select "$AZURE_ENV_NAME" >/dev/null 2>&1 \
      || azd env new "$AZURE_ENV_NAME" --no-prompt >/dev/null \
      || fail "Could not create the azd environment '$AZURE_ENV_NAME'."
    run_or_fail 15 "azd env set (location)" azd env set AZURE_LOCATION westus
    run_or_fail 15 "azd env set (resource group)" azd env set AZURE_RESOURCE_GROUP "$DEMO_RG"
    if [ "$NEW_NAME" -eq 1 ]; then
      # Rotate onto a fresh Foundry account name so a soft-deleted, unpurgeable workspace shadow
      # from a prior teardown cannot block this deploy. The salt feeds uniqueString in main.bicep;
      # it persists in the azd env, so later idempotent redeploys reuse this same rotated name.
      run_or_fail 15 "azd env set (name suffix)" azd env set AZURE_FOUNDRY_NAME_SUFFIX "$(date +%Y%m%d%H%M%S)"
      say "Deploying under a fresh Foundry account name (AZURE_FOUNDRY_NAME_SUFFIX rotated)."
    fi
    run_or_fail 60 "resource group create" az group create -n "$DEMO_RG" -l westus
    # Preflight: unless we are already rotating the name (--new-name), stop early if a prior
    # teardown left a soft-deleted account/workspace that would block a same-name redeploy.
    [ "$NEW_NAME" -eq 1 ] || warn_if_soft_deleted
    say "Provisioning + deploying (this is the slow part; watch for the (done) lines)"
    azd up --no-prompt \
      || fail "azd up failed to provision or deploy." \
              "Common causes: quota/region limits; the deployer lacks Owner/User Access Admin on the RG (needed for the role assignment); or a soft-deleted same-name account/workspace blocks the create (re-run: scripts/up.sh --new-name). Or clean up with scripts/down.sh."
    uv run python scripts/sync_env.py --rg "$DEMO_RG" --force \
      || fail "Deploy succeeded but reading its parameters failed." "Re-run: scripts/up.sh (it will reuse the stack)."
  fi
fi

# Sanity: .env must exist now, or the app will error on every request.
[ -f .env ] || fail ".env was not written; the app has no backend to talk to." "Re-run scripts/up.sh."
say "Backend ready; .env written."

# ---- Step 4: Pre-launch checks ----
# Two processes now: the backend API (api:api on 8001) and the UI (app:ui on 8000). Both ports
# must be free.
step "Checking the local ports are free"
API_URL="http://127.0.0.1:8001"
URL="http://127.0.0.1:8000"
for port in 8001 8000; do
  if lsof -iTCP:"$port" -sTCP:LISTEN -n >/dev/null 2>&1; then
    fail "Port $port is already in use (the demo may already be running)." \
         "Stop it first: scripts/down.sh  (or: pkill -f 'uvicorn api:api'; pkill -f 'uvicorn app:ui')."
  fi
done
say "Ports 8001 (API) and 8000 (UI) are free."

# ---- Step 5: Launch backend, then UI ----
# Start the backend API in the background, wait until it answers /health, then run the UI in
# the foreground pointed at it. If the backend never comes up, stop rather than launch a UI
# with nothing to talk to.
step "Starting the backend API ($API_URL) and the UI ($URL)"

# `python -m uvicorn` so it uses the uv environment's interpreter (the bare `uvicorn` console
# script can resolve to a different Python).
( cd src && exec uv run python -m uvicorn api:api --host 127.0.0.1 --port 8001 ) &
API_PID=$!

# If we exit for any reason from here on, take the backend down with us (no orphan process).
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

say "Waiting for the backend to answer /health..."
ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    fail "The backend API exited before it became ready." "Run it manually to see why: cd src && uv run python -m uvicorn api:api --port 8001"
  fi
  if curl -fsS -o /dev/null "$API_URL/health" 2>/dev/null; then ready=1; break; fi
  sleep 1
done
[ "$ready" -eq 1 ] || fail "The backend API did not become ready within 60s." "Check for errors above, then re-run."
say "Backend is up."

cat_say "backend ready, starting the app!"
printf '    /apply  /review  /chat        (Ctrl-C to stop both)\n'
say "The browser opens automatically once the UI responds..."

if [ "$OPEN" -eq 1 ]; then
  opener="$(command -v open || command -v xdg-open || true)"
  if [ -n "$opener" ]; then
    (
      for _ in $(seq 1 40); do
        if curl -fsS -o /dev/null "$URL/" 2>/dev/null; then
          "$opener" "$URL/" >/dev/null 2>&1 || true
          break
        fi
        sleep 1
      done
    ) &
  fi
fi

# The UI runs in the foreground; the client reaches the backend at BACKEND_URL. Ctrl-C stops
# the UI, and the EXIT trap above stops the backend too.
cd src && BACKEND_URL="$API_URL" exec uv run python -m uvicorn app:ui --host 127.0.0.1 --port 8000
