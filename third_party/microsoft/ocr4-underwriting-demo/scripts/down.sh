#!/usr/bin/env bash
# One command: tear down the deployed stack and stop the local app.
#
#   scripts/down.sh
#
# Deletes the azd-provisioned stack and purges the soft-deleted Foundry account (so its name
# and quota are freed immediately), stops the local Gradio app, and removes the local .env.
#
# For a manually deployed stack (deployed with `az`, not `azd`), pass its resource group:
#   scripts/down.sh --rg <name>
#
# Teardown is deliberately resilient: if one step fails (for example, the account was already
# purged), it warns and keeps going rather than aborting half-done. It reports at the end
# whether anything still needs attention.
set -uo pipefail   # NOTE: no -e; we handle failures per-step so cleanup always runs to the end
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"

warn() { printf 'WARNING: %s\n' "$*" >&2; }
say()  { printf '==> %s\n' "$*"; }

cat_say() {
  local msg="$1" c="" r=""
  if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then c=$'\033[36m'; r=$'\033[0m'; fi
  printf '%s\n' "${c} /\\_/\\   ${msg}${r}"
  printf '%s\n' "${c}( -.- )${r}"
  printf '%s\n' "${c} > ^ <${r}"
}

# wait_account_purged <account-name>: the Cognitive Services account purge is asynchronous.
# Poll the soft-deleted list until the name is gone (or we give up). This confirms only the
# Cognitive Services account itself. The account's backing Azure ML workspace shadow is a
# SEPARATE soft-delete that never appears in this list, so an empty list here does not mean a
# redeploy will succeed; purge_foundry_workspace below handles the shadow. Best-effort: never
# fails teardown, just reports.
wait_account_purged() {
  local name="$1" i
  [ -n "$name" ] || return 0
  for i in $(seq 1 20); do   # ~60s at 3s intervals
    if [ -z "$(az cognitiveservices account list-deleted --query "[?name=='$name'].name" -o tsv 2>/dev/null)" ]; then
      say "Cognitive Services account '$name' purge confirmed."
      return 0
    fi
    sleep 3
  done
  warn "Account '$name' is still soft-deleted after ~60s (purge is async). A redeploy into the"
  warn "same resource group may need another minute, or use a fresh resource group name."
}

# purge_foundry_workspace <account-name> <location>: an AI Foundry account (Cognitive Services,
# kind=AIServices) has a backing Azure ML workspace shadow of the same name, managed by the AML
# resource provider. Purging the Cognitive Services account does NOT always take the shadow with
# it, and the shadow soft-deletes on its own, so a same-name redeploy into the same resource
# group fails with "Soft-deleted workspace exists" (Kind: AmlRp). That shadow never shows in the
# Cognitive Services soft-deleted list, which is why purging the account alone is not enough.
# Fire an explicit purge against the AML deleted-workspace endpoint. Some subscriptions do not
# serve that endpoint; there this is a harmless no-op and the caller reports the residual risk.
# Returns 0 if a purge call succeeded, 1 otherwise. Never fails teardown.
purge_foundry_workspace() {
  local name="$1" loc="$2" sub api
  [ -n "$name" ] && [ -n "$loc" ] || return 1
  sub="$(az account show --query id -o tsv 2>/dev/null)" || return 1
  for api in 2024-10-01 2024-04-01 2023-10-01 2023-06-01-preview; do
    if az rest --method delete \
        --url "https://management.azure.com/subscriptions/$sub/providers/Microsoft.MachineLearningServices/locations/$loc/deletedworkspaces/$name?api-version=$api" \
        >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

RG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rg) RG="${2:-}"; [ -n "$RG" ] || { warn "--rg needs a resource group name."; exit 2; }; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) warn "unknown argument: $1"; exit 2 ;;
  esac
done

# Preflight: az is always needed; azd only for the default (azd) path.
command -v az >/dev/null 2>&1 || { warn "'az' (Azure CLI) is not installed; cannot tear down."; exit 1; }
if ! az account show >/dev/null 2>&1; then
  say "Signing in to Azure (az login)"
  az login >/dev/null || { warn "az login failed; cannot tear down. Re-run after signing in."; exit 1; }
fi

problems=0

cat_say "tearing it all down..."

say "Stopping the local app if running"
# Two processes now: the UI (app:ui) and the backend API (api:api). Match each pattern so both
# the `uv run` parent and the python child are stopped.
pkill -f "uvicorn app:ui" 2>/dev/null || true
pkill -f "uvicorn api:api" 2>/dev/null || true

if [ -n "$RG" ]; then
  if ! timeout 30 az group show -n "$RG" >/dev/null 2>&1; then
    say "Resource group '$RG' not found (already deleted?). Nothing to delete."
  else
    say "Deleting resource group $RG and purging its Foundry account"
    acct="$(timeout 30 az cognitiveservices account list -g "$RG" --query '[0].name' -o tsv 2>/dev/null || true)"
    loc="$(timeout 30 az group show -n "$RG" --query location -o tsv 2>/dev/null || echo westus)"
    if ! az group delete -n "$RG" --yes; then
      warn "Failed to delete resource group '$RG'. Delete it in the Azure portal."
      problems=1
    fi
    if [ -n "$acct" ]; then
      az cognitiveservices account purge --name "$acct" --location "$loc" --resource-group "$RG" 2>/dev/null \
        || say "(account purge skipped or already purged)"
      # Two async cleanups follow the delete: the Cognitive Services account name leaving the
      # soft-deleted list, and the account's backing Azure ML workspace shadow (AML RP, same
      # name) being purged. Only the first shows in the soft-deleted list; the workspace shadow
      # is what actually blocks a same-name redeploy, so purge it explicitly.
      wait_account_purged "$acct"
      if purge_foundry_workspace "$acct" "$loc"; then
        say "Backing Foundry workspace shadow purged; '$acct' is free for an immediate redeploy."
      else
        warn "Could not purge the backing Foundry workspace shadow '$acct' via API (that endpoint is"
        warn "not served on this subscription). Its purge is asynchronous, so an immediate same-RG"
        warn "redeploy may still fail with 'Soft-deleted workspace exists'. Wait a few minutes and"
        warn "retry, or deploy under a different resource group / account name."
      fi
    fi
  fi
else
  command -v azd >/dev/null 2>&1 || { warn "'azd' is not installed; if you deployed manually, use: scripts/down.sh --rg <name>"; exit 1; }
  say "azd down (delete + purge)"
  # Capture the account name before azd deletes the group, so we can confirm its (async) purge
  # propagated afterward. The azd path deploys into rg-mistral-demo (see scripts/up.sh).
  azd_acct="$(timeout 30 az cognitiveservices account list -g "rg-mistral-demo" --query '[0].name' -o tsv 2>/dev/null || true)"
  azd_loc="$(timeout 30 az group show -n "rg-mistral-demo" --query location -o tsv 2>/dev/null || echo westus)"
  if ! azd down --purge --force; then
    warn "azd down did not complete cleanly. Check the Azure portal for a leftover resource group, or retry: scripts/down.sh"
    problems=1
  fi
  # azd down --purge issues the Cognitive Services account purge, but two things are async: the
  # account name leaving the soft-deleted list, and the account's backing Azure ML workspace
  # shadow (AML RP, same name) being purged. wait_account_purged polls the first; the workspace
  # shadow never shows in the Cognitive Services soft-deleted list and is what actually blocks a
  # same-name redeploy ("Soft-deleted workspace exists"), so purge it explicitly.
  if [ -n "$azd_acct" ]; then
    wait_account_purged "$azd_acct"
    if purge_foundry_workspace "$azd_acct" "$azd_loc"; then
      say "Backing Foundry workspace shadow purged; '$azd_acct' is free for an immediate redeploy."
    else
      warn "Could not purge the backing Foundry workspace shadow '$azd_acct' via API (that endpoint is"
      warn "not served on this subscription). Its purge is asynchronous, so an immediate same-RG"
      warn "redeploy may still fail with 'Soft-deleted workspace exists'. Wait a few minutes and retry"
      warn "scripts/up.sh, or deploy under a different resource group / account name."
    fi
  fi
fi

rm -f .env && say "Removed local .env (it pointed at the deleted stack)" || true

if [ "$problems" -eq 0 ]; then
  cat_say "all clean, nothing left billing."
else
  cat_say "done, but check the warnings above."
  exit 1
fi
