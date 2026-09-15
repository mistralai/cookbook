"""Write a local .env from a deployed stack's parameters, so running the app or scripts
locally never means hand-copying endpoints and keys.

Source of the values, in order of preference:
  1. `azd env get-values` for the current azd environment (the azd deploy path). azd exposes
     every Bicep output; this reads them directly. Requires the azd CLI and a selected env.
  2. `--rg <name>` to read a manually deployed stack by resource group (the az deploy path):
     the account/storage names are discovered in that group.

Secrets are not Bicep outputs, so the account key and storage connection string are fetched
with `az` using the account/storage names from the stack. Requires `az login`.

Usage:
  uv run python scripts/sync_env.py                 # from the current azd environment
  uv run python scripts/sync_env.py --rg rg-mistral-ocr-example   # from a named stack
  uv run python scripts/sync_env.py --print         # print to stdout instead of writing .env

Writes to .env at the repo root (gitignored). Refuses to overwrite unless --force.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def _from_azd() -> dict:
    """Read the current azd environment's values (Bicep outputs + azd env vars)."""
    raw = _run(["azd", "env", "get-values"])
    vals = {}
    for line in raw.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            vals[k.strip()] = v.strip().strip('"')
    return vals


def _from_rg(rg: str) -> dict:
    """Discover the account + storage names in a resource group (manual deploy path)."""
    acct = _run(["az", "cognitiveservices", "account", "list", "-g", rg,
                 "--query", "[0].name", "-o", "tsv"])
    stg = _run(["az", "storage", "account", "list", "-g", rg,
                "--query", "[0].name", "-o", "tsv"])
    proj = _run(["az", "cognitiveservices", "account", "project", "list",
                 "--name", acct, "-g", rg, "--query", "[0].name", "-o", "tsv"])
    # `project list` returns the name as "<account>/<project>"; the endpoint needs just the
    # project segment.
    proj = proj.split("/")[-1]
    # App Insights is not returned by the discovery calls above, and on this path there are no
    # Bicep outputs to read it from. Find the component by resource type with core `az resource`
    # (no application-insights extension), so _resolve can wire up telemetry export. Missing it
    # is what left APPLICATIONINSIGHTS_CONNECTION_STRING="" and dropped every local OCR span.
    ai_name = _run(["az", "resource", "list", "-g", rg, "--resource-type",
                    "Microsoft.Insights/components", "--query", "[0].name", "-o", "tsv"])
    return {"foundryAccountName": acct, "storageAccountName": stg, "projectName": proj,
            "appInsightsName": ai_name, "AZURE_RESOURCE_GROUP": rg}


def _resolve(vals: dict, rg: str | None) -> dict:
    """Map stack values (however sourced) to the .env vars config.py reads."""
    acct = vals.get("foundryAccountName") or vals.get("accountName")
    stg = vals.get("storageAccountName")
    rg = rg or vals.get("AZURE_RESOURCE_GROUP")
    if not acct or not stg or not rg:
        sys.exit("Could not determine account, storage, and resource group from the stack.")

    host = f"https://{acct}.services.ai.azure.com"
    proj = vals.get("projectName") or _run(
        ["az", "cognitiveservices", "account", "project", "list", "--name", acct,
         "-g", rg, "--query", "[0].name", "-o", "tsv"]).split("/")[-1]

    # Secrets are not Bicep outputs; fetch them now.
    key = _run(["az", "cognitiveservices", "account", "keys", "list", "--name", acct,
                "-g", rg, "--query", "key1", "-o", "tsv"])
    conn = _run(["az", "storage", "account", "show-connection-string", "--name", stg,
                 "-g", rg, "--query", "connectionString", "-o", "tsv"])
    # Prefer the Bicep output (azd path). On the --rg path it is absent, so read the connection
    # string off the component by name with core `az resource show` — no application-insights
    # extension, so the demo stays self-contained on any machine. An empty value here silently
    # kills all local OpenTelemetry export, which is the bug that hid the OCR-4 spans.
    ai_conn = vals.get("appInsightsConnectionString", "")
    ai_name = vals.get("appInsightsName")
    if not ai_conn and ai_name:
        try:
            ai_conn = _run(["az", "resource", "show", "-g", rg, "-n", ai_name,
                            "--resource-type", "Microsoft.Insights/components",
                            "--query", "properties.ConnectionString", "-o", "tsv"])
        except subprocess.CalledProcessError:
            ai_conn = ""

    return {
        "AZURE_AI_KEY": key,
        "AZURE_OCR_ENDPOINT": vals.get("ocrEndpoint") or f"{host}/providers/mistral/azure/ocr",
        "AZURE_OCR_DEPLOYMENT": vals.get("ocrDeploymentName", "mistral-ocr-4"),
        "AZURE_INFERENCE_ENDPOINT": vals.get("foundryEndpoint") or host,
        "AZURE_CHAT_DEPLOYMENT": vals.get("AZURE_CHAT_DEPLOYMENT")
        or vals.get("mediumDeploymentName", "mistral-medium-3-5"),
        "AZURE_AI_PROJECT_ENDPOINT": vals.get("AZURE_AI_PROJECT_ENDPOINT")
        or vals.get("projectEndpoint") or f"{host}/api/projects/{proj}",
        "USE_FOUNDRY_AGENTS": "true",
        # Hosted-only: never silently fall back to a client-side chat, so every agent answer is a
        # server-side Foundry Agent Service run that shows under the agent's portal Traces tab.
        # Each call site degrades deterministically on error, so this cannot crash the flow.
        "FOUNDRY_AGENTS_STRICT": "true",
        "AZURE_STORAGE_CONNECTION_STRING": conn,
        "AZURE_STORAGE_INBOX_CONTAINER": "inbox",
        "AZURE_STORAGE_RESULTS_CONTAINER": "results",
        "APPLICATIONINSIGHTS_CONNECTION_STRING": ai_conn,
        "ENABLE_SENSITIVE_TELEMETRY": "true",
        "OTEL_SERVICE_NAME": "doc-pipeline-local",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a local .env from a deployed stack.")
    ap.add_argument("--rg", help="Read a manually deployed stack by resource group.")
    ap.add_argument("--print", action="store_true", dest="to_stdout",
                    help="Print to stdout instead of writing .env.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing .env.")
    args = ap.parse_args()

    try:
        vals = _from_rg(args.rg) if args.rg else _from_azd()
    except FileNotFoundError as e:
        sys.exit(f"Required CLI not found: {e}. Need `azd` (or `--rg` with `az`).")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Could not read the stack: {e.stderr or e}")

    env = _resolve(vals, args.rg)
    body = "# Generated by scripts/sync_env.py from the deployed stack. Gitignored.\n"
    body += "\n".join(f'{k}="{v}"' for k, v in env.items()) + "\n"

    if args.to_stdout:
        # Mask secrets when printing to a terminal.
        for line in body.splitlines():
            if any(s in line for s in ("KEY", "CONNECTION_STRING")) and '="' in line:
                k = line.split("=", 1)[0]
                print(f'{k}="***"')
            else:
                print(line)
        return 0

    if os.path.exists(ENV_PATH) and not args.force:
        sys.exit(f".env already exists at {ENV_PATH}. Re-run with --force to overwrite.")
    with open(ENV_PATH, "w") as f:
        f.write(body)
    print(f"wrote {ENV_PATH} ({len(env)} vars) from the deployed stack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
