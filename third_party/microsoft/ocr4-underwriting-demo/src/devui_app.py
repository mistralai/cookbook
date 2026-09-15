"""Serve the mortgage underwriting workflow in Agent Framework DevUI (human-in-the-loop).

Run from inside src/:

    cd src && uv run python devui_app.py

Then open the printed URL. Give it a case one of three ways:
  - Upload a document (image/PDF): OCR 4 reads it.
  - Type a sample path, e.g. `inputs/loan_application_1003.pdf`.
  - Type a plain case description (OCR 4 is skipped).

The run pauses for the applicant (missing fields) and for the reviewer (approve/decline);
answer inline in DevUI and it resumes. Auth is your `az login` identity (the Foundry User).
Requires USE_FOUNDRY_AGENTS=true and AZURE_AI_PROJECT_ENDPOINT in .env.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("USE_FOUNDRY_AGENTS", "true")

from agent_framework_devui import serve  # noqa: E402

from observability import setup_observability  # noqa: E402
from underwriting_workflow import build_underwriting_workflow  # noqa: E402


def main() -> None:
    # Export OTel (workflow executor spans + Agent Framework GenAI agent/token spans) to the
    # Application Insights connected to the Foundry project, so the runs and full-workflow
    # traces show in Foundry Operate/Tracing. Needs APPLICATIONINSIGHTS_CONNECTION_STRING (the
    # project-connected appi-mistral-docai) in .env. DevUI's own instrumentation is left off to
    # avoid a second tracer provider fighting the Azure Monitor exporter.
    setup_observability()

    workflow = build_underwriting_workflow()
    try:  # a readable name in the DevUI entity list (falls back to the auto name)
        workflow.name = "mortgage-underwriting"
    except Exception:
        pass
    serve(
        entities=[workflow],
        port=8090,
        host="127.0.0.1",
        auto_open=True,
        auth_enabled=False,           # local demo; no token prompt
        ui_enabled=True,
        instrumentation_enabled=False,  # Azure Monitor (setup_observability) is the exporter
    )


if __name__ == "__main__":
    main()
