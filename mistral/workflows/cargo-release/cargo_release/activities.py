"""Activities for the cargo release workflow.

All I/O, LLM calls, and OCR live here: never in workflow.py.
Activities are the only place where non-deterministic work is safe.
The workflow body calls these as awaitable activity proxies; the SDK
schedules them on a worker and persists the results in event history.
"""

import base64
import mimetypes
from datetime import timedelta
from pathlib import Path
from urllib.parse import unquote
from urllib.request import url2pathname

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from mistralai.client.models import DocumentURLChunk, OCRRequest

from .models import CargoClassification, ComplianceCheck, DangerousGoodsResult


def _confined_local_path(raw_path: str) -> Path:
    """Resolve a caller-supplied local path, confined to the project directory.

    ``document_uri`` comes from workflow input, which is untrusted. Reading an
    arbitrary path (e.g. ``/etc/passwd`` or ``file:///…``) and shipping its
    bytes to the OCR API would let a caller exfiltrate host files. We therefore
    resolve the path under the project working directory and reject anything
    that escapes it.
    """
    root = Path.cwd()
    candidate = Path(raw_path)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Refusing to read '{candidate}': outside the project directory '{root}'.")
    return candidate


def _resolve_document_uri(document_uri: str) -> str:
    """Convert a local path or ``file://`` URI to a base64 data URI for the OCR API.

    HTTPS URLs and existing data URIs are returned as-is. Local paths and
    ``file://`` URIs are read from disk (confined to the project directory,
    see :func:`_confined_local_path`) and encoded as ``data:<mime>;base64,...``
    so the OCR API can consume them without network access to the filesystem.
    """
    if document_uri.startswith(("https://", "data:")):
        return document_uri

    if document_uri.startswith("file://"):
        raw_path = url2pathname(unquote(document_uri[7:]))
    else:
        raw_path = document_uri

    path = _confined_local_path(raw_path)
    mime = mimetypes.guess_type(str(path))[0] or "text/plain"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


@workflows.activity(
    name="ocr_extract_shipping_doc",
    # Real OCR APIs are flaky (network timeouts, transient 5xx). Three attempts
    # with default exponential back-off gives us resilience without infinite loops.
    # start_to_close_timeout must be a timedelta: passing a bare int causes an
    # infinite retry loop (SDK footgun, documented in activities guide).
    retry_policy_max_attempts=3,
    start_to_close_timeout=timedelta(minutes=2),
)
async def ocr_extract_shipping_doc(document_uri: str) -> str:
    """Return the plain-text content of a shipping document via Mistral OCR."""
    response = await workflows_mistralai.mistralai_ocr(
        params=OCRRequest(
            model="mistral-ocr-latest",
            document=DocumentURLChunk(document_url=_resolve_document_uri(document_uri)),
        )
    )
    return "\n\n".join(page.markdown for page in response.pages)


@workflows.activity(
    name="classify_cargo_type",
    retry_policy_max_attempts=2,
    start_to_close_timeout=timedelta(minutes=1),
)
async def classify_cargo_type(extracted_text: str) -> CargoClassification:
    """Call the LLM with structured output to classify cargo as general,
    dangerous_goods, or perishable.

    chat_parse_to_model wraps mistralai_chat_parse and validates the JSON
    response against the Pydantic schema: no ad-hoc parsing required.
    """
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[
            workflows_mistralai.SystemMessage(
                content=(
                    "You are a maritime cargo classification expert. "
                    "Classify the cargo described in the shipping document extract. "
                    "Respond only with the JSON schema provided."
                )
            ),
            workflows_mistralai.UserMessage(
                content=(
                    f"Shipping document extract:\n\n{extracted_text}\n\n"
                    "Classify the cargo type. Choose one of: "
                    "general, dangerous_goods, perishable."
                )
            ),
        ],
    )
    return await workflows_mistralai.chat_parse_to_model(CargoClassification, request)


@workflows.activity(
    name="validate_dangerous_goods",
    retry_policy_max_attempts=2,
    start_to_close_timeout=timedelta(minutes=1),
)
async def validate_dangerous_goods(extracted_text: str, shipment_id: str) -> DangerousGoodsResult:
    """Run a structured-output LLM call to extract UN number, hazard class, and anomalies.

    This activity is called from DangerousGoodsValidationWorkflow (the sub-workflow),
    which gives it its own retry surface isolated from the parent workflow.
    """
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[
            workflows_mistralai.SystemMessage(
                content=(
                    "You are a dangerous goods compliance auditor. "
                    "Extract UN number and hazard class from the document. "
                    "List any anomalies: missing emergency contact, missing quantity, "
                    "missing proper shipping name, or any other compliance gap. "
                    "Set has_anomaly=true if anomalies list is non-empty."
                )
            ),
            workflows_mistralai.UserMessage(
                content=(f"Shipment ID: {shipment_id}\n\nShipping document extract:\n\n{extracted_text}")
            ),
        ],
    )
    return await workflows_mistralai.chat_parse_to_model(DangerousGoodsResult, request)


@workflows.activity(
    name="validate_customs_compliance",
    retry_policy_max_attempts=2,
    start_to_close_timeout=timedelta(minutes=1),
)
async def validate_customs_compliance(extracted_text: str, cargo_type: str, shipment_id: str) -> ComplianceCheck:
    """Check the document against standard customs rules.

    Validates presence of: HS code, country of origin, consignee details,
    declared value, and cargo weight. Returns a structured pass/fail result.
    """
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[
            workflows_mistralai.SystemMessage(
                content=(
                    "You are a customs compliance officer. Check whether the shipping "
                    "document satisfies all of these rules:\n"
                    "1. HS code present\n"
                    "2. Country of origin declared\n"
                    "3. Consignee name and address present\n"
                    "4. Declared value present\n"
                    "5. Gross weight declared\n"
                    "6. For dangerous_goods cargo: emergency contact present\n"
                    "Set passed=false and list failed_rules if any rule fails."
                )
            ),
            workflows_mistralai.UserMessage(
                content=(
                    f"Shipment ID: {shipment_id}\n"
                    f"Cargo type: {cargo_type}\n\n"
                    f"Shipping document extract:\n\n{extracted_text}"
                )
            ),
        ],
    )
    return await workflows_mistralai.chat_parse_to_model(ComplianceCheck, request)


@workflows.activity(
    name="generate_release_certificate",
    retry_policy_max_attempts=2,
    start_to_close_timeout=timedelta(minutes=1),
)
async def generate_release_certificate(
    shipment_id: str,
    cargo_type: str,
    compliance_notes: str,
) -> str:
    """Generate a formal release certificate as a templated LLM string.

    Returns plain text: the certificate is stored in the workflow result
    and can be emailed / archived downstream (email connector out of scope here).
    """
    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[
            workflows_mistralai.SystemMessage(
                content=(
                    "You are a port authority officer. Generate a concise, formal cargo "
                    "release certificate. Include: shipment ID, cargo type, clearance date "
                    "(today), compliance summary, and an authorisation statement."
                )
            ),
            workflows_mistralai.UserMessage(
                content=(
                    f"Shipment ID: {shipment_id}\n"
                    f"Cargo type: {cargo_type}\n"
                    f"Compliance notes: {compliance_notes}\n\n"
                    "Generate the release certificate."
                )
            ),
        ],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)
    # mistralai_chat_complete returns a ChatCompletionResponse;
    # extract the assistant message text.
    if not response.choices or not response.choices[0].message:
        raise ValueError("Empty response from certificate generation LLM call")
    return response.choices[0].message.content or ""
