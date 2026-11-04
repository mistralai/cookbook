"""Routes a Slack message to the workflow that owns its thread, starting that workflow if needed."""

import hashlib
from datetime import timedelta

import mistralai.workflows as workflows
from mistralai.client.errors import SDKError
from mistralai.client.workflows import Workflows
from mistralai.workflows.client import get_mistral_client

from models import SlackRequest, ThreadState

THREAD_WORKFLOW = "investigation-agent-thread"


@workflows.activity(start_to_close_timeout=timedelta(seconds=30), retry_policy_max_attempts=3)
async def route_message(request: SlackRequest) -> str:
    """Hand the message to its thread workflow and return "delivered" or "started"."""
    execution_id = thread_execution_id(request)
    client = get_mistral_client().workflows
    if await deliver(client, execution_id, request):
        return "delivered"
    if await start(client, execution_id, request):
        return "started"
    # Another router started the thread between our two calls, so it can take the update now.
    if await deliver(client, execution_id, request):
        return "delivered"
    raise RuntimeError("Thread workflow neither accepted the message nor started")


def thread_execution_id(request: SlackRequest) -> str:
    """Build a stable execution ID so every message of a Slack thread lands in the same workflow."""
    digest = hashlib.sha256(f"{request.team_id}:{request.channel}:{request.thread_ts}".encode()).hexdigest()[:12]
    return f"investigation-thread-{request.thread_ts.replace('.', '-')}-{digest}"


async def deliver(client: Workflows, execution_id: str, request: SlackRequest) -> bool:
    """Send the message update to the running thread workflow. Returns False when no such workflow is running."""
    try:
        await client.executions.update_workflow_execution_async(execution_id=execution_id, name="message", input=request.model_dump())
    except SDKError as error:
        status = error.raw_response.status_code
        # Any 4xx other than rate limiting means the execution is missing or finished; anything else is a real failure.
        if not 400 <= status < 500 or status == 429:
            raise
        return False
    return True


async def start(client: Workflows, execution_id: str, request: SlackRequest) -> bool:
    """Start a thread workflow seeded with this message. Returns False when the execution already exists (409)."""
    state = ThreadState(channel=request.channel, thread_ts=request.thread_ts, pending=[request], known_ts=[request.message_ts])
    try:
        await client.execute_workflow_async(workflow_identifier=THREAD_WORKFLOW, execution_id=execution_id, input=state.model_dump(), wait_for_result=False)
    except SDKError as error:
        if error.raw_response.status_code != 409:
            raise
        return False
    return True
