"""Modernize legacy Python code using GLM and the GitHub connector."""

import asyncio
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from mistralai.client import Mistral

load_dotenv()

# Step 1 — Initialize the client
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Step 2 — Define the modernization target
REPO = os.environ.get("GITHUB_REPO", "mistralai/cookbook")
BRANCH = os.environ.get("GITHUB_BRANCH", "main")
FILE_PATHS = [
    "mistral/agents/glm_code_modernizer/legacy_app/app.py",
    "mistral/agents/glm_code_modernizer/legacy_app/utils.py",
]


def extract_python_blocks(text: str) -> list[str]:
    """Extract Python code blocks from the model response."""
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


def get_response_text(response) -> str:
    """Extract text content from a conversation response."""
    parts = []
    for output in response.outputs:
        if output.type == "message.output":
            content = output.content
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.append(
                    "".join(
                        chunk.text if hasattr(chunk, "text") else str(chunk)
                        for chunk in content
                    )
                )
    return "\n".join(parts)


async def main() -> None:
    # Step 3 — Authenticate the GitHub connector
    auth_result = await client.beta.connectors.get_auth_url_async(
        connector_id_or_name="github_app",
    )
    print(f"Authenticate the GitHub connector:\n{auth_result.auth_url}")
    input("Press Enter once you've completed the OAuth flow in your browser...")

    agent_id: str | None = None
    try:
        # Step 4 — Create the agent with the GitHub connector
        file_list = "\n".join(f"- `{path}`" for path in FILE_PATHS)
        agent = await client.beta.agents.create_async(
            name="code_modernizer",
            description="Reads legacy Python files from GitHub and produces modernized versions",
            model="zai-glm-5-2",
            instructions=(
                "You are an expert Python developer who modernizes legacy code.\n\n"
                "IMPORTANT: You MUST use the GitHub connector to read each file's "
                "actual contents from the repository. Do NOT guess or invent code. "
                "Read the real source code first, then modernize it.\n\n"
                "Your workflow:\n"
                "1. Use the GitHub connector to read each file from the repository\n"
                "2. Analyze the actual code you retrieved for legacy patterns\n"
                "3. Produce a modernized version of THAT SAME CODE — same logic, "
                "same structure, same functionality, but with modern Python idioms\n\n"
                "Apply these modernizations to the code you read:\n"
                "- Replace %-formatting with f-strings\n"
                "- Replace os.path with pathlib.Path\n"
                "- Replace manual open()/close() with 'with' statements\n"
                "- Replace json.loads(f.read()) with json.load(f)\n"
                "- Replace bare except: with specific exceptions\n"
                "- Replace type() checks with isinstance()\n"
                "- Replace mutable default arguments with None sentinels\n"
                "- Replace sys.argv parsing with argparse\n"
                "- Add type hints to all functions\n"
                "- Replace == True/False comparisons with truthiness checks\n"
                "- Replace range(len(...)) with enumerate or direct iteration\n\n"
                "Return each modernized file in a separate ```python code fence. "
                "Include a comment with the original filename at the top of each block. "
                "The modernized code must preserve the original functionality exactly."
            ),
            tools=[
                {
                    "type": "connector",
                    "connector_id": "github_app",
                },
            ],
        )
        agent_id = agent.id
        print(f"Created agent: {agent.name} ({agent.id})")

        # Step 5 — Start the conversation
        print(f"Reading files from {REPO} and modernizing...")
        print("This may take a few minutes.\n")

        response = await client.beta.conversations.start_async(
            agent_id=agent.id,
            inputs=[
                {
                    "role": "user",
                    "content": (
                        f"Read these files from the `{REPO}` repository "
                        f"(branch: `{BRANCH}`) using the GitHub connector, "
                        f"then produce modernized versions of each:\n\n{file_list}\n\n"
                        "Return each modernized file in a ```python code fence "
                        "with a comment at the top indicating the original filename."
                    ),
                }
            ],
            timeout_ms=600_000,
        )

        text = get_response_text(response)

        # Step 6 — Save the modernized code
        code_blocks = extract_python_blocks(text)

        if not code_blocks:
            print("No Python code blocks found in the response.")
            print("\nRaw response:\n")
            print(text)
            return

        output_dir = Path("modernized")
        output_dir.mkdir(exist_ok=True)

        filenames = [Path(p).name for p in FILE_PATHS]
        for i, block in enumerate(code_blocks):
            name = filenames[i] if i < len(filenames) else f"file_{i}.py"
            output_path = output_dir / name
            output_path.write_text(block.strip() + "\n", encoding="utf-8")
            print(f"Saved: {output_path}")

        print(f"\nModernized {len(code_blocks)} file(s) in {output_dir}/")

    finally:
        # Cleanup — Delete the agent
        if agent_id:
            await client.beta.agents.delete_async(agent_id=agent_id)
            print(f"Deleted agent: {agent_id}")


if __name__ == "__main__":
    asyncio.run(main())
