# Multiple Authentication Credentials Cookbook

Store and manage multiple sets of credentials for a single connector, then call tools with specific credentials to control access at runtime.

This cookbook covers two authentication patterns:
- **[Bearer token](#part-1-multiple-bearer-token-credentials-github-mcp)** — static tokens (GitHub PATs), stored directly via the API.
- **[OAuth2](#part-2-multiple-oauth2-credentials-outlook-calendar-mcp)** — delegated auth flows (Microsoft accounts), initiated via `get_auth_url`.

> **API status:** Credentials management uses `client.beta.connectors`. This is a **beta** endpoint and may change.

---

## Table of Contents

- [Part 1: Multiple Bearer Token Credentials (GitHub MCP)](#part-1-multiple-bearer-token-credentials-github-mcp)
  - [Prerequisites](#prerequisites-bearer)
  - [When to Use Multiple Bearer Credentials](#when-to-use-multiple-bearer-credentials)
  - [1. Initialize the Client](#1-initialize-the-client)
  - [2. Create a GitHub MCP Connector](#2-create-a-github-mcp-connector)
  - [3. Get Authentication Methods](#3-get-authentication-methods)
  - [4. Store Bearer Credentials](#4-store-bearer-credentials)
  - [5. List Credentials](#5-list-credentials)
  - [6. Call a Tool with Specific Credentials](#6-call-a-tool-with-specific-credentials)
  - [7. Delete Credentials](#7-delete-credentials)
- [Part 2: Multiple OAuth2 Credentials (Outlook Calendar MCP)](#part-2-multiple-oauth2-credentials-outlook-calendar-mcp)
  - [Prerequisites](#prerequisites-oauth2)
  - [When to Use Multiple OAuth2 Credentials](#when-to-use-multiple-oauth2-credentials)
  - [1. Get the Outlook Calendar Connector](#1-get-the-outlook-calendar-connector)
  - [2. Get Authentication Methods](#2-get-authentication-methods-1)
  - [3. Authenticate Accounts via OAuth2](#3-authenticate-accounts-via-oauth2)
  - [4. List Credentials](#4-list-credentials-1)
  - [5. Call a Tool with Specific Credentials](#5-call-a-tool-with-specific-credentials)
  - [6. Promote Credentials to Default](#6-promote-credentials-to-default)
  - [7. Delete Credentials](#7-delete-credentials-1)
- [Naming Conventions](#naming-conventions)
- [Troubleshooting](#troubleshooting)
- [Error Codes Reference](#error-codes-reference)

---

## Part 1: Multiple Bearer Token Credentials (GitHub MCP)

### Prerequisites (Bearer)

```bash
MISTRAL_API_KEY=your-mistral-api-key
GITHUB_PAT_FULL=ghp_yourFullAccessToken
GITHUB_PAT_LIMITED=ghp_yourLimitedOrInvalidToken
```

- `GITHUB_PAT_FULL` — a PAT with `repo` read scope, used to successfully list issues.
- `GITHUB_PAT_LIMITED` — a PAT with no scopes or an invalid value, used to demonstrate a rejected call.

Create GitHub personal access tokens in your [GitHub developer settings](https://github.com/settings/tokens).

**Script:** `python/src/scripts/07_multiple_bearer_authentication.py`

---

### When to Use Multiple Bearer Credentials

- **Test access tiers** — verify that a restricted token cannot reach resources a full token can.
- **Rotate credentials safely** — add the new credentials, promote it to default, then delete the old one with no downtime.
- **Scope tool calls explicitly** — pass `credentials_name` to `call_tool` to choose which identity executes the request.

---

### 1. Initialize the Client

**Python:**

```python
import os
from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
```

**curl:**

```bash
export MISTRAL_API_KEY="your-api-key"
export BASE_URL="https://api.mistral.ai"
```

---

### 2. Create a GitHub MCP Connector

You can skip this step if you use an existing connector with bearer authentication. 

**Python:**

```python
import asyncio
import json
import os
import subprocess

BASE_URL = "https://api.mistral.ai"
API_KEY = os.environ["MISTRAL_API_KEY"]


async def main() -> None:
    result = subprocess.run(
        [
            "curl", "-s", "-X", "POST",
            f"{BASE_URL}/v1/connectors",
            "-H", f"Authorization: Bearer {API_KEY}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "name": "my_github",
                "description": "GitHub MCP connector for issue and PR management",
                "server": "https://api.githubcopilot.com/mcp/",
                "visibility": "private",
                "auth_scheme": {"type": "http", "scheme": "Bearer"},
            }),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    connector = json.loads(result.stdout)
    if "id" not in connector:
        raise RuntimeError(f"Failed to create connector: {result.stdout}")
    print(f"ID:   {connector['id']}")
    print(f"Name: {connector['name']}")


asyncio.run(main())
```

**curl:**

```bash
curl -X POST "${BASE_URL}/v1/connectors" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_github",
    "description": "GitHub MCP connector for issue and PR management",
    "server": "https://api.githubcopilot.com/mcp/",
    "visibility": "private",
    "auth_scheme": {"type": "http", "scheme": "Bearer"}
  }'
```

**Output:**

```
ID:   a1b2c3d4-5678-90ab-cdef-1234567890ab
Name: my_github
```

| Error | Cause | Fix |
|---|---|---|
| `409 Conflict` | A connector named `my_github` already exists | Choose a different name or delete the existing one first |

---

### 3. Get Authentication Methods

**Goal:** Discover which authentication schemes the connector supports before storing credentials.
> **Note**: the `connector_id_or_name` is a bit ugly and will be replaced by `connector_ref` in future versions.

**Python:**

```python
import asyncio
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    methods = await client.beta.connectors.get_authentication_methods_async(
        connector_id_or_name="my_github",
    )
    for method in methods:
        print(f"Auth type: {method.method_type}")


asyncio.run(main())
```

**curl:**

```bash
curl -X GET "${BASE_URL}/v1/connectors/my_github/authentication_methods" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

**Example output:**

```
Auth type: bearer
```

---

### 4. Store Bearer Credentials

**Goal:** Store named bearer-token credentials on the connector.

Credentials can be stored at three scopes:

| Scope | SDK method | Who can use it |
|---|---|---|
| `user` | `create_or_update_user_credentials` | Only the authenticated user |
| `workspace` | `create_or_update_workspace_credentials` | Everyone in the workspace |
| `organization` | `create_or_update_organization_credentials` | Everyone in the organization |

**Python:**

```python
import asyncio
import os
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    # Credentials A — full repo read access, set as default
    result = await client.beta.connectors.create_or_update_user_credentials_async(
        connector_id_or_name="my_github",
        name="github-pat-full",
        credentials={"bearer_token": os.environ["GITHUB_PAT_FULL"]},
        is_default=True,
    )
    print(result.message)

    # Credentials B — no scopes / invalid token
    result = await client.beta.connectors.create_or_update_user_credentials_async(
        connector_id_or_name="my_github",
        name="github-pat-limited",
        credentials={"bearer_token": os.environ["GITHUB_PAT_LIMITED"]},
    )
    print(result.message)


asyncio.run(main())
```

**curl:**

```bash
# Credentials A — full repo read access (set as default)
curl -X POST "${BASE_URL}/v1/connectors/my_github/user/credentials" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"github-pat-full\",
    \"credentials\": {\"bearer_token\": \"${GITHUB_PAT_FULL}\"},
    \"is_default\": true
  }"

# Credentials B — no scopes / invalid token
curl -X POST "${BASE_URL}/v1/connectors/my_github/user/credentials" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"github-pat-limited\",
    \"credentials\": {\"bearer_token\": \"${GITHUB_PAT_LIMITED}\"}
  }"
```

**Output:**

```
Credentials 'github-pat-full' saved successfully
Credentials 'github-pat-limited' saved successfully
```

**How it works:**
- `is_default: true` marks the credentials as default — calls that omit `credentials_name` will use it.
- Calling the same endpoint again with an existing `name` **updates** the stored token in place.
- The raw token is never returned by list or get endpoints.

| Error | Cause | Fix |
|---|---|---|
| `400 Bad Request` | Empty credentials object | Provide at least `bearer_token` |
| `422 Unprocessable Entity` | Invalid credentials name | Use alphanumeric names with hyphens only |

---

### 5. List Credentials

**Python:**

```python
import asyncio
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    response = await client.beta.connectors.list_user_credentials_async(
        connector_id_or_name="my_github",
    )
    for cred in response.credentials:
        default_marker = " (default)" if cred.is_default else ""
        print(f"  {cred.name}  [{cred.authentication_type}]{default_marker}")


asyncio.run(main())
```

**curl:**

```bash
curl -X GET "${BASE_URL}/v1/connectors/my_github/user/credentials" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

**Output:**

```
  github-pat-full  [bearer] (default)
  github-pat-limited  [bearer]
```

---

### 6. Call a Tool with Specific Credentials

**Python:**

```python
import asyncio
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    # Call with the full-access credentials — should succeed
    result = await client.beta.connectors.call_tool_async(
        connector_id_or_name="my_github",
        tool_name="list_issues",
        arguments={"owner": "octocat", "repo": "hello-world", "state": "open"},
        credentials_name="github-pat-full",
    )
    print(f"[github-pat-full] {result.content[:200]}")

    # Call with the limited/invalid credentials — access error is in the response content
    result = await client.beta.connectors.call_tool_async(
        connector_id_or_name="my_github",
        tool_name="list_issues",
        arguments={"owner": "octocat", "repo": "hello-world", "state": "open"},
        credentials_name="github-pat-limited",
    )
    print(f"[github-pat-limited] {result.content[:200]}")


asyncio.run(main())
```

**curl:**

```bash
curl -X POST "${BASE_URL}/v1/connectors/my_github/call_tool" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "list_issues",
    "arguments": {"owner": "octocat", "repo": "hello-world", "state": "open"},
    "credentials_name": "github-pat-full"
  }'
```

**Example output:**

```
[github-pat-full] [{"number": 42, "title": "Fix typo in README", "state": "open", ...}]
[github-pat-limited] {"error": "Bad credentials", "status": 401}
```

**How it works:**
- `credentials_name` selects which stored credentials the MCP server receives. Omit it to use the default.
- If the named credentials does not exist, the call returns a `404`.

---

### 7. Delete Credentials

> **Note:** You cannot delete the default credentials while other credentials exist. Promote a different credentials to default first, then delete the old one.

**Python:**

```python
import asyncio
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    result = await client.beta.connectors.delete_user_credentials_async(
        connector_id_or_name="my_github",
        credentials_name="github-pat-limited",
    )
    print(result.message)


asyncio.run(main())
```

**curl:**

```bash
curl -X DELETE "${BASE_URL}/v1/connectors/my_github/user/credentials/github-pat-limited" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

**Output:**

```
Credentials 'github-pat-limited' deleted successfully
```

| Error | Cause | Fix |
|---|---|---|
| `404 Not Found` | Credentials name does not exist | Check the name with `list_user_credentials` first |
| `409 Conflict` | Trying to delete the current default while others exist | Promote another credentials to default first |

---

## Part 2: Multiple OAuth2 Credentials (ex: Outlook Calendar MCP)

### Prerequisites (OAuth2)

```bash
MISTRAL_API_KEY=your-mistral-api-key
```

- The `outlook_calendar` connector must be enabled in your workspace. Enable it in the [Mistral Console](https://console.mistral.ai/connectors).
- You need two Microsoft accounts to authenticate separately.

**Script:** `python/src/scripts/08_multiple_oauth_authentication.py`

---

### When to Use Multiple OAuth2 Credentials

- **Multi-account access** — let a single user authenticate with multiple identities (e.g., work and personal Microsoft accounts) and switch between them at call time.
- **Per-user delegation** — each user in a workspace authenticates their own account; `credentials_name` routes tool calls to the right one.
- **Safe rotation** — authenticate the new account under a new name, promote it to default, then revoke the old one.

---

### 1. Get the Outlook Calendar Connector

**Python:**

```python
import asyncio
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    connector = await client.beta.connectors.get_async(
        connector_id_or_name="outlook_calendar",
    )
    print(f"ID:   {connector.id}")
    print(f"Name: {connector.name}")


asyncio.run(main())
```

**curl:**

```bash
curl -X GET "${BASE_URL}/v1/connectors/outlook_calendar" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

---

### 2. Get Authentication Methods

**Python:**

```python
import asyncio
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    methods = await client.beta.connectors.get_authentication_methods_async(
        connector_id_or_name="outlook_calendar",
    )
    for method in methods:
        print(f"Auth type: {method.method_type}")


asyncio.run(main())
```

**curl:**

```bash
curl -X GET "${BASE_URL}/v1/connectors/outlook_calendar/authentication_methods" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

**Example output:**

```
Auth type: oauth2
```

---

### 3. Authenticate Accounts via OAuth2

**Goal:** Obtain OAuth2 authorization URLs and let each account complete the browser flow. The `credentials_name` parameter controls which named slot the resulting token is stored in. Omitting it stores the token as the default credentials.

**Python:**

```python
import asyncio
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    # Account A — stored as the default credentials
    result = await client.beta.connectors.get_auth_url_async(
        connector_id_or_name="outlook_calendar",
        # no credentials_name => stored under name="default"
    )
    print(f"Follow this link to authenticate account A: {result.auth_url}")
    input("Press Enter once done")

    # Account B — stored under the name "personal"
    result = await client.beta.connectors.get_auth_url_async(
        connector_id_or_name="outlook_calendar",
        credentials_name="personal",
    )
    print(f"Follow this link to authenticate account B: {result.auth_url}")
    input("Press Enter once done")


asyncio.run(main())
```

**curl:**

```bash
# Account A — default credentials
curl -X GET "${BASE_URL}/v1/connectors/outlook_calendar/auth_url" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"

# Account B — named "personal"
curl -X GET "${BASE_URL}/v1/connectors/outlook_calendar/auth_url?credentials_name=personal" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

**How it works:**
- `get_auth_url` returns a URL the user must open in a browser to complete the OAuth2 consent flow.
- Once the flow completes, the token is stored automatically under the given `credentials_name` (or as `default` if omitted).
- The script pauses with `input()` to give the user time to complete the browser flow before proceeding.

---

### 4. List Credentials

**Python:**

```python
import asyncio
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    response = await client.beta.connectors.list_user_credentials_async(
        connector_id_or_name="outlook_calendar",
    )
    for cred in response.credentials:
        default_marker = " (default)" if cred.is_default else ""
        print(f"  {cred.name}  [{cred.authentication_type}]{default_marker}")


asyncio.run(main())
```

**Output:**

```
  default  [oauth2] (default)
  personal  [oauth2]
```

---

### 5. Call a Tool with Specific Credentials

**Goal:** Invoke a calendar tool using a named credentials to query a specific account's calendar.

**Python:**

```python
import asyncio
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    # Query the default account
    result = await client.beta.connectors.call_tool_async(
        connector_id_or_name="outlook_calendar",
        tool_name="search_calendar_events",
        arguments={"query": "meeting"},
        credentials_name="default",
    )
    print(f"[default] {result.content[:300]}")

    # Query the personal account
    result = await client.beta.connectors.call_tool_async(
        connector_id_or_name="outlook_calendar",
        tool_name="search_calendar_events",
        arguments={"query": "meeting"},
        credentials_name="personal",
    )
    print(f"[personal] {result.content[:300]}")


asyncio.run(main())
```

**curl:**

```bash
curl -X POST "${BASE_URL}/v1/connectors/outlook_calendar/call_tool" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "search_calendar_events", "arguments": {"query": "meeting"}, "credentials_name": "default"}'
```

---

### 6. Promote Credentials to Default

To change which account is used when `credentials_name` is omitted, update `is_default` without providing a new token — the stored OAuth2 token is preserved.

**Python:**

```python
import asyncio
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    await client.beta.connectors.create_or_update_user_credentials_async(
        connector_id_or_name="outlook_calendar",
        name="personal",
        is_default=True,
    )
    print("Promoted 'personal' to default")

    # Now call without specifying credentials — uses personal account
    result = await client.beta.connectors.call_tool_async(
        connector_id_or_name="outlook_calendar",
        tool_name="search_calendar_events",
        arguments={"query": "meeting"},
    )
    print(f"[default] {result.content[:300]}")


asyncio.run(main())
```

---

### 7. Delete Credentials

> **Note:** You cannot delete the default credentials while other credentials exist. Promote another credentials to default first.

**Python:**

```python
import asyncio
from mistralai import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    for name in ("default", "personal"):
        result = await client.beta.connectors.delete_user_credentials_async(
            connector_id_or_name="outlook_calendar",
            credentials_name=name,
        )
        print(result.message)


asyncio.run(main())
```

**curl:**

```bash
curl -X DELETE "${BASE_URL}/v1/connectors/outlook_calendar/user/credentials/personal" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

---

## Naming Conventions

| Concept | Python |
|---|---|
| Get connector | `client.beta.connectors.get_async(connector_id_or_name=)` |
| Get auth methods | `get_authentication_methods_async(connector_id_or_name=)` |
| Get OAuth2 URL | `get_auth_url_async(connector_id_or_name=, credentials_name=)` |
| Store bearer creds | `create_or_update_user_credentials_async(connector_id_or_name=, name=, credentials={"bearer_token": ...}, is_default=)` |
| Promote to default | `create_or_update_user_credentials_async(connector_id_or_name=, name=, is_default=True)` |
| List user creds | `list_user_credentials_async(connector_id_or_name=)` |
| Delete user creds | `delete_user_credentials_async(connector_id_or_name=, credentials_name=)` |
| Call tool | `call_tool_async(connector_id_or_name=, tool_name=, arguments=, credentials_name=)` |
| Scope: workspace | `*_workspace_credentials*` |
| Scope: organization | `*_organization_credentials*` |

---

## Troubleshooting

**Credentials don't take effect when calling a tool**
- Verify the credentials was saved: `list_user_credentials`.
- Check that `credentials_name` matches the stored name exactly (case-sensitive).
- If `credentials_name` is omitted, the default credentials is used — confirm which is default with `list_user_credentials`.

**OAuth2 token not stored after completing browser flow**
- Make sure you pressed Enter _after_ completing the consent in the browser, not before.
- If the auth URL expired, call `get_auth_url` again to get a fresh one.

**`401 Unauthorized` from the MCP server (GitHub)**
- The PAT may have expired. Re-run `create_or_update_user_credentials` with the same name to rotate it in place.
- The PAT may lack the required scopes.

**Cannot delete the default credentials**
- Promote another credentials to default first: `create_or_update_user_credentials(..., is_default=True)`, then delete the old one.

**`403 Forbidden` on credentials management endpoints**
- Organization-level credentials require `ModifyConnector` organization permission.
- Workspace-level credentials require `ModifyConnector` workspace permission.
- User-level credentials only require authentication.

---

## Error Codes Reference

| HTTP Status | When it occurs | What to do |
|---|---|---|
| `400 Bad Request` | Empty credentials object, or `is_default: false` on the only existing credentials | Provide `bearer_token`; always keep one credentials as default |
| `401 Unauthorized` | Invalid Mistral API key, or the MCP server rejected the stored token | Check your `MISTRAL_API_KEY`; rotate the credentials |
| `403 Forbidden` | Insufficient permissions for the chosen scope | Use a lower scope or request the `ModifyConnector` permission |
| `404 Not Found` | Credentials name or connector does not exist | Verify names with `list_user_credentials` |
| `409 Conflict` | Connector name already taken, or deleting the active default | Rename the connector or promote a different credentials to default first |
| `422 Unprocessable Entity` | Invalid credentials name format | Use alphanumeric characters and hyphens only |
