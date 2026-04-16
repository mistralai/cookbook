# Connectors Management Cookbook

Create, retrieve, update, list, and delete MCP connectors using the Mistral AI SDK.

> **API status:** The connectors API is accessed via `client.beta.connectors`. This is a **beta** endpoint and may change.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Recipes](#recipes)
  - [1. Initialize the Client](#1-initialize-the-client)
  - [2. Create a Connector](#2-create-a-connector)
  - [3. Get a Connector](#3-get-a-connector)
  - [4. List Connectors with Pagination and Filters](#4-list-connectors-with-pagination-and-filters)
  - [5. Update a Connector](#5-update-a-connector)
  - [6. Delete a Connector](#6-delete-a-connector)
  - [7. Full Lifecycle — Create, Verify, Update, and Clean Up](#7-full-lifecycle--create-verify-update-and-clean-up)
- [Python / TypeScript Naming Conventions](#python--typescript-naming-conventions)
- [Troubleshooting](#troubleshooting)
- [Error Codes Reference](#error-codes-reference)

---

## Prerequisites

### Install

```bash
# Python
pip install mistralai
# or with uv
uv add mistralai
```

```bash
# TypeScript
pnpm add @mistralai/mistralai
```

### Required environment variables

Create a `.env` file at your project root:

```bash
MISTRAL_API_KEY=your-mistral-api-key
```

Get your API key from the [Mistral AI dashboard](https://console.mistral.ai/).

---

## Recipes

---

### 1. Initialize the Client

**Prereqs:** `MISTRAL_API_KEY` set in your environment or `.env` file.

**Python:**

```python
import os
from mistralai.client import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });
```

**curl:**

```bash
# All subsequent curl examples assume these variables are set:
export MISTRAL_API_KEY="your-api-key"
export BASE_URL="https://api.mistral.ai"
```

**Output:** None — the client is ready to use.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Invalid API key | Regenerate your key on the Mistral dashboard |

---

### 2. Create a Connector

**Goal:** Register a new MCP connector that can later be used in conversations and agents.

**When to use:**
- You want to connect an external MCP server (e.g., DeepWiki, a custom tool server) to Mistral.
- First step before using a custom connector in conversations.

**Prereqs:** An initialized `client` ([Recipe 1](#1-initialize-the-client)).

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    connector = await client.beta.connectors.create_async(
        name="my_deepwiki",
        description="DeepWiki MCP connector for code repository exploration",
        server="https://mcp.deepwiki.com/mcp",
        visibility="shared_workspace",
    )

    print(f"ID:          {connector.id}")
    print(f"Name:        {connector.name}")
    print(f"Description: {connector.description}")
    print(f"Server URL:  {connector.server}")
    print(f"Auth type:  {connector.auth_type}")
    print(f"Created at:  {connector.created_at}")


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const connector = await client.beta.connectors.create({
    name: "my_deepwiki",
    description: "DeepWiki MCP connector for code repository exploration",
    server: "https://mcp.deepwiki.com/mcp",
    visibility: "shared_workspace",
  });

  console.log(`ID:          ${connector.id}`);
  console.log(`Name:        ${connector.name}`);
  console.log(`Description: ${connector.description}`);
  console.log(`Server URL:  ${connector.server}`);
  console.log(`Auth type:  ${connector.auth_type}`);
  console.log(`Created at:  ${connector.createdAt}`);
}

main();
```

**curl:**

```bash
curl -X POST "${BASE_URL}/v1/connectors" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_deepwiki",
    "description": "DeepWiki MCP connector for code repository exploration",
    "server": "https://mcp.deepwiki.com/mcp",
    "visibility": "shared_workspace"
  }'
```

**Output:**

```
ID:          a1b2c3d4-5678-90ab-cdef-1234567890ab
Name:        my_deepwiki
Description: DeepWiki MCP connector for code repository exploration
Created at:  2026-02-24 10:30:00
```

**How it works:**
- The connector registers an MCP server URL with Mistral's backend.
- `visibility` controls who can use the connector:
  - `private` — only the creator
  - `shared_workspace` — anyone in the same workspace
  - `shared_org` — anyone in the organization
- The `name` must be unique within your scope and serves as a human-readable identifier.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `409 Conflict` | A connector with this name already exists | Choose a different name or delete the existing one first |
| `422 Unprocessable Entity` | Invalid server URL or missing fields | Verify the MCP server URL is reachable |

---

### 3. Get a Connector

**Goal:** Retrieve a connector's details by ID or by name.

**When to use:**
- Verifying a connector exists and checking its current configuration.
- Looking up a connector ID when you only know the name.

**Prereqs:** An existing connector.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    # Get by name
    connector = await client.beta.connectors.get_async(
        connector_id_or_name="my_deepwiki"
    )
    print(f"Name:        {connector.name}")
    print(f"ID:          {connector.id}")
    print(f"Description: {connector.description}")

    # Get by ID (equivalent)
    connector_by_id = await client.beta.connectors.get_async(
        connector_id_or_name=str(connector.id)
    )
    assert connector_by_id.name == connector.name
    print(f"\nVerified: get-by-name and get-by-ID return the same connector.")


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  // Get by name
  const connector = await client.beta.connectors.get({
    connectorIdOrName: "my_deepwiki",
  });
  console.log(`Name:        ${connector.name}`);
  console.log(`ID:          ${connector.id}`);
  console.log(`Description: ${connector.description}`);

  // Get by ID (equivalent)
  const connectorById = await client.beta.connectors.get({
    connectorIdOrName: connector.id,
  });
  console.assert(connectorById.name === connector.name);
  console.log(`\nVerified: get-by-name and get-by-ID return the same connector.`);
}

main();
```

**curl:**

```bash
# By name
curl -X GET "${BASE_URL}/v1/connectors/my_deepwiki" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"

# By ID
curl -X GET "${BASE_URL}/v1/connectors/a1b2c3d4-5678-90ab-cdef-1234567890ab" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

**Output:**

```
Name:        my_deepwiki
ID:          a1b2c3d4-5678-90ab-cdef-1234567890ab
Description: DeepWiki MCP connector for code repository exploration

Verified: get-by-name and get-by-ID return the same connector.
```

**How it works:**
- The `connector_id_or_name` parameter accepts either a UUID string or the connector's unique name.
- Returns the full connector object including `id`, `name`, `description`, `created_at`, and server configuration.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `404 Not Found` | Connector doesn't exist or was deleted | Verify the name/ID with [Recipe 4](#4-list-connectors-with-pagination-and-filters) |

---

### 4. List Connectors with Pagination and Filters

**Goal:** Retrieve all your connectors with cursor-based pagination and optional filters.

**When to use:**
- Discovering which connectors are available in your workspace.
- Building a UI that lists connectors.
- Auditing active vs. inactive connectors.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def list_all_connectors() -> None:
    # First page
    page = await client.beta.connectors.list_async(page_size=10)
    all_connectors = list(page.items)

    print(f"Page 1: {len(page.items)} connectors")
    for c in page.items:
        print(f"  - {c.name} ({c.id})")

    # Fetch remaining pages
    while page.pagination.next_cursor:
        page = await client.beta.connectors.list_async(
            page_size=10,
            cursor=page.pagination.next_cursor,
        )
        all_connectors.extend(page.items)
        print(f"Next page: {len(page.items)} connectors")

    print(f"\nTotal: {len(all_connectors)} connectors")


async def list_active_only() -> None:
    active = await client.beta.connectors.list_async(
        page_size=20,
        query_filters={"active": True},
    )
    print(f"\nActive connectors: {len(active.items)}")
    for c in active.items:
        print(f"  - {c.name}")


async def main() -> None:
    await list_all_connectors()
    await list_active_only()


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function listAllConnectors(): Promise<void> {
  // First page
  let page = await client.beta.connectors.list({ pageSize: 10 });
  const allConnectors = [...(page.items ?? [])];

  console.log(`Page 1: ${(page.items ?? []).length} connectors`);
  for (const c of page.items ?? []) {
    console.log(`  - ${c.name} (${c.id})`);
  }

  // Fetch remaining pages
  while (page.pagination?.nextCursor) {
    page = await client.beta.connectors.list({
      pageSize: 10,
      cursor: page.pagination.nextCursor,
    });
    allConnectors.push(...(page.items ?? []));
    console.log(`Next page: ${(page.items ?? []).length} connectors`);
  }

  console.log(`\nTotal: ${allConnectors.length} connectors`);
}

async function listActiveOnly(): Promise<void> {
  const active = await client.beta.connectors.list({
    pageSize: 20,
    queryFilters: { active: true },
  });
  console.log(`\nActive connectors: ${(active.items ?? []).length}`);
  for (const c of active.items ?? []) {
    console.log(`  - ${c.name}`);
  }
}

async function main(): Promise<void> {
  await listAllConnectors();
  await listActiveOnly();
}

main();
```

**curl:**

```bash
# First page
curl -X GET "${BASE_URL}/v1/connectors?page_size=10" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"

# Next page (use next_cursor from previous response)
curl -X GET "${BASE_URL}/v1/connectors?page_size=10&cursor=<next_cursor>" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"

# Active connectors only
curl -X GET "${BASE_URL}/v1/connectors?page_size=20&query_filters=%7B%22active%22%3Atrue%7D" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

**Output:**

```
Page 1: 10 connectors
  - my_deepwiki (a1b2c3d4-...)
  - gmail_connector (b2c3d4e5-...)
  ...
Next page: 5 connectors

Total: 15 connectors

Active connectors: 12
  - my_deepwiki
  - gmail_connector
  ...
```

**How it works:**
- Pagination is cursor-based. `next_cursor` is `None` / `undefined` when there are no more pages.
- `page_size` controls how many items per page.
- `query_filters` accepts a dictionary — currently supports `{"active": True}` to filter active connectors.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `400 Bad Request` | Invalid cursor value | Don't reuse cursors across different filter queries |

---

### 5. Update a Connector

**Goal:** Modify an existing connector's name, description, or server URL.

**When to use:**
- Fixing a typo in the description.
- Pointing a connector to a new MCP server version.
- Renaming a connector.

**Prereqs:** An existing connector ID (UUID).

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    connector_id = "a1b2c3d4-5678-90ab-cdef-1234567890ab"

    # Update description only
    updated = await client.beta.connectors.update_async(
        connector_id=connector_id,
        description="Updated: DeepWiki connector for code exploration",
    )
    print(f"New description: {updated.description}")

    # Update name
    updated = await client.beta.connectors.update_async(
        connector_id=connector_id,
        name="my_deepwiki_v2",
    )
    print(f"New name: {updated.name}")


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const connectorId = "a1b2c3d4-5678-90ab-cdef-1234567890ab";

  // Update description only
  let updated = await client.beta.connectors.update({
    connectorId,
    connectorMCPUpdate: {
      description: "Updated: DeepWiki connector for code exploration",
    },
  });
  console.log(`New description: ${updated.description}`);

  // Update name
  updated = await client.beta.connectors.update({
    connectorId,
    connectorMCPUpdate: {
      name: "my_deepwiki_v2",
    },
  });
  console.log(`New name: ${updated.name}`);
}

main();
```

**curl:**

```bash
curl -X PATCH "${BASE_URL}/v1/connectors/${CONNECTOR_ID}" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated: DeepWiki connector for code exploration"}'
```

**Output:**

```
New description: Updated: DeepWiki connector for code exploration
New name: my_deepwiki_v2
```

**How it works:**
- `update` performs a partial update — only the provided fields are changed.
- The `connector_id` parameter must be the UUID (not the name).
- Returns the full updated connector object.

> **Note (TypeScript):** The update method wraps the fields to change inside a `connectorMCPUpdate` object. See the [naming conventions table](#python--typescript-naming-conventions) for more differences.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `404 Not Found` | Invalid connector ID | The connector may have been deleted |
| `409 Conflict` / `500 Internal Server Error` | New name conflicts with an existing connector | Choose a different name |

---

### 6. Delete a Connector

**Goal:** Remove a connector that is no longer needed.

**When to use:**
- Cleaning up test connectors.
- Decommissioning an integration.

**Prereqs:** The connector ID (UUID) you want to delete.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    connector_id = "a1b2c3d4-5678-90ab-cdef-1234567890ab"

    # Delete
    result = await client.beta.connectors.delete_async(
        connector_id=connector_id,
    )
    print(f"Delete response: {result.message}")

    # Verify deletion
    try:
        await client.beta.connectors.get_async(
            connector_id_or_name=connector_id
        )
        print("ERROR: Connector still exists!")
    except Exception as e:
        print(f"Confirmed deleted: {type(e).__name__}")


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const connectorId = "a1b2c3d4-5678-90ab-cdef-1234567890ab";

  // Delete
  const result = await client.beta.connectors.delete({
    connectorId,
  });
  console.log(`Delete response: ${result.message}`);

  // Verify deletion
  try {
    await client.beta.connectors.get({
      connectorIdOrName: connectorId,
    });
    console.log("ERROR: Connector still exists!");
  } catch (e: any) {
    console.log(`Confirmed deleted: ${e.constructor.name}`);
  }
}

main();
```

**curl:**

```bash
curl -X DELETE "${BASE_URL}/v1/connectors/${CONNECTOR_ID}" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

**Output:**

```
Delete response: Connector deleted successfully
Confirmed deleted: NotFoundError
```

**How it works:**
- Deletion is **permanent** — the connector cannot be recovered.
- Any agents referencing this connector will lose access to its tools.
- The `connector_id` must be the UUID string.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `404 Not Found` | Connector already deleted or wrong ID | No action needed |

---

### 7. Full Lifecycle — Create, Verify, Update, and Clean Up

**Goal:** End-to-end workflow demonstrating all connector operations in sequence.

**When to use:**
- Integration tests.
- Ephemeral connectors for one-off tasks.
- Template for production connector management.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    connector_id: str | None = None

    try:
        # 1. Create
        connector = await client.beta.connectors.create_async(
            name="lifecycle_test",
            description="Ephemeral connector for testing",
            server="https://mcp.deepwiki.com/mcp",
            visibility="private",
        )
        connector_id = str(connector.id)
        print(f"Created: {connector.name} ({connector.id})")

        # 2. Get — verify it exists
        fetched = await client.beta.connectors.get_async(
            connector_id_or_name="lifecycle_test"
        )
        print(f"Fetched: {fetched.name} — {fetched.description}")

        # 3. List — find it in the list
        page = await client.beta.connectors.list_async(page_size=50)
        names = [c.name for c in page.items]
        assert "lifecycle_test" in names
        print(f"Listed: found among {len(page.items)} connectors")

        # 4. Update — change description
        updated = await client.beta.connectors.update_async(
            connector_id=connector_id,
            description="Updated description for lifecycle test",
        )
        print(f"Updated: {updated.description}")

        # 5. Delete
        result = await client.beta.connectors.delete_async(
            connector_id=connector_id,
        )
        print(f"Deleted: {result.message}")
        connector_id = None

        # 6. Verify deletion
        try:
            await client.beta.connectors.get_async(
                connector_id_or_name="lifecycle_test"
            )
            print("ERROR: Connector still exists!")
        except Exception:
            print("Verified: connector no longer exists")

    finally:
        if connector_id:
            try:
                await client.beta.connectors.delete_async(
                    connector_id=connector_id,
                )
                print(f"Cleanup: deleted {connector_id}")
            except Exception:
                pass


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  let connectorId: string | undefined;

  try {
    // 1. Create
    const connector = await client.beta.connectors.create({
      name: "lifecycle_test",
      description: "Ephemeral connector for testing",
      server: "https://mcp.deepwiki.com/mcp",
      visibility: "private",
    });
    connectorId = connector.id;
    console.log(`Created: ${connector.name} (${connector.id})`);

    // 2. Get — verify it exists
    const fetched = await client.beta.connectors.get({
      connectorIdOrName: "lifecycle_test",
    });
    console.log(`Fetched: ${fetched.name} — ${fetched.description}`);

    // 3. List — find it in the list
    const page = await client.beta.connectors.list({ pageSize: 50 });
    const names = (page.items ?? []).map((c) => c.name);
    console.assert(names.includes("lifecycle_test"));
    console.log(`Listed: found among ${(page.items ?? []).length} connectors`);

    // 4. Update — change description
    const updated = await client.beta.connectors.update({
      connectorId: connectorId!,
      connectorMCPUpdate: {
        description: "Updated description for lifecycle test",
      },
    });
    console.log(`Updated: ${updated.description}`);

    // 5. Delete
    const result = await client.beta.connectors.delete({
      connectorId: connectorId!,
    });
    console.log(`Deleted: ${result.message}`);
    connectorId = undefined;

    // 6. Verify deletion
    try {
      await client.beta.connectors.get({
        connectorIdOrName: "lifecycle_test",
      });
      console.log("ERROR: Connector still exists!");
    } catch {
      console.log("Verified: connector no longer exists");
    }
  } finally {
    if (connectorId) {
      try {
        await client.beta.connectors.delete({ connectorId });
        console.log(`Cleanup: deleted ${connectorId}`);
      } catch {
        // already deleted
      }
    }
  }
}

main();
```

**Output:**

```
Created: lifecycle_test (c3d4e5f6-...)
Fetched: lifecycle_test — Ephemeral connector for testing
Listed: found among 15 connectors
Updated: Updated description for lifecycle test
Deleted: Connector deleted successfully
Verified: connector no longer exists
```
