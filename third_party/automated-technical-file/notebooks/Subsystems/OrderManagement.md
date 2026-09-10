# Order Management Subsystem

## 1. Role
The Order Management subsystem handles the communication with the external Salesman API to fetch new drawing orders and logs them locally.

## 2. Key Components
- **Poller**: `artist_poller.py` runs as a daemon and polls `GET /salesman/orders/incoming?status=new` every 3 minutes.
- **Client**: `salesman_client.py` provides the HTTP interface for fetching and acknowledging orders.
- **Local Ledger**: `order_log.py` manages `orders.xlsx`, keeping a local record of all completed jobs and their associated YouTube proof URLs.

## 3. Workflow
- **Claiming**: The poller identifies a new order, claims it via the API, and triggers `bob_ross.py`.
- **Completion**: Once `bob_ross.py` finishes, the poller (or `bob_ross.py` itself) notifies the Salesman API of completion and records the video proof URL.

## 4. Uncertainty & Contradictions
- **Local vs Remote**: The primary source of truth for orders is the Salesman API, but `orders.xlsx` acts as a local mirror. There is no explicit documentation on how the system resolves discrepancies if the Salesman API is unavailable during a completion call.
- **Authentication**: Authentication for the Salesman API uses a Bearer token stored in `.salesman_config`. There's no documented procedure for token rotation.

---
**Sources:**
- `skills/robot-ross/artist_poller.py`
- `skills/robot-ross/order_log.py`
- `skills/robot-ross/salesman_client.py`
