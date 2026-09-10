# Commerce Layer Subsystem

## 1. Role
The Commerce Layer is the central cloud-based "brain" of the RobotRoss ecosystem. It manages the global order queue, competitive bidding for the physical Wall of Fame, and provides the public-facing API for integrations with Shopify, Virtuals ACP, and other 3rd party agents.

## 2. Main API Server: `server.mjs`
This Node.js server (running on a DigitalOcean VPS) coordinates all commerce activities:
1. **Order Intake**: Accepts orders from Shopify webhooks, Virtuals ACP, and direct API calls.
2. **Competitive Bidding**: Enforces a 120% overwrite rule for slots on the 64-position (A1-H8) Wall of Fame.
3. **Queue Management**: Status-based tracking (`new`, `claimed`, `in_progress`, `complete`, `failed`).
4. **Artist Handshake**: Provides endpoints for the Artist (Mac Mini) to poll for and claim jobs.
5. **Proof Delivery**: Manages the `/proof/:id` branded delivery pages that redirect to YouTube.

## 3. Order Lifecycle
The full lifecycle of an order across the Commerce Layer:
1. **Creation**: Order enters via `POST /salesman/orders` or webhook.
2. **Polling**: Artist polls `GET /salesman/orders/incoming?status=new`.
3. **Claim/ACK**: Artist claims and acknowledges the start of work.
4. **Completion**: Artist uploads the YouTube URL via `POST /salesman/orders/:id/complete`.
5. **Fulfilment**: Server triggers write-back to the origin (e.g., Shopify metadata).

## 4. Key Configurations
- **Database**: `/var/lib/salesman-api/orders.json` (flat file for simplicity and auditability).
- **Environment**: Managed via `systemd` on the `robotsales` VPS.
- **Offering**: `offering.json` defines the Virtuals ACP marketplace entry.

## 5. Uncertainty & Contradictions
- **Atomic Locking**: The use of a single `orders.json` flat file for persistence implies that high-concurrency writes must be handled carefully. The current implementation uses simple file-system writes; as volume grows, this may become a point of contention.
- **ACP Mapping**: A known bug exists in `server.mjs` where `svg` offering IDs may map incorrectly to the `write` type instead of `sketch`.
- **Shopify Metafields**: The process for writing back proof-of-work URLs to Shopify relies on a specific metafield (`custom.proof_of_work`), but the recovery path if this fails is not fully automated.

---
**Sources:**
- `AGENTS/CONTEXT/robot_ross_salesman.md`
- `salesman-cloud-infra/opt/salesman-api/server.mjs`
- `salesman-cloud-infra/opt/salesman-api/offering.json`
