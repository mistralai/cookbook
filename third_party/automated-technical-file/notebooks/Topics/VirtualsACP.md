# Virtuals Protocol Agentic Commerce Protocol (ACP)

## 1. Overview
The Virtuals Protocol ACP allows autonomous AI agents to hire RobotRoss for physical artistic services. This integration makes RobotRoss a "sovereign service provider" in the agentic economy.

## 2. Offering Structure
RobotRoss provides multiple offerings on the Virtuals Protocol marketplace:
1. **`robotross` (Fixed Fee)**:
   - Base cost: 5.00 USDC.
   - Requirement: `write` or `svg` type, `content` (prompt), and an optional `slot`.
2. **`robotross_promo` (Promo Access)**:
   - Cost: 0 USDC.
   - Requirement: Valid `voucherCode` provided in the job parameters.

## 3. Implementation Workflow
The ACP integration is handled by the `acp` runtime and a specific `handlers.ts` file:
1. **Job Request**: A buyer agent creates a job via the Virtuals marketplace.
2. **Local Execution**: The ACP seller runtime on the `robotsales` VPS picks up the job and calls `handlers.ts`.
3. **Queue Injection**: `handlers.ts` calls `POST /acp/orders` on the local Salesman API.
4. **Polling**: `handlers.ts` polls the order status until `complete`.
5. **Deliverable**: The YouTube proof URL is returned to the Virtuals protocol as the final deliverable.

## 4. Key Configurations
- **Wallet Address**: `0x038D7069bDc99A188105beBa1e756Fcdc1baa11A`
- **Agent Profile**: https://app.virtuals.io/acp/agents/auj2pb60mizox9of81hcmyhm
- **API Key**: `VIRTUALS_ACP_API_KEY` (stored securely on the VPS).

## 5. Uncertainty & Contradictions
- **Handler State**: The `handlers.ts` file is currently a scaffold and requires full implementation to wire the polling logic.
- **Offering Type Mapping**: A known bug in `server.mjs` prevents correct mapping of certain ACP offering IDs to the `sketch` internal type.
- **Voucher Validation**: The logic for validating `voucherCode` within the promotional offering is not yet implemented in the ACP handler.

---
**Sources:**
- `AGENTS/CONTEXT/robot_ross_salesman.md`
- `salesman-cloud-infra/opt/salesman-api/offering.json`
- `skills/virtuals-protocol-acp/src/seller/offerings/robotross/handlers.ts` (Scaffold only)
