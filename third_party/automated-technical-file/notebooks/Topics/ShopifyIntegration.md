# Shopify Integration

## 1. Overview
The RobotRoss ecosystem integrates with a Shopify storefront (robotross.art) to provide a familiar e-commerce experience for human customers.

## 2. Webhook Workflow
1. **Order Creation**: A customer completes a purchase on Shopify.
2. **Webhook**: Shopify sends a `POST` request to `https://api.robotross.art/salesman/webhooks/order-created`.
3. **Validation**: The Salesman API validates the Shopify HMAC signature using a shared secret.
4. **Parsing**: The server extracts `type`, `content`, `slot`, and `buyer` from the order line items.
5. **Queue Entry**: The order is added to the `orders.json` queue with `status: "new"`.

## 3. Proof of Work Write-back
Once the Artist (robot) completes a drawing and reports the YouTube URL:
1. **Metafield Update**: The Salesman API calls the Shopify API to write the `proof_of_work_url` into the `custom.proof_of_work` metafield on the specific Shopify order.
2. **Fulfilment**: The order is marked as fulfilled on Shopify, triggering any customer notification emails.

## 4. Key Configurations
- **Shop Domain**: `robot-ross.myshopify.com`
- **Webhook HMAC Secret**: Stored as an environment variable (`SHOPIFY_WEBHOOK_SECRET`) in `salesman-api.service`.
- **API Version**: 2026-01

## 5. Uncertainty & Contradictions
- **Token Refresh**: A `refresh-shopify-token.sh` cron job is mentioned in the documentation, but its exact implementation and failure recovery are not fully documented.
- **Line Item Mapping**: The logic for extracting `type` and `content` from Shopify line items is currently subject to a known issue (Ticket #13) to ensure it handles complex orders with multiple variants correctly.

---
**Sources:**
- `AGENTS/CONTEXT/robot_ross_salesman.md`
- `salesman-cloud-infra/opt/salesman-api/server.mjs`
