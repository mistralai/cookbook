# Bidding & Wall of Fame Rules

## 1. Overview
The Wall of Fame is a physical 8x8 grid (64 slots total, A1 through H8) where drawings are pinned. Drawings stay on the wall until their specific slot is "outbid" by another customer.

## 2. Competitive Overwrite Rule
To maintain a fair and dynamic physical board, the following rules apply:
1. **Minimum Entry**: The base price for any empty or available slot is 5.00 USD/USDC.
2. **The 120% Rule**: To overwrite an existing drawing in a specific slot, the new bid must be at least 120% of the current price of that slot.
3. **Price Persistence**: Once a slot is outbid, the new price becomes the base for future bids. Prices in high-demand slots naturally escalate.

## 3. Slot Allocation (The Slot Sniffer)
If an order does not explicitly specify a slot (e.g., via a Shopify line item), the `server.mjs` "Slot Sniffer" logic kicks in:
1. **Content Parsing**: The server scans the order `content` for alphanumeric identifiers like "A3", "H7", etc.
2. **Defaulting**: If no slot is identified, the order is placed into the first available slot (lowest price/oldest entry) based on the current state of `orders.json`.

## 4. Physical Requirements
- **64 slots**: Physical limits of the board (A1-H8).
- **20% Overwrite**: Simplified phrasing used in some marketing materials to describe the 120% bid rule.

## 5. Uncertainty & Contradictions
- **Slot Conflict**: It is not explicitly documented what happens if two identical bids for the same slot arrive at the exact same millisecond. The current system relies on sequential JSON file updates.
- **Physical Removal**: While the software tracks the "overwrite", the actual physical removal of the old drawing and placement of the new one is performed by the Artist (robot) as part of the job sequence, but the disposal/archiving of old drawings is not documented.

---
**Sources:**
- `AGENTS/CONTEXT/robot_ross_salesman.md`
- `salesman-cloud-infra/opt/salesman-api/server.mjs`
