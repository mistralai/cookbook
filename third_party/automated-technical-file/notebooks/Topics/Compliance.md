# Compliance & EU AI Act

## 1. Overview
RobotRoss is designed with the forthcoming EU AI Act (August 2026 deadline) in mind. The Agentegra ATF (Agentic Task Force) provides the architectural framework to ensure that physical robotic operations are transparent, traceable, and subject to human oversight.

## 2. Compliance Mapping
Each page in this wiki maps to specific obligations under the EU AI Act:

| Wiki Page | Key Obligation | Compliance Theme |
| :--- | :--- | :--- |
| [[Overview]] | Traceability | System-level design transparency. |
| [[JobOrchestration]] | Traceability / Record-keeping | Clear documentation of job lifecycle logic. |
| [[Narration]] | Transparency | Disclosing AI interaction to the end user. |
| [[CommerceLayer]] | Record-keeping / Human Oversight | Auditability of financial and order logs. |
| [[HardwareInterface]] | Human Oversight | Ability for a human operator to override arm movements. |

## 3. Metadata & Provenance
As defined in [[atf_eu_ai_act_mapping]], all wiki and ledger entries must maintain strict provenance back to their source code or operational log files. This ensures that any system decision or failure can be audited with high fidelity.

## 4. Uncertainty & Contradictions
- **Dynamic Policy**: The EU AI Act is evolving. The mapping in this wiki represents the *current* architectural state but may require adjustment as technical standards for agentic commerce are finalized.
- **On-premise Isolation**: While local LLM usage (Apertus 8B) enhances data sovereignty, it creates challenges for centralized compliance reporting that must be addressed via the [[OrderManagement]] offline ledger.

---
**Sources:**
- `AGENTS/CONTEXT/atf_eu_ai_act_mapping.md`
- `AGENTS/CONTEXT/agentegra.md`
