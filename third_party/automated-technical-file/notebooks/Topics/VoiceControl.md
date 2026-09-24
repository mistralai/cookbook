# Voice Control

## 1. Overview
Voice control is a top-level part of the ATF architecture, not a side note. It provides spoken interaction with the same local evidence base used by the text query path.

## 2. Speech-to-Text
Voxtral is the speech-to-text engine for converting operator prompts into local text queries. Spoken prompts are interpreted against the compiled wiki and the operational ledger.

## 3. Reasoning Path
The model answers from the RobotRoss knowledge corpus and ledger evidence. This query path is designed to run with the same grounding and provenance expectations as the text channel — every spoken answer traces back to a source document.

## 4. Text-to-Speech
Voxtral is also used for spoken output. Spoken answers summarise the same evidence-backed response returned in the text channel.

## 5. Notes and Open Points
- Voice interaction should remain aligned with the same provenance expectations as the text query path.
- The UI should present voice as a first-class control surface once the runtime hookup is complete.
- The voice path intentionally runs against the same Document Library as the Q&A endpoint — no separate knowledge base.

---
**Sources:**
- `AGENTS/CONTEXT/agentegra_atf_architecture.md`
