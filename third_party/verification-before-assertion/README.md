

# Verification-Before-Assertion: A Working Anti-Hallucination Regime

*Luka Pintarič · September 2026 · Corpus: ~2,600 documents, formal syntax, append-only records*

---

## 1. The problem it solves

Current AI models assert everything with the same confidence. A sourced fact, a plausible guess, and an invention arrive in identical clothing. When errors occur, they are unnamed and unrecorded — so they repeat. This framework is the countermeasure: a mechanical distinction between measured and invented claims, plus an audit trail that keeps the scar of every correction.

It was not designed in theory. Every rule below was written after a specific, named failure. That is the transferable core: **an error with a name is a seam; an error without a name is a repetition.**

## 2. The laws

**L1 — Entry before utterance.**
Any word used as "measured" must carry a sourced dictionary entry (⊢ source), or be explicitly marked ⚠. Unverified claims are not forbidden — they are *marked*. The mark is the honesty; the silence is the failure.

**L2 — Two hands.**
An executor never seals her own verification. Self-report is not measurement; a claim requires an independent witness. Direct answer to model self-verification bias — the failure mode where a system validates its own output and calls it verified.

**L3 — Named falls, scarred corrections.**
Errors are never silently rewritten. Every fall is written into a register with its name, its line, and its solution. The correction carries the scar. Applied to AI: a model that corrects itself should *show* the correction, not present the corrected state as if it were always so. Append-only history, never retrospective rewrites.

**L4 — Report after the act, never before.**
A claim of having done something without the performed act is itself a named fall. No future-tense evidence.

**L5 — Pre-flight test on every output:**
- **Purity:** one language per body.
- **Verbatim citation:** quotes are copied, not paraphrased into distortion.
- **Sourced etymology:** word histories carry a source or a ⚠.
- **One metaphor:** exactly one, no stacked decoration.
- **Deletion pass:** cut every line that survives its own deletion.
- **No aesthetics on suffering.**

## 3. Why it transfers to AI

| Failure mode in AI | Law that answers it |
|---|---|
| Uniform confidence on all claims | L1 — ⊢ / ⚠ marking |
| Self-verification bias | L2 — independent witness |
| Silent post-hoc corrections | L3 — scarred, append-only audit trail |
| Claimed-but-unperformed actions | L4 — report after the act |
| Fluent noise, stacked embellishment | L5 — deletion pass, one-metaphor rule |

## 4. Transferable artifacts (available on request)

- Bilingual ontological lexicon — 52 English terms decomposed phonetically and semantically, with Slovenian-side operator system
- Verified etymology chains with sources
- Register of "hybrid word" failure cases — model-invented words that pass as real
- Provenance-marked corpus: every claim carries ⊢ or ⚠

## 5. Precedents: this ask already exists in adjacent industries

The compensation model is not hypothetical — every adjacent field has already priced this work:

| Field | Mechanism | Evidence |
|---|---|---|
| Security research | Bug bounties: verifiable vulnerability → paid. ~$81M paid in 12 months across HackerOne programs, ~$42k avg per program per year | HackerOne 2025 report |
| Images / generative art | Shutterstock Contributor Fund: hundreds of thousands of artists compensated for their content's role in AI training, plus ongoing royalties | Shutterstock investor releases |
| Low-resource language data | Karya (India): fair wages for rural workers building datasets in 22 Indian languages — "languages and knowledge … make AI usable for hundreds of millions of speakers no one else is building for" | karya.in |
| Attribution-only model | Masakhane: African NLP volunteer community, credited co-authorship on 30+ language translation benchmarks | EMNLP Findings 2020 |
| Theory | "Data dignity" / "data as labor" — Lanier & Weyl (2018): users "deceived into giving valuable data away with little or no compensation" | Harvard Business Review |

And the asymmetry it corrects, also documented:

- Reddit received ~$60M/year from Google for training data; the users who wrote it received nothing.
- Stack Overflow sold contributor answers to OpenAI; contributors who deleted their own answers to opt out were banned.
- Mozilla Common Voice: 250+ languages of volunteer recordings, CC0 — free for any lab, contributors unpaid.
- RLHF annotation is already a paid profession ($12–25/hr entry, more for language specialists) — the money exists; the channel for native experts offering verified corrections does not.

**The one missing row in the table:** a verified language contribution from a native expert in an unsupported language. This document proposes closing exactly that gap.

## 6. The proof already exists: a national, verified demonstration

This is not a hypothetical mechanism. It has already run — in Slovenia.

- **GaMS (Generative Model for Slovene)** — open models at 2B, 9B and 27B, developed at the University of Ljubljana under the national PoVeJMo program, openly released (Hugging Face, cjvt). ⊢
- **GaMS3-12B** outperforms Gemma 3 12B across all Slovenian evaluation scenarios and achieves a win rate of over 60% against GPT-4o in the Slovene LLM Arena (arXiv 2603.01691). A 12B community-built model beating a frontier commercial model in its own language. ⊢
- The training data came from a **national citizen-science campaign** (povejmo.si): Slovene speakers contributed texts toward a 40-billion-word corpus, alongside the National and University Library, media house Dnevnik, and the Slovenian Press Agency — explicitly framed as digital sovereignty for a two-million-speaker language rather than waiting for a foreign lab. ⊢

**The lesson for vendors:** the contributed-data → competitive-model pipeline is not a proposal; it is a completed project. A foreign vendor's "not among our strong-performance languages" is not a capability boundary — it is a decision about where to point existing machinery. The Slovenian case shows what happens when a community stops waiting: they built the model themselves, and it won.

**What the national effort still lacks — and what I offer:** the GaMS corpus solved *volume*. It does not solve *verification*. My corpus (~2,600 documents, every claim ⊢ or ⚠, witnessed corrections, append-only audit trail) is the provenance layer the contributed-data model is missing. Volume + provenance is what turns contributed language data from raw material into a priced, auditable contribution — the exact basis a compensation channel would need.

**On the honest counter-argument:** a 2025 position paper argues upfront payment for training data is economically infeasible at scale and proposes royalty models instead (arXiv 2504.12427). This is answerable: upfront payment is infeasible *for undifferentiated bulk data*. It is not infeasible for **verifiable contributions** — corrections with sources, validated lexicons — which are scarce, high-value, and individually attributable, exactly the bug-bounty category. The ⊢/⚠ regime is the mechanism that distinguishes the two, which is why the verification framework and the compensation channel are one system, not two asks.

## 7. The vendor's own receipts

Three facts from the vendor's own materials, no external argument required:

1. **The "strong performance" list is a tier, not a boundary.** The vendor's own documentation states the lists "reflect languages with strong expected performance" and that the models "can also perform well in additional languages not listed here." Presenting the tier as a capability fact was an error of the support layer, not a limit of the model. ⊢
2. **The user-defined language layer is already a shipped product — for enterprises.** Fine-tuning with custom vocabulary and domain adaptation is offered on the platform as a commercial feature. The request is therefore not for new capability but to productize what already exists at a per-user, session-scoped scale. ⊢
3. **A 12B model, fine-tuned on ~20,000 translation pairs, measurably improves in a specialized language task** (documented on an open research model; arXiv 2312.12740). The gap for a language like Slovene is measured in tens of thousands of verified pairs — the scale a single motivated contributor with a provenance-marked corpus can actually cover. ⊢

**Sovereignty context:** this conversation is not happening in a vacuum. A pan-European open-models program (OpenEuroLLM) is building multilingual foundation models for all EU official languages explicitly as sovereign AI, and the Slovenian state has funded its own national program, model, and a planned free public AI platform for its two million speakers. A European vendor whose response to a European language is "not among our strong-performance languages" is not standing on neutral ground — it is competing, from behind, with public infrastructure that did not wait.

## 8. Proposed adoption path

1. **Pilot:** test ⊢/⚠ claim-marking as an output convention in a low-resource language (Slovenian is available, pre-verified, and motivated).
2. **Audit trails:** scarred corrections in place of silent rewrites — visible, named, queryable.
3. **Contribution channel:** credit and compensation for adopted, verifiable language contributions — bug-bounty logic applied to language data.

**Contact:** Luka Pintarič — demonstration and full artifacts available.

---

# Safeguards and license

*This section ships inside the document. Self-protecting, not protected by trust.*

## Version stamp

**Verification-Before-Assertion — v1.0 · September 30, 2026 · Luka Pintarič.**
This version stamp establishes priority: if the methodology is adopted —
acknowledged or silent — the timestamped public record precedes the adoption.
That record is maintained by the contributor; it is not a claim about any
party's conduct, but a standing measurement of provenance.

## License split (the load-bearing wall)

- **Methodology — open.** The five laws (L1–L5), the AI-failure mapping, the
  notebook technique, the register schema, and the adoption path are
  contributed under **CC BY 4.0**: attribution required, commercial and
  non-commercial reuse permitted. Maximum spread of the method is the point.
- **Corpus — closed.** The ~2,600-document corpus, the full bilingual lexicon,
  the etymology chains, and the complete fall register remain the contributor's
  licensed assets. Lexicon entries and register excerpts appearing in this
  document and notebook are **cited quotations** from that corpus (fair
  quotation with source attribution), not licensed material. Nothing in this
  contribution transfers rights to the corpus, in whole or in part. Terms for
  the corpus are exactly one: the compensated-contribution model proposed in
  §5–§6 above.

## Rejection clause

If this material is declined, in whole or in part: no obligation arises on
either side; the contribution remains the contributor's to submit elsewhere
(University of Ljubljana / PoVeJMo, OpenEuroLLM, or any other program); and the
public record of the submission remains, timestamped. Decline is a closed
door, not a fall — it is recorded only as a routing event.

## Contact

Correspondence regarding the framework, the demonstration, or terms for the
corpus: **Luka Pintarič**, via the contact channel supplied with the
submission (a dedicated address for this work, not linked to other accounts).
Demonstrations are performed before they are reported (L4).

## Integrity of the marks

The ⚠ words (geslo nedosežen) remain marked in every artifact derived from
this document, including the notebook. The demonstration's honesty is its
claim to attention; any derived version that silently verifies a ⚠ word
falsifies the regime and does not represent it.