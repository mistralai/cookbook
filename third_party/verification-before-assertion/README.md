
# Verification-Before-Assertion: A Working Anti-Hallucination Regime

**Luka Pintarič · v1.1 · September 30, 2026**
*Corpus: ~2,600 documents, formal syntax, append-only records. Companion Slovenian edition: REZ body (same version, same laws).*

## 1. The problem

Current AI models assert everything in the same voice. A sourced fact, a plausible guess, and an invention arrive in identical clothing. When errors occur they are unnamed, so they repeat. This framework is the countermeasure: a mechanical distinction between measured and invented claims, plus a record that keeps the scar of every correction.

It was not designed in theory. Every rule below was written after a specific, named failure. That is the transferable core: **an error with a name is a seam; an error without a name is a repetition.**

## 2. The five laws

**L1 — Entry before utterance.** Any word used as "measured" must carry a sourced dictionary entry (⊢ source), or be explicitly marked ⚠. Unverified claims are not forbidden; they are marked. The mark is the honesty — the silence is the failure.

**L2 — Two hands.** An executor never seals her own verification. Self-report is not measurement; a claim requires an independent witness. This answers model self-verification bias directly.

**L3 — Named falls, scarred corrections.** Errors are never silently rewritten. Every fall is entered into a register with its name, its line, and its solution. The correction carries the scar. Applied to AI: a model that corrects itself should *show* the correction, not present the corrected state as if it were always so.

**L4 — Report after the act, never before.** Claiming a performed act without the performed act is itself a named fall.

**L5 — Pre-flight test on every output:** one language per body; verbatim citation; sourced etymology or ⚠; one metaphor per body; a deletion pass (cut every line that survives its own deletion); no aesthetics on suffering.

## 3. Why it transfers to AI

| Failure mode in AI | Law that answers it |
|---|---|
| Uniform confidence on all claims | L1 — ⊢ / ⚠ marking |
| Self-verification bias | L2 — independent witness |
| Silent post-hoc corrections | L3 — append-only, scarred audit trail |
| Claimed-but-unperformed actions | L4 — report after the act |
| Fluent noise, stacked embellishment | L5 — deletion pass, one-metaphor rule |

## 4. Evidence: the regime catching its own falls

This is no longer a description. In the current register, the regime was run against its own wordlist (SSKJ², fran.si, accessed September 30, 2026). Two falls were caught and recorded on the same day:

1. **lúka** — a seed entry had sealed a working gloss ("opening") with a ⊢ mark. The dictionary entry for *lúka* means *harbor*; "opening" belongs to *lúknja*. A mark applied before an entry — the exact failure L1 exists to catch — found in our own seed, named, corrected with the scar kept.
2. **rodovnik** — a recorded fall from September 24 was misnamed. The word *does* have a dictionary entry (genealogical tree); what was invented was a *meaning*, not a word. The fall was renamed: a fall is named after the thing that fell.

Result of the run: 20 ⊢ entries, each carrying a verbatim citation with source and access date; 3 ⚠ entries, each carrying a *measured* reason for being open (including one invented word verified as absent from every dictionary on the portal). Machine-readable body: `wordlist-v0.6.json`.

A third, structural proof is the document you are reading: v1.1 exists because v1.0 was measured against its own L5 and failed (mixed languages, stacked metaphors) — and the rewrite shows the correction rather than hiding it.

## 5. Transferable artifacts (available on request)

- Bilingual ontological lexicon — 52 English terms decomposed phonetically and semantically, with the Slovenian-side operator system
- Verified etymology chains with sources
- Register of "hybrid word" failures — model-invented words that pass as real (three recorded falls; a fourth's absence verified by measurement)
- Provenance-marked corpus: every claim carries ⊢ or ⚠

## 6. Precedents: the compensation model already exists

| Field | Mechanism | Evidence |
|---|---|---|
| Security research | Bug bounties: verifiable vulnerability → paid; ~$81M paid in 12 months across HackerOne programs | HackerOne 2025 report |
| Generative images | Shutterstock Contributor Fund: artists compensated for content's role in AI training, plus royalties | Shutterstock investor releases |
| Low-resource language data | Karya (India): fair wages for dataset work in 22 Indian languages | karya.in |
| Attribution-only | Masakhane: credited co-authorship on 30+ African translation benchmarks | EMNLP Findings 2020 |
| Theory | "Data dignity" / "data as labor" — Lanier & Weyl (2018) | Harvard Business Review |

The asymmetry it corrects, documented: Reddit received ~$60M/year from Google for training data; the writers received nothing. Stack Overflow sold contributor answers to OpenAI; contributors who deleted their own answers to opt out were banned. Mozilla Common Voice: 250+ languages of volunteer recordings, CC0, contributors unpaid. RLHF annotation is already a paid profession ($12–25/hr) — the money exists; the channel for native experts offering *verified* corrections does not. The missing row is a verified language contribution from a native expert in an unsupported language. This document proposes closing exactly that gap.

## 7. The national proof

- **GaMS** (Generative Model for Slovene): open models at 2B, 9B, 27B, University of Ljubljana, national PoVeJMo program, openly released. ⊢
- **GaMS3-12B** outperforms Gemma 3 12B on all Slovenian evaluation scenarios and wins over 60% against GPT-4o in the Slovene LLM Arena (arXiv 2603.01691). ⊢
- Training data from a **national citizen-science campaign** (povejmo.si) toward a 40-billion-word corpus, with the National and University Library, Dnevnik, and the Slovenian Press Agency — framed as digital sovereignty for a two-million-speaker language. ⊢

What the national effort solved is *volume*. What it does not yet solve is *verification*. My corpus is the provenance layer the contributed-data model is missing. Volume + provenance turns contributed language data from raw material into a priced, auditable contribution.

The honest counter-argument (arXiv 2504.12427): upfront payment for training data is economically infeasible at scale. Answerable: infeasible for *undifferentiated bulk data*; not infeasible for *verifiable contributions* — sourced corrections, validated lexicons — which are scarce, attributable, exactly the bug-bounty category. The ⊢/⚠ regime is the mechanism that distinguishes the two. The verification framework and the compensation channel are one system.

## 8. The vendor's own receipts

1. The "strong performance" list is a tier, not a boundary: the vendor's own documentation states models "can also perform well in additional languages not listed here." ⊢
2. Fine-tuning with custom vocabulary and domain adaptation is already a shipped commercial feature — the request is to productize what exists at a per-user, session-scoped scale. ⊢
3. A 12B model fine-tuned on ~20,000 translation pairs measurably improves in a specialized language task (arXiv 2312.12740). ⊢

Sovereignty context: OpenEuroLLM is building multilingual foundation models for all EU official languages as sovereign AI; Slovenia has funded its own national program, model, and a planned free public platform. A European vendor answering a European language with "not among our strong-performance languages" is competing, from behind, with public infrastructure that did not wait.

## 9. Proposed adoption path

1. **Pilot:** ⊢/⚠ claim-marking as an output convention in a low-resource language (Slovenian: available, pre-verified, motivated).
2. **Audit trails:** scarred corrections in place of silent rewrites — visible, named, queryable.
3. **Contribution channel:** credit and compensation for adopted, verifiable language contributions.

**Notebook:** the demo notebook (separate body, `notebook.md` in the repository) ships the technique: lexicon as session contract, three mechanical detection layers, append-only fall register. Its one open cell awaits an executed run — by L4, the run will be reported when it is performed, not before. The two falls in §4 are the register's current proof.

## 10. Version, license, terms

**Verification-Before-Assertion — v1.1 · September 30, 2026 · Luka Pintarič.** The timestamped public record precedes any adoption, acknowledged or silent.

- **Methodology — open.** The five laws, the AI-failure mapping, the notebook technique, the register schema, the adoption path: **CC BY 4.0**, attribution required.
- **Corpus — closed.** The ~2,600-document corpus, the full bilingual lexicon, the etymology chains, and the complete fall register remain the contributor's licensed assets. Entries appearing in this document are cited quotations. Terms for the corpus are exactly one: the compensated-contribution model of §6–§7.

**Rejection clause.** If declined: no obligation on either side; the contribution remains the contributor's to submit elsewhere (University of Ljubljana / PoVeJMo, OpenEuroLLM, or any program); the timestamped public record remains. Decline is a closed door, recorded as a routing event.

**Integrity of the marks.** ⚠ words stay marked in every derived artifact. A derived version that silently verifies a ⚠ word falsifies the regime and does not represent it.

**Contact:** Luka Pintarič — demonstration performed before it is reported (L4).
