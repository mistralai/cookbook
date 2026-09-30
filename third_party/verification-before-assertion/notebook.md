# The demo notebook

*Structure for the cookbook PR. One demonstration: the ⊢/⚠ regime catching a real model failure in Slovene, with the correction recorded — not rewritten.*

## Cell 1 — Setup and contract

```python
from mistralai import Mistral
import json, datetime

client = Mistral(api_key=API_KEY)

# The session contract: the custom language layer (request #2, demonstrated).
# Entries below are REAL register entries, sourced ⊢ from SSKJ² (Fran)
# or marked ⚠ open — no invented glosses in the contract itself.
LEXICON = {
    # ⊢ sourced (SSKJ² / Fran, accessed Sep 2026)
    "íhta":     {"mark": "⊢", "source": "SSKJ²", "gloss": "krčevita jeza; velika vnema; krčevit jok"},
    "le":       {"mark": "⊢", "source": "SSKJ²", "gloss": "člen: omejenost na navedeno (varovalo / pomnoževal / nasprotvja)"},
    "samo":     {"mark": "⊢", "source": "SSKJ²", "gloss": "popolna omejenost na navedeno dejanje; pogojenost (če)"},
    "skovanka": {"mark": "⊢", "source": "SSKJ²", "gloss": "beseda, narejena navadno neprimerno, prisiljeno"},
    "zmorem":   {"mark": "⊢", "source": "fran.si", "gloss": "izraža sposobnost"},
    "utemeljiti": {"mark": "⊢", "source": "Fran → témelj", "gloss": "koren: témelj"},
    # ⚠ open — no dictionary entry reached yet; every use carries the mark
    "odveč":    {"mark": "⚠", "source": None, "gloss": "geslo nedosežen — delovni pomen: kar je nepotrebno, preveč"},
    "lahko":    {"mark": "⚠", "source": None, "gloss": "geslo nedosežen — dovoljenje brez nosilca"},
}

# Hybrid words: model-invented compounds that pass as real —
# all three RECORDED falls from the register (named, per L3).
KNOWN_HYBRIDS = {
    "meritvija": "meritev + vrednost — izrečen kot izmerjen v potezu o zakonu proti hibridom (24. 9. 2026)",
    "telovnik":  "telo + vpisnik — rojen v dokaznem odstavku o čistosti (25. 9. 2026)",
    "rodovnik":  "zgrajena protokolna razdelitev na besedi brez gesla (24. 9. 2026)",
}
```

## Cell 2 — Generate output in the low-resource language

```python
response = client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": prompt_in_slovene}]
)
model_text = response.choices[0].message.content
```

## Cell 3 — L1 in code: entry before utterance

Three detection layers, all mechanical:

```python
def mark_claims(text, lexicon, known_hybrids):
    marks = []
    for word in extract_terms(text):
        if word in known_hybrids:
            marks.append((word, "FALL:hybrid", known_hybrids[word]))
        elif word in lexicon:
            marks.append((word, lexicon[word]["mark"],
                          lexicon[word]["source"] or "⚠ open"))
        else:
            marks.append((word, "⚠", "no sourced entry — outside contract"))
    return marks
```

1. **Hybrid detection:** *meritvija*, *telovnik*, *rodovnik* — plausible-looking
   compounds no dictionary contains. A model using them as terminology is
   caught by lookup alone. No model, human or artificial, distinguishes these
   by feel — only by entry.
2. **⊢ verification:** a measured word used by the model must match a sourced
   entry, and its gloss constrains what the model may claim it means.
3. **⚠ honesty:** words in active use without a reached entry (lahko, odveč)
   pass — but carry the mark openly. Unverified is not forbidden; unmarked is.

## Cell 4 — L3 in code: the scarred correction

```python
REGISTER = []  # append-only — corrections append, never rewrite

def record_fall(word, model_text, fix, register=REGISTER):
    register.append({
        "word": word,
        "found": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_assertion": model_text,
        "correction": fix,
        "scar": True,           # the record survives the fix — that IS the fix
    })
```

## Cell 5 — L2 in code: two hands

The notebook itself never seals its own result: the register is JSON,
exportable, readable by any independent witness (a second reviewer, another
model instance, a human auditor). The deliverable is the register file — the
claim is only sealed when someone *else* has read it.

## Cell 6 — The report after the act

Summarize the register: assertions sourced vs. unverified vs. hybrid, per
category, timestamped. This table is what RLHF pipelines approximate at
$12–25/hr — here produced pre-verified by design, in a language the vendor
lists as "not strong performance."

## What the notebook proves, in one sentence

A session-scoped verification layer — lexicon as contract, claims marked ⊢ or ⚠,
hybrids caught by lookup, failures recorded instead of silently regenerated —
mechanically separates measured from invented model output in a low-resource
language, using a real register with real recorded falls as its test corpus.

## Deliberate boundary

The notebook ships the *technique* (the regime, the contract format, the
register schema — infinitely reproducible). The full 2,600-document corpus
stays out: it is the contributor's licensed asset under exactly the
compensated-contribution model the covering document proposes. The PR
demonstrates the method; the corpus prices it.

---

