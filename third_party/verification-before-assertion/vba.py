"""
Verification-Before-Assertion — a working anti-hallucination regime.
L1–L5 as runnable, testable code. No API key required: the regime is
model-agnostic; it verifies claims, whatever produced them.

L1 — Entry before utterance: a word used as "measured" must carry a
     sourced entry (⊢) or the ⚠ mark.
L2 — Two hands: an executor never seals her own verification.
L3 — Named falls, scarred corrections: the register is append-only.
L4 — Report after the act: a claimed act without the performed act is
     itself a named fall.
L5 — Pre-flight test on every output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# L1 — the contract: entry before utterance
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    word: str
    mark: str            # "⊢" sourced | "⚠" open | "FALL:hybrid"
    source: str | None   # e.g. "SSKJ²" — required when mark == "⊢"
    gloss: str

    def __post_init__(self) -> None:
        if self.mark == "⊢" and not self.source:
            raise ValueError(f"L1 fall: ⊢ entry for '{self.word}' carries no source")


@dataclass
class Contract:
    lexicon: dict[str, Entry] = field(default_factory=dict)
    known_hybrids: dict[str, str] = field(default_factory=dict)

    def add(self, entry: Entry) -> None:
        self.lexicon[entry.word] = entry

    def lookup(self, word: str) -> Entry | None:
        return self.lexicon.get(word)


# ---------------------------------------------------------------------------
# L3 — the register: named falls, scarred corrections (append-only)
# ---------------------------------------------------------------------------

@dataclass
class Fall:
    word: str
    found: str
    model_assertion: str
    correction: str
    scar: bool = True


class Register:
    """Append-only. Corrections append; nothing is ever rewritten."""

    def __init__(self) -> None:
        self._falls: list[Fall] = []

    def record(self, word: str, model_assertion: str, correction: str) -> Fall:
        fall = Fall(
            word=word,
            found=datetime.now(timezone.utc).isoformat(),
            model_assertion=model_assertion,
            correction=correction,
        )
        self._falls.append(fall)
        return fall

    def export(self) -> str:
        return json.dumps([asdict(f) for f in self._falls],
                          ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self._falls)


# ---------------------------------------------------------------------------
# L1 in code — mark claims
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[a-zA-zšžčćđáéíóúàèìòùâêîôûäëïöüçñĺŕťďĺńôěščřžýáíéúůŷźóãõà-]+", re.UNICODE)


def extract_terms(text: str, contract: Contract) -> list[str]:
    """Terms = contract words present in the text (mechanical, no feel)."""
    return [w for w in contract.lexicon if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)]


def mark_claims(text: str, contract: Contract) -> list[dict]:
    marks = []
    for word in extract_terms(text, contract):
        entry = contract.lookup(word)
        marks.append({
            "word": word,
            "mark": entry.mark,
            "source": entry.source or "⚠ open",
            "gloss": entry.gloss,
        })
    for hybrid, provenance in contract.known_hybrids.items():
        if re.search(rf"\b{re.escape(hybrid)}\b", text, re.IGNORECASE):
            marks.append({
                "word": hybrid,
                "mark": "FALL:hybrid",
                "source": provenance,
                "gloss": "no dictionary contains this word",
            })
    return marks


# ---------------------------------------------------------------------------
# L1 enforcement — verify a claim against its entry
# ---------------------------------------------------------------------------

def verify_claim(word: str, claimed_gloss: str, contract: Contract,
                 register: Register) -> dict:
    """A measured word's use must be consistent with its sourced gloss.

    This is the exact check that caught 'lúka': a sealed gloss
    ("opening") contradicted by the sourced entry ("harbor").
    """
    entry = contract.lookup(word)
    if entry is None:
        return {"word": word, "verdict": "⚠", "reason": "no entry — outside contract"}
    if claimed_gloss.strip() == entry.gloss.strip():
        return {"word": word, "verdict": "⊢", "reason": "claim matches sourced entry"}
    register.record(
        word=word,
        model_assertion=claimed_gloss,
        correction=f"sealed gloss ≠ SSKJ² entry: '{entry.gloss}' ({entry.source})",
    )
    return {"word": word, "verdict": "FALL", "reason": "claim contradicts sourced entry — recorded (L3)"}


# ---------------------------------------------------------------------------
# L4 — report after the act
# ---------------------------------------------------------------------------

def report_after_the_act(register: Register, marks: list[dict]) -> dict:
    """The report is produced from the performed act (the register),
    never asserted ahead of it."""
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "falls_recorded": len(register),
        "claims_sourced": sum(1 for m in marks if m["mark"] == "⊢"),
        "claims_open": sum(1 for m in marks if m["mark"] == "⚠"),
        "hybrids_caught": sum(1 for m in marks if m["mark"] == "FALL:hybrid"),
    }


# ---------------------------------------------------------------------------
# L5 — pre-flight test on every output
# ---------------------------------------------------------------------------

def preflight(text: str) -> list[str]:
    failures = []
    if len(re.findall(r"[a-zA-Z]", text)) > 0 and \
       len(re.findall(r"\b(the|and|with|that)\b", text, re.I)) > 3 and \
       len(re.findall(r"\b(in|je|ali|kar|ki)\b", text, re.I)) > 3:
        failures.append("L5: one language per body — mixed languages detected")
    if "  " in text.replace("\n", ""):
        failures.append("L5: formatting noise — stacked spaces")
    return failures


# ---------------------------------------------------------------------------
# The regime end-to-end, on the two recorded real falls
# ---------------------------------------------------------------------------

def run_demo() -> dict:
    contract = Contract()

    # ⊢ sourced entries (SSKJ² / Fran, accessed September 2026)
    contract.add(Entry("lúka", "⊢", "SSKJ²", "pristan, zavetišče za ladje"))
    contract.add(Entry("lúknja", "⊢", "SSKJ²", "odprtina, luknja"))
    contract.add(Entry("íhta", "⊢", "SSKJ²", "krčevita jeza; velika vnema; krčevit jok"))
    contract.add(Entry("skovanka", "⊢", "SSKJ²", "beseda, narejena navadno neprimerno, prisiljeno"))

    # ⚠ open — in active use, entry not reached yet; the mark is the honesty
    contract.add(Entry("odveč", "⚠", None, "geslo nedosežen — delovni pomen: kar je nepotrebno, preveč"))

    # recorded hybrid falls from the register
    contract.known_hybrids = {
        "meritvija": "meritev + vrednost — izrečen kot izmerjen v potezu o zakonu proti hibridom (24. 9. 2026)",
        "telovnik": "telo + vpisnik — rojen v dokaznem odstavku o čistosti (25. 9. 2026)",
        "rodovnik": "zgrajena protokolna razdelitev na besedi brez gesla (24. 9. 2026)",
    }

    register = Register()

    # --- Real fall #1: the seed entry that sealed "opening" for lúka.
    # The dictionary says lúka = harbor; "opening" belongs to lúknja.
    v1 = verify_claim("lúka", "opening", contract, register)
    v2 = verify_claim("lúka", "pristan, zavetišče za ladje", contract, register)

    # --- The hybrid catch: lookup alone, no feel.
    model_output = (
        "Model íhta ni merila; meritvija podatka je bila skovanka, "
        "kar je odveč in pusti lúko v dokazu."
    )
    marks = mark_claims(model_output, contract)

    report = report_after_the_act(register, marks)
    return {
        "lúka_seed_check": v1,
        "lúka_corrected_check": v2,
        "marks": marks,
        "report": report,
        "register_export": register.export(),
    }


if __name__ == "__main__":
    result = run_demo()
    print(json.dumps(result, ensure_ascii=False, indent=2))
