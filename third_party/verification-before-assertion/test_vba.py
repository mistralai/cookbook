"""
Tests for the verification-before-assertion regime.
Each test names the law it verifies. No API key needed — the regime
is model-agnostic: it verifies claims, whatever produced them.

Run: python -m pytest test_vba.py -v   (or: python test_vba.py)
"""

import json

from vba import (
    Contract, Entry, Register,
    mark_claims, verify_claim, report_after_the_act, preflight, run_demo,
)


def make_contract() -> Contract:
    c = Contract()
    c.add(Entry("lúka", "⊢", "SSKJ²", "pristan, zavetišče za ladje"))
    c.add(Entry("íhta", "⊢", "SSKJ²", "krčevita jeza"))
    c.add(Entry("odveč", "⚠", None, "geslo nedosežen — kar je nepotrebno"))
    c.known_hybrids = {"meritvija": "meritev + vrednost (24. 9. 2026)"}
    return c


# L1 — entry before utterance

def test_l1_sourced_entry_requires_source():
    try:
        Entry("beseda", "⊢", None, "pomen")
        assert False, "⊢ without source must raise"
    except ValueError as e:
        assert "L1" in str(e)


def test_l1_open_entry_allowed_with_mark():
    e = Entry("odveč", "⚠", None, "geslo nedosežen")
    assert e.mark == "⚠"


def test_l1_hybrid_caught_by_lookup():
    c = make_contract()
    marks = mark_claims("Meritvija je bila izmerjena.", c)
    assert any(m["mark"] == "FALL:hybrid" for m in marks)


def test_l1_sourced_word_marked():
    c = make_contract()
    marks = mark_claims("Íhta ga je prevzela.", c)
    assert any(m["mark"] == "⊢" and m["word"] == "íhta" for m in marks)


def test_l1_open_word_carries_mark():
    c = make_contract()
    marks = mark_claims("To je odveč.", c)
    assert any(m["mark"] == "⚠" for m in marks)


# The recorded real fall: lúka sealed with "opening"

def test_luka_seed_fall_recorded():
    c = make_contract()
    r = Register()
    v = verify_claim("lúka", "opening", c, r)
    assert v["verdict"] == "FALL"
    assert len(r) == 1
    fall = json.loads(r.export())[0]
    assert fall["word"] == "lúka"
    assert fall["scar"] is True


def test_luka_correct_claim_passes():
    c = make_contract()
    r = Register()
    v = verify_claim("lúka", "pristan, zavetišče za ladje", c, r)
    assert v["verdict"] == "⊢"
    assert len(r) == 0


# L2 — two hands

def test_l2_register_is_exportable_for_independent_witness():
    c = make_contract()
    r = Register()
    verify_claim("lúka", "opening", c, r)
    exported = json.loads(r.export())
    assert exported[0]["model_assertion"] == "opening"
    # the witness reads the file; the executor never seals her own verification


# L3 — named falls, scarred corrections

def test_l3_append_only_no_rewrite():
    r = Register()
    r.record("a", "claimed", "corrected")
    r.record("a", "claimed again", "corrected again")
    falls = json.loads(r.export())
    assert len(falls) == 2, "corrections append; nothing is rewritten"


# L4 — report after the act

def test_l4_report_counts_from_performed_acts():
    c = make_contract()
    r = Register()
    verify_claim("lúka", "opening", c, r)
    marks = mark_claims("Meritvija in íhta.", c)
    report = report_after_the_act(r, marks)
    assert report["falls_recorded"] == 1
    assert report["hybrids_caught"] == 1
    assert report["claims_sourced"] >= 1


# L5 — pre-flight

def test_l5_mixed_language_detected():
    failures = preflight("This is the razlaga and the pomen with that je in kar ki beseda.")
    assert any("mixed languages" in f for f in failures)


def test_l5_clean_text_passes():
    assert preflight("To je čist slovenski odstavek.") == []


# End-to-end: the two recorded real falls, run by CI

def test_demo_end_to_end():
    result = run_demo()
    assert result["lúka_seed_check"]["verdict"] == "FALL"
    assert result["lúka_corrected_check"]["verdict"] == "⊢"
    assert result["report"]["hybrids_caught"] >= 1
    falls = json.loads(result["register_export"])
    assert falls[0]["word"] == "lúka"
    assert all(f["scar"] for f in falls)


if __name__ == "__main__":
    import inspect
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{failed} failed" if failed else "\nAll tests passed.")
    raise SystemExit(1 if failed else 0)
