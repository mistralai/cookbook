# Green coding with Mistral Vibe: audit your code against the RGESN referential

Authors: [Institut du Numérique Responsable](https://institutnr.org) (GitHub: [@Institut-du-Numerique-Responsable](https://github.com/Institut-du-Numerique-Responsable))

Run a digital-sobriety audit of a codebase while you build with Mistral Vibe, using [Green Vibe](https://github.com/Institut-du-Numerique-Responsable/green-vibe) — a Numérique Responsable plugin that steers the Vibe agent (running Mistral models) towards eco-designed code, checks its own output deterministically, and teaches the rules it applied: from selecting only the applicable rules to a normalized, evidence-based bilan.

> The Green Vibe skill registry / marketplace referencing is under discussion in [mistral-vibe#1279](https://github.com/mistralai/mistral-vibe/issues/1279). This cookbook uses the plugin's local installation, which works today on any current Mistral Vibe install.

## Prerequisites

To complete this cookbook, you will need:

- Mistral Vibe installed ([installation](https://docs.mistral.ai/vibe/code/overview))
- `python3 >= 3.10` on your PATH
- A codebase to audit (the examples below use Green Vibe's own test fixtures, then your project)
- No Mistral API key required for this tutorial: the validators are local and deterministic

## Environment setup

Everything needed to run the audit is installed in two steps: the plugin itself, then a Vibe reload.

### Install

Clone the Green Vibe repository and run its installer, which copies the plugin into Vibe's user plugin directory:

```bash
git clone https://github.com/Institut-du-Numerique-Responsable/green-vibe.git
cd green-vibe
./scripts/install.sh          # copies the plugin to ~/.vibe/plugins/green-vibe
```

In Vibe, run `/reload-plugins` and confirm in `/plugins` that `green-vibe` is listed. The skill is now available as `/green-vibe` and is loaded automatically when you work on code.

### Required environment variables

None. The plugin performs no network calls and reads only the files you point it at.

## Step 1: Select the rules that apply to your project

Green Vibe holds 149 rules (13 cross-cutting families, 6 languages, 16 agent-usage practices) but never loads them all. The `select` command detects the context (languages, project markers) and returns only the relevant rule set:

```bash
GV="$HOME/.vibe/plugins/green-vibe"
python3 "$GV/validators/green-vibe" select src/ | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['families'], d['count'], 'règles')"
```

Run it on Green Vibe's own fixture project to see the mechanism:

```bash
python3 "$GV/validators/green-vibe" select "$GV/tests/fixtures"
```

## Step 2: Run the deterministic audit

Scan the codebase and collect the findings as JSON with the audit command:

```bash
python3 "$GV/validators/green-vibe" audit src/ > /tmp/gv-audit.json
```

Findings come in two kinds: `FAIL` (deterministic detector, e.g. `SELECT *` in SQL) and `REVIEW` (candidate needing context, e.g. an HTTP call inside a loop). Each finding carries the rule id, file:line, a note describing the detector's limits, and the recommended fix. Strings and comments are masked before matching, so a pattern quoted in a literal is not flagged.

## Step 3: Produce the evidence-based report

Turn the audit findings into a normalized report, declaring the rules whose checks actually ran:

```bash
python3 "$GV/validators/green-vibe" report \
  --audit /tmp/gv-audit.json \
  --verified "NR-DATA-004=tests/test_pagination.py::test_last_page" \
  --intervention "Cursor pagination on /orders" \
  --mode pedagogique
```

The key discipline: a rule is `verified` only when its own check actually ran and passed (declared via `--verified RULE=EVIDENCE`). Model inspection alone yields `not_tested`, never `verified` — and a confirmed violation can never be overridden by a claim.

## Step 4: Read the pedagogical bilan

The report ends with a French summary (modes: `discret`, `pédagogique`, `audit`) listing the rules applied, their verification status, the points left to examine, and one or two transferable takeaways — all grounded in the rules that were really selected and checked, never a generic best-practice list. Technical gains (fewer bytes, fewer queries) are reported as technical claims; no environmental (energy/CO2) figure is ever derived from a proxy.

## Step 5: Let the agent apply the rules while coding

The plugin also ships a fail-open `post_tool` hook: after every `write_file`/`edit`, Vibe runs a diff-scoped audit and returns candidates to the agent in the same turn, so the audit happens even when nobody thinks of it. In a session, just ask for work on a file and read the one-line bilan the skill produces at the end.

## Clean up

Remove the plugin and the audit artefact when you are done experimenting:

```bash
rm -rf ~/.vibe/plugins/green-vibe   # remove the plugin
rm /tmp/gv-audit.json               # remove the audit artefact
```

Then `/reload-plugins` in Vibe.

## Summary

This cookbook showed how to install the Green Vibe plugin for Mistral Vibe and run the full Numérique Responsable pipeline — context-driven rule selection, deterministic static audit, and an evidence-based normalized report — on any codebase, without any API key.

**What this cookbook covers:**

- Installing the Green Vibe plugin (Agent Plugins 1.0) into Mistral Vibe
- Selecting only the applicable rules for a project context (`select`)
- Running a deterministic sobriety audit with masked literals and per-finding limits (`audit`)
- Producing a normalized report where `verified` requires executed evidence (`report`)
- Reading the pedagogical bilan and the automatic post-edit hook

**Mistral features used:**

- Mistral Vibe (CLI coding agent)
- Agent Plugins 1.0 (skills, hooks, knowledge)
- Vibe `post_tool` hooks (fail-open, diff-scoped audit)
- User-invocable skills (`/green-vibe`) and `/reload-plugins`

Green Vibe is maintained by the Institut du Numérique Responsable (co-author of the GR491, co-pilot of the RGESN mission) under Apache-2.0 / CC BY 4.0; its rule catalog records provenance and the related RGESN criterion for every rule.

Learn more about Vibe plugins and skills in the [Mistral Vibe documentation](https://docs.mistral.ai/vibe/code/overview).
