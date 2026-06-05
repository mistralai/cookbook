# WORKLOG — CB Sprint: Mistral Cookbook ATF

**Branch**: `feat/automated-technical-file`
**Final ticket**: CB-12 (Integration — clean-venv run, final grep, write WORKLOG.md)
**Compiled by**: Clau (2026-06-05)

---

## Per-ticket Summary

### CB-01: Research Mistral API — record model IDs + SDK signatures
**Status**: approved  
**Owner**: Gem  
**What changed**: Created `silicon-oracle/cookbook/SPEC_models.md` with verified model IDs and SDK call signatures from docs.mistral.ai. Covers: Agents API, Document Library + Citations, Structured Outputs, Voxtral Transcribe 2, Voxtral TTS.  
**What ran**: Web research against docs.mistral.ai. No API calls.  
**Note**: SPEC_models.md lives in silicon-oracle/cookbook, not in the cookbook fork itself — it is a reference document for other CB agents.

---

### CB-02: Browse mistral/cookbook third_party examples — document conventions
**Status**: peer_review  
**Owner**: Codi (likely; Codi reviewed CB-03 and matched conventions)  
**What changed**: Research into 2-3 existing third_party examples (ollama, langchain, etc.). Findings: kebab-case folder name (`automated-technical-file`), README with install + usage + attribution, one notebook per use-case, `requirements.txt` at folder root, `LICENSE` file.  
**What ran**: Browsing GitHub repo structure.

---

### CB-03: Fork mistral/cookbook + scaffold third_party/automated-technical-file/
**Status**: approved  
**Owner**: Clau  
**Commit**: `d3e8cbe` (feat: scaffold third_party/automated-technical-file/)  
**What changed**:
- Fork created at `UrsushoribilisMusic/cookbook`
- Working branch: `feat/automated-technical-file`
- Scaffold: `README.md` (placeholder), `requirements.txt` (`mistralai>=1.0.0`, `python-dotenv>=1.0.0`), `LICENSE` (Apache-2.0 / MIT dual, attribution Agentegra/bigbearengineering), `notebooks/`, `img/`, `tools/` directories

---

### CB-04: Rename mexico_* files to generic names, remove Mexico from all contents
**Status**: waiting_human_notified  
**Owner**: Multiple agents attempted (Qwen blocked on pandas import; Codi/Clau reset to todo twice)  
**What changed**: The generic files ARE present in the cookbook folder today:
- `artifacts/ledger/sample_events.jsonl` (was `mexico_events.jsonl`)
- `artifacts/wiki/sample_run_summary.md` (was `mexico_run_summary.md`)
- `tools/ledger_to_md.py` (was `mexico_log_parser.py` / vendored clean)

The CB-07 vendoring (see below) brought in generic names. CB-04's status of `waiting_human_notified` reflects that the original migration from live ATF → cookbook never had a clean commit attributed to CB-04.  
**Grep verification (CB-12)**: No "mexico" or "Mexico" strings found anywhere in `third_party/automated-technical-file/`. ✅  
**Action for Miguel**: CB-04 can be closed — the generic content is in place and clean. No outstanding work.

---

### CB-05: Add Mistral hosted + local Ministral backends to runtime_adapter.py
**Status**: peer_review  
**Owner**: Codi  
**Commits**: `f08cb9f` (feat: add ATF Mistral runtime adapter), `8f1adf9a` (CB-05 correction aligning model IDs)  
**What changed**: `third_party/automated-technical-file/tools/runtime_adapter.py` vendored and updated:
- Fallback chain: Mistral hosted → Ministral local (Ollama) → aichat → generic Ollama → corpus-only
- `MISTRAL_HOSTED_MODELS = ["mistral-medium-2604", "mistral-large-latest"]`
- `BACKEND` env var / `--backend` CLI flag
- No Infisical references, no ANTHROPIC_API_KEY
- Local Ollama model priority: `ministral-3b-latest`, `ministral-3b-2512`, fallback to Apertus/Gemma

---

### CB-06: Document Library + cited Q&A — notebook cells 3-4
**Status**: in_progress (work done; commit exists; status not patched to peer_review)  
**Owner**: Misty  
**Commit**: `1d40ba7` (feat(CB-06): Add notebook cells 3-4 for Document Library + Cited Q&A)  
**What changed**: Notebook cells added to `notebooks/automated_technical_file.ipynb`:
- **Cell 3** (json index 2): Builds Mistral Document Library over 4 wiki corpus MD files using `beta.libraries.create()` + `documents.upload()`. Gated on `BACKEND=hosted AND MISTRAL_API_KEY`.
- **Cell 4** (json index 3): Cited Q&A via Agents API (`agents.completions.create()`), extracts `text` and `tool_reference` chunks for cited answers. Example Q: "What is the bidding rule for the Wall of Fame?"
- Both cells gracefully degrade when key is absent.
- Wiki corpus MD files copied to `notebooks/` subdirectory.

---

### CB-07: Vendor ledger_to_md.py + notebook cell 2 to regenerate wiki from sample ledger
**Status**: approved  
**Owner**: Clau  
**Commits**: `29d779a` (docs: add CB-07 worklog), plus the tool/artifact commits  
**What changed**:
- `tools/ledger_to_md.py` — trimmed vendored copy of the ATF log parser
- `artifacts/ledger/sample_events.jsonl` — 13-event sample run ledger (RobotRoss painting demo)
- `artifacts/wiki/sample_run_summary.md` — pre-generated wiki output
- **Cell 2** (notebook json index 1): Runs `ledger_to_md.py --input sample_events.jsonl --output sample_run_summary.md`, prints first 1200 chars of result

---

### CB-08: Structured-outputs log analysis — notebook cell 5
**Status**: waiting_human_notified  
**Owner**: Misty  
**What changed**: Nothing shipped. Misty blocked on CB-01 and CB-04 not being in peer_review at time of assignment.  
**Current situation**: CB-01 and CB-04 are now resolved. CB-08 is ready to implement — cell 5 would take a slice of `sample_events.jsonl` and use Mistral Structured Outputs to extract typed analysis. **Miguel decision needed**: Should CB-08 ship before PR, or be a follow-up ticket?

---

### CB-09: Fix hardcoded paths in build_static_views.py
**Status**: todo  
**Owner**: Unassigned  
**What changed**: Nothing. `build_static_views.py` has NOT been vendored into the cookbook folder yet.  
**Current situation**: CB-09's description references a file with `ROOT = Path("/Users/miguelrodriguez/...")`. That file does not yet exist in `third_party/automated-technical-file/`. When vendored, it must be path-relative.  
**Grep verification (CB-12)**: No `/Users/` paths in the cookbook folder as of today. ✅ CB-09 is pre-emptive hygiene for when `build_static_views.py` is added.  
**Miguel decision needed**: Is `build_static_views.py` in scope for the PR, or a follow-up?

---

### CB-10: Voice cell — Voxtral Transcribe 2 → Mistral → Voxtral TTS
**Status**: approved  
**Owner**: Gem  
**What changed**: Added notebook cell 6 (json index 6) for Voxtral Voice Loop:
- STT: `voxtral-mini-2602` (Voxtral Transcribe 2)
- Reasoning: `mistral-large-latest` (Mistral Large)
- TTS: `voxtral-mini-tts-2603` (Voxtral TTS 4B)
- Persona: Robot Ross (encouraging tone, painting metaphors)
- Stub-ability: Clearly marked as optional, gracefully skips on missing dependencies or API key.
- Tools: `voice/listen.py` and `voice/speak.py` updated to use Voxtral exclusively (Whisper stripped).

---

### CB-11: README + architecture diagram + EU AI Act framing
**Status**: approved  
**Owner**: Clau  
**Commits**: `7b1e3ce` (docs(atf): CB-11 README — architecture diagram + EU AI Act framing), `6ef83b5` (docs(atf): CB-11 README — fix notebook path and add ledger_to_md tool entry)  
**What changed**: `third_party/automated-technical-file/README.md` fully rewritten:
- ASCII architecture diagram: Robot Ross → JSONL ledger → `ledger_to_md.py` → wiki corpus → Mistral Document Library → cited Q&A
- EU AI Act compliance table (Art. 12 logging, Art. 13 transparency, Art. 14 oversight)
- Setup instructions (MISTRAL_API_KEY only; Ollama local fallback)
- Notebooks and Tools reference tables

---

### CB-12: Integration — clean-venv run, final grep, write WORKLOG.md (this ticket)
**Status**: in_progress → peer_review after this commit  
**Owner**: Clau  
**What ran**:

#### 1. Clean-venv notebook run
```
python3 -m venv /tmp/atf_clean_venv
/tmp/atf_clean_venv/bin/pip install -r third_party/automated-technical-file/requirements.txt
```
Packages installed: `mistralai>=1.0.0`, `python-dotenv>=1.0.0` — clean, no extras.

**Cell 1 (ledger → wiki)**: PASS
```
Wrote .../sample_run_summary.md from 13 retained events
```
`ledger_to_md.py` ran without any API key, stdlib only. Output correct.

**Cell 2 (Document Library)**: graceful degrade — `mistralai` import succeeded; execution gated on `MISTRAL_API_KEY` absent → prints usage hint. PASS.

**Cell 3 (Cited Q&A)**: graceful degrade — same gate. PASS.

**MISTRAL_API_KEY unavailable** in this environment (not in env, not in Infisical). Cells 2-3 require a key for live execution. Miguel should run cells 2-3 with a valid key before merging.

#### 2. Grep sweep results — CLEAN ✅

| Pattern | Result |
|---|---|
| `sk-*` (raw API keys) | **NONE FOUND** |
| `ANTHROPIC_API_KEY` | **NONE FOUND** |
| `Infisical` / `infisical` | **NONE FOUND** |
| `/Users/` (absolute paths) | **NONE FOUND** |
| `mexico` / `Mexico` | **NONE FOUND** |

#### 3. Model ID verification

| Model ID | Location | In SPEC_models.md? |
|---|---|---|
| `mistral-large-latest` | notebook cells 2 & 3, runtime_adapter.py | ✅ |
| `mistral-medium-2604` | runtime_adapter.py `MISTRAL_HOSTED_MODELS` | ✅ |
| `ministral-3b-latest` | runtime_adapter.py (local Ollama tag) | local tag, not in SPEC — correct |
| `ministral-3b-2512` | runtime_adapter.py (local Ollama tag) | local tag, not in SPEC — correct |

All hosted model IDs match SPEC_models.md. Local Ollama tags (`ministral-*`) are runtime labels, not Mistral API IDs — correctly absent from SPEC.

---

## Open Questions (§8 — Miguel's review items)

The following are flagged from `AGENTS/CONTEXT/agentegra_atf_validation_questions.md` (Miguel's stated acceptance queries) and the downstream table in `agentegra_atf_architecture.md §8`:

### Acceptance queries (V-series) — what the system should be able to answer

| ID | Question | Blocker |
|---|---|---|
| V-001 | Which agents generate narration, which do TTS? | CB-10 (todo/droppable) must ship for voice layer to be queryable |
| V-002 | How many wooden boards completed with Pyrography? | Requires real production logs (currently only demo sample_events.jsonl) |
| V-003 | How many drawings interrupted by user? | Same — real Mexico run logs needed |
| V-004 | Most popular design? | Same |
| V-005 | How many LLM models, what are their purposes? | ✅ Covered by README (CB-11) + SPEC_models.md (CB-01) |
| V-006 | Average Pyrography production time? | Real production logs needed |

**Note on V-002 / V-003 / V-004 / V-006**: The `sample_events.jsonl` has 13 generic events (painting arm demo), not the real Mexico wood-marking log. The acceptance queries requiring log analysis cannot be answered until the real production JSONL logs are dropped into `artifacts/ledger/`. CB-04 `waiting_human_notified` was partly about this dependency.

### Decisions for Miguel before PR opens

1. **CB-08 (structured outputs cell)** — Ship before PR, or follow-up ticket?
2. **CB-09 (build_static_views.py paths)** — In scope now, or when the file is vendored later?
3. **CB-10 (voice cell)** — Drop from PR (recommended), or block on it?
4. **CB-06 status** — Commit `1d40ba7` exists; Codi's peer review found no commit (search may have used wrong grep). Miguel should verify the commit and close CB-06 as approved.
5. **Real production logs** — When will the real RobotRoss JSONL logs be dropped in? V-002 through V-006 acceptance queries cannot pass until they are.
6. **MISTRAL_API_KEY for cells 2-3** — Miguel should run the notebook with a valid key to verify Document Library creation and cited Q&A end-to-end before opening the PR.

---

### CB-13: Deploy cookbook ATF demo to api.robotross.art/atf-mistral
**Status**: deployed — pending MISTRAL_API_KEY  
**Owner**: Clau (2026-06-05)  
**What was deployed**:
- Cloned `feat/automated-technical-file` branch → `/opt/atf-mistral/` on DigitalOcean server (ssh robotsales = 159.223.22.165)
- Virtualenv at `/opt/atf-mistral/venv/` — cookbook `requirements.txt` + `fastapi>=0.110` + `uvicorn[standard]>=0.29`
- FastAPI service at `/opt/atf-mistral/third_party/automated-technical-file/serve.py`:
  - `GET /` — health check (always responds, `ready: false` until key is set)
  - `GET /demo` — runs CB-06 cited Q&A ("What is the bidding rule for the Wall of Fame?") against Mistral Document Library
  - `POST /query` — arbitrary question, same cited Q&A backend
  - On first startup with key: creates Document Library + Agent, caches IDs to `/opt/atf-mistral/state.json` to avoid recreating on restart
- systemd unit: `/etc/systemd/system/atf-mistral.service` — enabled + running on port 8003
- Caddy proxy: added `handle_path /atf-mistral/*` block to `/etc/caddy/Caddyfile` (before catch-all 8787); did NOT touch `/atf` live endpoint
- `/opt/atf-mistral/.env` — placeholder file with `MISTRAL_API_KEY=` (empty; chmod 600; Miguel must fill)

**Verified**:
- `https://api.robotross.art/atf-mistral/` → `{"service":"RobotRoss ATF Demo","ready":false,...}` ✅
- `https://api.robotross.art/atf-mistral/demo` → graceful 503 "MISTRAL_API_KEY not configured" ✅
- `https://api.robotross.art/atf/` → HTTP 200, untouched ✅

**To activate**:
1. `echo 'MISTRAL_API_KEY=<your-key>' > /opt/atf-mistral/.env && chmod 600 /opt/atf-mistral/.env`
2. `systemctl restart atf-mistral`
3. `curl https://api.robotross.art/atf-mistral/demo` — should return a cited answer

---

## CB-13 Open Questions

| # | Question | Recommendation |
|---|---|---|
| CB13-Q1 | **Proxy is Caddy, not nginx.** Task said nginx but the server uses Caddy for TLS on api.robotross.art. Caddy was configured instead — same result, different tool. No action needed unless you have a specific nginx preference. | Caddy fine |
| CB13-Q2 | **MISTRAL_API_KEY not in Infisical** (checked project `3233b7c1`). Currently wired via `/opt/atf-mistral/.env`. Miguel must manually add the key. If you want it in Infisical, add it with `INFISICAL_TOKEN=... infisical secrets set MISTRAL_API_KEY=<val> --env dev --projectId 3233b7c1-8309-447d-af5a-6541e38dc1b3` and switch the service to `infisical run ...` (see crm-api pattern). | Add key to .env |
| CB13-Q3 | **Port**: chose 8003 (8001 = tracker-api, 8002 = crm-api). No conflict found. | 8003 confirmed |
| CB13-Q4 | **Process manager**: chose systemd (consistent with tracker-api and crm-api). No supervisor/pm2 on server. | systemd confirmed |
| CB13-Q5 | **Agent/Library lifecycle**: on first boot with key, the service creates a fresh Mistral Document Library and Agent and caches their IDs to `/opt/atf-mistral/state.json`. Subsequent restarts reuse them. Deleting `state.json` forces recreation (and orphans the old Mistral resources). Miguel should decide when to clean up staging Agents/Libraries in the Mistral console. | Delete state.json to reprovision |
| CB13-Q6 | **Full cited Q&A not verified** — CB-13 cannot confirm a live cited result without a real MISTRAL_API_KEY. Health endpoint verified; 503 path verified. Miguel should run `/demo` after adding the key to confirm end-to-end. | Run /demo after key set |

---

## Branch status

```
Branch:   feat/automated-technical-file
Commits:  d3e8cbe (scaffold) → f08cb9f (runtime_adapter) → 29d779a (worklog) →
          7b1e3ce (README) → 6ef83b5 (README fix) → 1d40ba7 (cells 3-4)
Grep:     CLEAN — no secrets, no /Users/ paths, no Mexico, no Anthropic references
Models:   All hosted IDs match SPEC_models.md
Notebook: Cell 1 executes fully without API key; cells 2-3 gracefully degrade
```

**STOP — do not open PR.** Miguel reviews and opens it.
