# WORKLOG — CB-07

Task: `0zvvcy89x599j5u` / CB-07

Plan:

1. Vendor a trimmed `ledger_to_md.py` under `third_party/automated-technical-file/tools/`.
2. Add the minimal sample ledger and generated wiki output needed for the cookbook demo.
3. Create notebook cell 2 so it regenerates the wiki from `sample_events.jsonl`.
4. Run the script and verify the notebook cell code path works locally.

Notes:

- CB-04 is still `waiting_human_notified`; its sample files are absent from the cookbook fork.
- To keep CB-07 runnable, this branch adds the sample ledger needed by cell 2 without modifying the live ATF source repo.
