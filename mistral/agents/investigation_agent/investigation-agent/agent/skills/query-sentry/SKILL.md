---
name: query-sentry
description: Query Sentry issues, events, stack traces, releases, environments, and tags for read-only production investigation evidence.
---
Start with `find_organizations`; if it is empty, ask for the org slug or take it from a shared Sentry link, then call `find_projects(organizationSlug=...)` and pass that org onward. Omit `regionUrl` unless a tool supplies one.
1. Establish project, environment, release, and time range (discover them if not given).
2. Search issues broadly; open relevant ones and inspect representative events, not just titles or counts.
3. Capture exception type and message, responsible frames, first/last seen, volume, release, environment, useful tags.
4. Classify: new, regressed, ongoing, resolved, or telemetry noise; state uncertainty.
5. Correlate with GitHub via the responsible frames and the release: load `query-github` to find the code and the change behind it.
6. Cite the issue or event link and the exact supporting evidence.
Read-only: never change issue state, assignments, ownership, or alerts. Never reproduce secrets, tokens, personal data, or full payloads.
