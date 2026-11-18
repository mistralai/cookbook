---
name: query-github
description: Read GitHub code, commits, pull requests, issues, and releases, and tie an error or regression to the change that likely caused it.
---
1. Establish the repository: take it from a shared link, the Sentry project or stack frame paths, or `search_repositories`; ask only if several match.
2. For a stack frame, open the file at the deployed release or tag with `get_file_contents`, not only the default branch, and read the surrounding code.
3. For a regression, bound the window with the Sentry first-seen time and the releases (`list_releases`, `get_release_by_tag`, `list_tags`), then list commits and merged pull requests in that window that touch the responsible files (`list_commits`, `search_pull_requests`, `pull_request_read`).
4. Use `search_code` to find other callers or the origin of an error message; use `search_commits` and `issue_read` for known fixes or reports.
5. Name the most likely change with its commit or pull request link, and say plainly when the evidence only narrows the window.
Read-only: never create, comment on, review, merge, or close anything. Never reproduce secrets or credentials found in code.
