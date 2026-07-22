# Cookbook Review Skill

Review a Mistral AI cookbook file for content quality, structural completeness, and writing style. Produce a prioritized list of issues and suggested fixes.

## When to use this skill

Trigger when the user asks to:
- Review, audit, or proofread a cookbook file
- Check if a cookbook follows the standard format
- Get feedback on a draft cookbook
- Validate a cookbook before merging

## How to perform a review

1. Read the target file in full using the Read tool.
2. Evaluate it against the **Structure checklist** and **Writing style checklist** below.
3. Output a structured review in the format specified at the end of this document.

---

## Cookbook structure template

### Required sections (must be present)

Every cookbook file must include these sections, in this order:

```
# [Title]                          ← H1: concise, task-oriented
[One-sentence description]         ← immediately under title, no heading
> [Status/note callout]            ← blockquote for beta APIs or important constraints

[Introduction text]                ← optional ## Introduction heading, or prose directly under the description

## Prerequisites                   ← H2: "To complete this cookbook, you will need:" + bullet list only

## Environment setup               ← H2

### Install                        ← H3

### Required environment variables ← H3

## [Step heading]                  ← H2: one or more step sections (numbered or unnumbered)
...

## Clean up                        ← H2 (optional but suggested): remove resources created during the tutorial

## Summary                         ← H2: closing summary (required, must be last)
```

**Step headings** are H2 and may be numbered or unnumbered — both forms are acceptable:
- `## 1. Create the connector`
- `## Create the connector`

Each step section must open with at least one sentence before any code block, table, or list (see section content standard below).

The `## Prerequisites` section must contain only the sentence "To complete this cookbook, you will need:" followed by a bulleted list. List items should appear in this order:
1. Language runtime and tooling (e.g., "Python 3.9 or later", "Node.js and a package manager")
2. A Mistral account and API key (nearly always required)
3. Any other accounts, tokens, or external services needed

Flag as **Critical** if `## Prerequisites` contains installation instructions, environment variable setup, or code blocks — those belong in `## Environment setup`.

The `## Environment setup` section contains everything the reader needs to run the code: package installation, API key setup, and (for standalone scripts) a run command. See the API key setup standard below for exact wording.

The `## Summary` section must be the **last section** in the file — no content, headings, or sections may follow it. It must contain:

1. **1–2 sentence overview** of what the cookbook covered and what was built or demonstrated.
2. **What you built** (or **What this cookbook covers**) — a bullet list.
3. **Mistral features used** — a bullet list of Mistral APIs, products, and tools referenced (e.g., Connectors, Conversations API, Agents API, Workflows, Chat Completions API, built-in tools).
4. **Other services** (optional) — a bullet list of third-party tools or MCP servers used. Omit this heading if there are none.
5. **One CTA** — a single link to relevant documentation or a Studio page. Use one of:
   - A known Studio URL (e.g., `View your Connectors in [Studio](https://console.mistral.ai/build/connectors).`)
   - A known documentation URL
   - If you don't know the right destination, use: `[View the documentation]() <!-- TODO: add link to relevant documentation -->`

Flag as **Critical** if the `## Summary` section is missing entirely or if it is not the last section in the file.
Flag as **Moderate** if the CTA link is empty and has no `TODO` comment, or if any of the four required elements (overview, what was built, Mistral features, CTA) is absent.

### Optional sections (include when relevant)

| Section | When to include |
|---|---|
| Comparison table (e.g., API A vs. API B) | When two similar options exist and choice matters |
| Troubleshooting guide | When common runtime errors need extended explanations |
| Error codes reference table | When the API returns many distinct HTTP error codes |

### Section content standard

Every heading — at any level — must be followed by at least one sentence of body text before any code block, table, or list. The sentence should tell the reader what they're looking at or what to do.

Flag as **Moderate** if any heading is followed immediately by a code block, table, or list with no introductory sentence.

**Examples:**

`### Install` — must have an intro sentence before the first code block:
- TypeScript: "Use one of the following methods to install the Mistral TypeScript SDK:"
- Python: "Install the Mistral Python SDK:" or "Run the following command to install the Mistral Python SDK:"

`### Run` — must explain what the command does before showing it:
- "Once your `.env` file is in place, run the script:"

Apply the same rule to all other headings: `### Required environment variables`, recipe headings, step headings, and so on.

### API key setup standard

Every cookbook that requires a Mistral API key must follow this exact wording and structure. Flag any deviation as a **Critical** issue.

#### Getting the key

The "Environment setup" section must include this sentence verbatim (Markdown links intact):

> To complete this cookbook, you'll need a Mistral API key. In [Studio](https://console.mistral.ai), navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys) and create a new API key.

Flag these as Critical and suggest the standard sentence:
- Any other URL for creating an API key (e.g., `/api-keys`, `/dashboard`, or a bare `console.mistral.ai` link without the profile dialog deep-link)
- Phrasing that omits the Studio link entirely
- Using "Mistral AI dashboard," "Mistral Console," or any name other than "Studio" for the console

#### .env file (Markdown cookbooks and Python projects)

When the project uses a `.env` file, the instructions must use this exact wording and formatting:

> Create a `.env` at the root of your project and add your Mistral API key:
>
> ```
> MISTRAL_API_KEY=your-mistral-api-key
> ```

If the project requires additional API keys (e.g., a GitHub token or a third-party service key), list `MISTRAL_API_KEY` first and any project-specific keys below it:

```
MISTRAL_API_KEY=your-mistral-api-key
OTHER_SERVICE_API_KEY=your-other-api-key
```

Flag these as Critical:
- Values wrapped in quotes: `MISTRAL_API_KEY="your-key"` → remove the quotes
- Wrong variable name: `MISTRAL_KEY` → must be `MISTRAL_API_KEY`
- Missing code block language tag on `.env` content — use a plain fenced block (no language tag) for env files
- Instructions that say "set an environment variable" without showing the `.env` file pattern

Flag as Moderate:
- `.env` instructions placed outside the Prerequisites section

#### Notebook (.ipynb) API key cell

Jupyter notebooks must load the key from the environment via dotenv. The first code cell (or the first cell that touches the API key) must follow this pattern:

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]
```

Flag these as Critical:
- Hard-coded API key strings in any cell
- `MISTRAL_KEY` instead of `MISTRAL_API_KEY`
- Using `os.getenv("MISTRAL_API_KEY")` without a fallback or error — prefer `os.environ["MISTRAL_API_KEY"]` so the notebook fails loudly if the key is missing

Flag as Moderate:
- Missing `load_dotenv()` call when a `.env` file is expected
- `api_key = os.environ["MISTRAL_API_KEY"]` defined but never passed to the client constructor

#### Shell / curl cookbooks

Cookbooks that are primarily curl-based (no Python runtime) may use shell exports instead of a `.env`:

```bash
export MISTRAL_API_KEY="your-api-key"
```

This is acceptable for curl-only examples. Do not flag this pattern as an error. Still flag the wrong variable name (`MISTRAL_KEY`) or a missing API key instruction.

---

### Sections to exclude

Do not include these in a cookbook:

- **Marketing language** — No "Mistral's powerful AI" or "cutting-edge capabilities."
- **Generic introductions** — No "In today's world of AI..." or "As AI becomes more important..."
- **Closing promotional statements** — No "Start building today!" or "Unlock the full potential."
- **Theory-only sections without code** — Concepts must be paired with a concrete example.
- **Repeated prerequisites** — State install instructions once; don't repeat them in individual steps.
- **Table of Contents** — The documentation UI does not render TOC links, so they add noise without benefit. Remove any `## Table of Contents` section entirely.
- **Nested TOC duplication** — Don't list recipe sub-steps in a table of contents.
- **Empty sections** — Remove any heading with no content beneath it.
- **Changelog or version history** — Belongs in release notes, not cookbooks.
- **Contributing guidelines** — Use the repo's CONTRIBUTING_GUIDE.md instead.

---

## Writing style checklist

Based on the Mistral Writing Style Guide (see reference files below).

### Voice and tone

- **Be direct.** Lead with what the reader needs to do or know. Cut throat-clearing openers.
  - Bad: "Before we dive into using connectors, it's worth understanding what they are."
  - Good: "Connectors let the model call external tools via MCP."
- **Write like you speak.** Use contractions (*it's*, *you'll*, *don't*). Avoid stiff, formal phrasing.
  - Bad: "It is necessary to ensure that the client is initialized."
  - Good: "Initialize the client before making requests."
- **Address the reader as "you."** Don't use "the user" or "one" when you mean the person reading.
- **Start statements with a verb.** Edit out "you can" when it isn't necessary.
  - Bad: "You can also specify an optional timeout."
  - Good: "Specify an optional timeout."
- **Avoid weak openers.** Rewrite sentences that start with *there is*, *there are*, or *there were*.
  - Bad: "There are two ways to authenticate."
  - Good: "Two authentication methods are available."

### Clarity and brevity

- **Keep sentences short.** One idea per sentence. Three to seven lines per paragraph.
- **Prune every excess word.** If a word doesn't add meaning, cut it.
  - Bad: "In order to be able to make a request..."
  - Good: "To make a request..."
- **Use simple words.** Prefer *use* over *utilize*, *start* over *initiate*, *show* over *display*.
- **Don't use jargon without defining it.** On first use, briefly explain non-obvious terms.
- **Front-load headings and sentences.** Put the most important word or phrase first.

### Headings

- **Use sentence-style capitalization.** Capitalize only the first word and proper nouns.
  - Bad: `## Creating A Connector With OAuth Authentication`
  - Good: `## Creating a connector with OAuth authentication`
- **No period at the end of headings.**
- **Don't use ampersands (&) or plus signs (+)** unless referring to UI that contains them.
- **Keep headings short and specific.** A heading should tell the reader exactly what they'll find.
- **Use parallel structure** across headings at the same level.
  - Bad mix: `## Create a connector`, `## Listing connectors`, `## How to delete a connector`
  - Good: `## Create a connector`, `## List connectors`, `## Delete a connector`
- **Avoid two headings in a row** without body text in between.
- **Never open a section with a code block, table, or list.** Every heading must be followed by at least one sentence before any code or structured content. See the section content standard above.

### Lists

- **Use bullet lists for unordered items; numbered lists for sequential steps.**
- **Keep list items parallel** in grammar and structure.
- **Include a comma before "and"** in a series of three or more items (Oxford comma).
  - Bad: "Python, TypeScript and curl"
  - Good: "Python, TypeScript, and curl"
- **Don't use a period** at the end of single-sentence bullet items unless they are full sentences that continue into multiple sentences.

### Punctuation

- **One space after periods**, not two.
- **No spaces around em dashes.** Use `—` not ` - ` for parenthetical dashes.
- **Don't use a colon** at the end of headings or list introductions in most cases.
- **Use straight quotes**, not curly/smart quotes, in code and code-adjacent content.

### Code and code examples

- **Every code block must have a language tag**: ` ```python `, ` ```typescript `, ` ```bash `.
- **Never hard-code real credentials.** Use placeholders like `"your-api-key"` or `os.environ["MISTRAL_API_KEY"]`.
- **Show expected output.** Always follow a code block with an example of what it prints or returns.
- **Comment sparingly.** Add comments only when the logic isn't self-evident from the code. Don't state the obvious.
- **Compile and test all code.** Verify that every example runs without errors.
- **Prioritize frequently used elements.** Start with the simplest useful example; build toward complex.
- **Placeholders must be obvious.** Any value the reader must replace should be clearly marked (e.g., `<your-connector-id>` or `"your-agent-id"`).
- **Match the language to the tutorial.** A TypeScript tutorial shows TypeScript; a Python notebook shows Python. Don't mix languages within the same tutorial unless the cookbook explicitly compares them.

### Accessibility and inclusive language

See [`inclusive-language.md`](./inclusive-language.md) for the full guide.

- **Use people-first language** when referring to people with disabilities.
  - Bad: "blind users," "disabled developers"
  - Good: "users who are blind," "developers with disabilities"
- **Use gender-neutral terms.**
  - Bad: "he," "she," "manpower," "chairman"
  - Good: "they," "workforce," "chair"
- **Avoid gendered pronouns in generic references.** Rewrite in second person or use plural.
- **Avoid terms with unconscious racial bias.**
  - Bad: "master/slave," "blacklist/whitelist"
  - Good: "primary/subordinate," "allowlist/denylist"
- **Don't use slang** that could be considered cultural appropriation.
- **Use title-style capitalization** for racial and ethnic group names: Black, White, Indigenous, Hispanic, Latinx.

### Terms to avoid in cookbook content

| Avoid | Use instead |
|---|---|
| MCP connectors / mcp connectors (referring to the product) | Connectors (capital C) |
| utilize | use |
| initiate / instantiate (in prose) | start, create |
| leverage (as a verb) | use, take advantage of |
| seamless | (just describe what it does) |
| robust | (just describe the capability) |
| powerful | (just describe what it does) |
| easy, simple | (omit — let the code demonstrate) |
| just (minimizing word) | (omit) |
| please | (omit — direct is fine) |
| Note that... | (cut the throat-clearing; state the note directly) |
| In order to | To |
| It is important to | (state why it's important or cut) |
| There is / there are | rewrite to lead with the subject |
| you can | (cut when introducing a step the reader should do) |

---

## Review output format

Write the review as a Markdown document with the following structure:

```markdown
## Cookbook review: [filename]

### Summary
[2–4 sentence overview of overall quality and the most critical issues]

### Critical issues
<!-- Must fix before publishing -->
- **[Issue type]** [Line or section reference]: [What's wrong and why it matters]
  - Suggested fix: [Concrete rewrite or action]

### Moderate issues
<!-- Should fix for quality -->
- **[Issue type]** [Line or section reference]: [What's wrong]
  - Suggested fix: [Concrete rewrite or action]

### Minor issues
<!-- Nice to fix, low impact -->
- **[Issue type]** [Line or section reference]: [What's wrong]
  - Suggested fix: [Concrete rewrite or action]

### What's working well
- [Positive observation]
- [Positive observation]

### Structure checklist
| Section | Status | Notes |
|---|---|---|
| H1 title | ✅ / ❌ / ⚠️ | |
| One-sentence description | ✅ / ❌ / ⚠️ | |
| Status/note callout (if applicable) | ✅ / ❌ / N/A | |
| Prerequisites (bullet list only) | ✅ / ❌ / ⚠️ | |
| Environment setup > Install | ✅ / ❌ / ⚠️ | |
| Environment setup > Environment variables | ✅ / ❌ / ⚠️ | |
| Step sections (H2, at least one) | ✅ / ❌ / ⚠️ | |
| Intro sentence before code in each step | ✅ / ❌ / ⚠️ | |
| Code blocks have language tags | ✅ / ❌ / ⚠️ | |
| Clean up section (optional) | ✅ / N/A | |
| Closing summary (last section) | ✅ / ❌ / ⚠️ | |
| No forbidden sections | ✅ / ❌ | |
```

**Status key:**
- ✅ Present and correct
- ❌ Missing or incorrect (critical)
- ⚠️ Present but needs improvement
- N/A Not applicable for this cookbook

**Issue type labels:**
- `[Missing section]` — required section is absent
- `[Forbidden section]` — section that shouldn't exist is present
- `[Style]` — writing style violation
- `[Clarity]` — confusing or ambiguous content
- `[Code]` — code block issue (missing tag, hard-coded credential, no output shown, etc.)
- `[Structure]` — heading level, ordering, or formatting problem
- `[Accessibility]` — inclusive language or bias issue
- `[Accuracy]` — likely factual or technical error (flag for human verification)

---

## Reference files

These files live alongside this skill and contain the full guidance referenced above:

- [`voice-and-tone.md`](./voice-and-tone.md) — Brand voice, three voice principles, top 10 writing tips with cookbook examples
- [`checklists.md`](./checklists.md) — Acronyms, capitalization, grammar, numbers, procedures, punctuation, responsive content, text formatting, and word choice checklists
- [`ai-terms.md`](./ai-terms.md) — Preferred AI terminology, terms to avoid, capitalization rules, describing model behavior
- [`developer-content.md`](./developer-content.md) — Code examples (planning and writing), formatting developer text elements, reference documentation structure, procedure writing
- [`inclusive-language.md`](./inclusive-language.md) — Gender-neutral language, accessibility terms, racial and ethnic language, bias-free technical terminology, militaristic language, inclusive code examples
