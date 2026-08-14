# Modernize legacy Python code with GLM

Use the `zai-glm-5-2` (GLM) model and the GitHub connector to read outdated Python code from a repository and produce modernized versions.

## Prerequisites

### Install

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Required environment variables

To complete this cookbook, you'll need a Mistral API key. In [Studio](https://console.mistral.ai), navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys) and create a new API key.

Copy the example `.env` file and add your Mistral API key:

```bash
cp .env.example .env
```

```
MISTRAL_API_KEY=your-mistral-api-key
GITHUB_REPO=mistralai/cookbook
GITHUB_BRANCH=main
```

To modernize files from a different repository, change `GITHUB_REPO` and `GITHUB_BRANCH`.

## 1. Understand the approach

GLM (`zai-glm-5-2`) is a code-generation model available through the Mistral API. In this cookbook, you pair it with the GitHub connector (`github_app`) to read legacy Python files from a repository and produce modernized versions.

The workflow is:

1. **Authenticate** the GitHub connector via OAuth
2. **Create an agent** with GLM and the GitHub connector
3. **Start a conversation** that instructs the agent to read specific files and modernize them
4. **Extract the code** from the response and save it locally

The agent uses the GitHub connector to read files server-side — you don't need to clone the repository or handle any tool calls yourself.

The `legacy_app/` directory contains an intentionally outdated to-do list CLI packed with Python 2/3.5-era patterns: `%`-formatting, `os.path`, manual `open()`/`close()`, bare `except:`, `type()` checks, `sys.argv` parsing, and more. GLM modernizes these into idiomatic, type-annotated Python.

## 2. What gets modernized

The legacy app uses these outdated patterns, which GLM replaces:

| Outdated pattern | Modern replacement |
|---|---|
| `%`-formatting | f-strings |
| `os.path` | `pathlib.Path` |
| Manual `open()`/`close()` | `with` statements |
| `json.loads(f.read())` | `json.load(f)` |
| Bare `except:` | Specific exceptions |
| `type()` checks | `isinstance()` |
| Mutable default argument | `None` sentinel |
| `sys.argv` parsing | `argparse` |
| No type hints | Full type annotations |
| `== True` comparisons | Truthiness checks |
| `range(len(...))` | `enumerate` / direct iteration |

## Run

Run the script to modernize the legacy app:

```bash
python modernize_code.py
```

The script prompts you to authenticate the GitHub connector via OAuth, then creates a GLM agent, reads the legacy files from the repository, produces modernized versions, and saves them to a `modernized/` directory.

Example output:

```
Authenticate the GitHub connector:
https://github.com/login/oauth/authorize?client_id=...&scope=repo+...
Press Enter once you've completed the OAuth flow in your browser...

Created agent: code_modernizer (a1b2c3d4-5678-90ab-cdef-1234567890ab)
Reading files from mistralai/cookbook and modernizing...
This may take a few minutes.

Saved: modernized/app.py
Saved: modernized/utils.py

Modernized 2 file(s) in modernized/
Deleted agent: a1b2c3d4-5678-90ab-cdef-1234567890ab
```

## Summary

You built a code modernizer that reads legacy Python files from GitHub and produces idiomatic, modern Python.

- Created a GLM agent with the GitHub connector for repository access
- Used the Conversations API to read and modernize files in a single request
- Extracted and saved the modernized code locally

**Mistral features used:**

- Agents API (beta) with the `zai-glm-5-2` model
- Conversations API (beta) for server-side tool execution
- GitHub connector (`github_app`) for repository file access
- Extended timeout (`timeout_ms`) for long code generation

Try pointing the script at a different repository or different files. For more on connectors, see the [connectors documentation](https://docs.mistral.ai/capabilities/connectors/).
