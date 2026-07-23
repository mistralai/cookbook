# Developer content guidelines

Developer documentation tends to be more technical than general content, but the same voice principles apply: be warm and relaxed, crisp and clear, and ready to lend a hand.

Assume developers bring a solid understanding of programming concepts. Skip the basics and focus on information specific to the Mistral API that helps them accomplish their goals.

---

## Code examples

Code examples show how to use an API element to implement specific functionality. They might include:

- Simple, one-line examples interspersed with text.
- Short, self-contained examples that illustrate a specific point.
- Longer examples that illustrate multiple features, complex scenarios, or best practices.

Developers use code examples to:
- Assess an API during planning.
- Learn or explore a language or technology.
- Write and debug code.

Many developers copy example code from documentation directly into their projects.

### Planning code examples

- Create concise examples that cover key tasks. Start with the simplest useful example and build up complexity.
- Prioritize frequently used elements and elements that may be difficult to understand or tricky to use.
- Don't use code examples to illustrate obvious points or contrived scenarios.
- Reserve complicated examples for tutorials and walkthroughs, where you can provide a step-by-step explanation.
- Add an introduction to describe the scenario and explain anything that might not be clear from the code. List requirements and dependencies.
- Provide an easy way for developers to copy and run the code.

### Writing code examples

- Design code for reuse. Help developers identify what to modify. Add comments to explain non-obvious details—don't state the obvious.
- Show expected output, either in a separate section after the code or in code comments.
- Write secure code. Never hard-code credentials. Use placeholders like `os.environ["MISTRAL_API_KEY"]` or `"your-api-key"`.
- Show exception handling only when it's intrinsic to the example.
- Always compile and test your code before publishing.

### Code formatting rules

- Every code block must have a language tag: ` ```python `, ` ```typescript `, ` ```bash `.
- Use code style (backticks) for all programmatic elements inline: function names, parameters, methods, file paths, error messages, and environment variable names.
- Placeholders the reader must replace should be obvious: use `"your-api-key"`, `<connector-id>`, or `YOUR_API_KEY`.
- Keep language parity: if you show Python, show TypeScript and curl too, unless there's a clear reason not to.
- Show realistic output—not just "response printed" but the actual structure of the response.

### Package manager install commands

When showing install commands for multiple package managers, give each its own labeled code block. Do not combine them in a single block with `#` comments.

**Correct:**

```
**npm**:
```bash
npm install @mistralai/mistralai
```

**pnpm**:
```bash
pnpm add @mistralai/mistralai
```

**yarn**:
```bash
yarn add @mistralai/mistralai
```
```

**Incorrect:**

```
```bash
# npm
npm install @mistralai/mistralai

# pnpm
pnpm add @mistralai/mistralai

# yarn
yarn add @mistralai/mistralai
```
```

---

## Formatting developer text elements

Use code style (backticks in Markdown) for all programmatic elements. The following table covers common cases:

| Element | Format | Example |
|---|---|---|
| AI prompts (in prose) | Quotation marks | "Summarize this document." |
| Attributes | Code style | `connector_id` |
| Classes | Code style | `Mistral`, `ChatMessage` |
| Code samples and keywords | Code fenced block | ` ```python ``` ` |
| Command-line commands | Code style | `pip install mistralai` |
| Environment variables | Code style | `MISTRAL_API_KEY` |
| Error messages (in prose) | Code style | `401 Unauthorized` |
| File extensions | Code style | `.py`, `.ts` |
| File names | Code style | `main.py` |
| File paths | Code style | `/v1/chat/completions` |
| HTTP methods | Code style | `POST`, `GET` |
| Method and function names | Code style | `chat.complete()` |
| Operators | Code style | `=`, `!=` |
| Parameter names | Code style | `model`, `messages`, `tools` |
| Property names | Code style | `connector_id`, `agent_id` |
| Return values | Code style | `choices[0].message.content` |
| String literals | Code style | `"mistral-small-latest"` |
| Type names | Code style | `str`, `Optional[str]` |

---

## Reference documentation structure

Reference articles follow a consistent structure so developers can find information quickly. Include these sections where applicable:

| Section | Contents |
|---|---|
| **Title and description** | The name of the element and a concise one-to-two sentence description of what it does. Don't just repeat the element name. |
| **Declaration/syntax** | The code signature. If multiple languages are supported, provide syntax for each. |
| **Parameters** | Each parameter with its data type, whether it's required or optional, and what it does. Don't just repeat the parameter name. |
| **Return value** | What the function returns and its data type. |
| **Remarks** | Additional context, comparison with similar elements, potential issues. |
| **Example** | A working code example. |
| **See also** | Links to related elements or articles. |

### Writing element descriptions

- Start with a verb that describes what the element does: *Returns*, *Sets*, *Creates*, *Deletes*.
- Don't start with "This method...", "This class...", or the element name.
- Be specific. Describe the behavior, not just the type.

**Examples**
- Bad: `complete()` — "This is the method for chat completions."
- Good: `complete()` — "Sends a list of messages to the specified model and returns the model's response."

---

## Procedure writing for developers

When documenting a multi-step process:

1. Use numbered steps.
2. Start each step with an imperative verb: *Install*, *Set*, *Run*, *Call*, *Copy*.
3. Include the expected result of a step when it's not obvious.
4. Provide code for steps that involve code.
5. Call out prerequisites before the first step.
6. Show the complete working state at the end (e.g., expected output or a confirmation message).

**Example structure:**

```
## Set up authentication

1. Get your API key from the [Mistral console](https://console.mistral.ai/).

2. Set the `MISTRAL_API_KEY` environment variable:
   ```bash
   export MISTRAL_API_KEY="your-api-key"
   ```

3. Verify the key is set:
   ```bash
   echo $MISTRAL_API_KEY
   ```
```
