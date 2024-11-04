# Tokenization

Tokenization is a crucial concept around LLMs, and it can be more complex than one may think!

For our tokenization implementation, please refer to [mistral-common](https://github.com/mistralai/mistral-common).

We currently have 3 main versions of our tokenizer:
- V1: The tokenizer behind our very first models.
- V2: Introducing control tokens and function calling!
- V3: Better function calling implementation.
    - V3-Tekken: Different version based on `tiktoken`, opposed to the other versions based on `sentencepiece`.

## Overview

| Section                  | Description                                                                 |
|:------------------------:|:---------------------------------------------------------------------------:|
| [Basics](basics.md)               | Basics of tokenization. |
| [Boundaries & Token Healing](boundaries.md)               | Main problems with tokenization and token healing. |
| [Control Tokens](control_tokens.md)               | Introduction to Control Tokens and their advantages. |
| [Templates](templates.md)               | A summarized list of all our tokenizers with their chat templates.           |
| [Tokenizer](tokenizer.md)          | Make your own tokenizer with sentencepiece.                             |
| [Tool Calling](tool_calling.md)          | Learn about how tokenization for our tool calling works.                            |
|          |                            |
| [Chat Templates](chat_templates.md)          | Legacy documentation around our chat templates.                             |