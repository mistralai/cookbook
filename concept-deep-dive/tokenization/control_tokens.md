# Tokenization

## Control Tokens

### Boundaries - Recap

Boundary problems occur when the model encounters unexpected token sequences, leading to inaccurate predictions. For example, the model may struggle with sequences like `http` + `s` that it never saw during training if it instead saw `https`.

To address boundary issues, Token Healing removes the last token of the sequence and constrains the next token generation to start with a string corresponding to the removed token. This solves boundary problems at the cost of one token generation.

### Control Tokens

Control tokens tackle many problems at once, but let's focus on three of them:
- **Efficiency**: Control Tokens allow for improved efficiency by using fewer tokens.
- **Security**: Control Tokens allow models to be more robust against prompt injections.
- **Boundary issues**: Control Tokens also help mitigate boundary problems by default.

Previously, we have been encoding as follows:
```
BOS_ID
+ encode("[INST] Hello, how are you? [/INST]")
+ encode("Fine, and you?") + EOS_ID
+ encode("[INST] I'm doing great! [/INST]")
+ encode("Glad to hear!") + EOS_ID
```

Our efficiency problem comes from the fact that the special strings `[INST]` and `[/INST]` are being tokenized normally with the rest of the text. Why is this not efficient? As far as we know, the tokenizer may encode them as `[` + `INST` + `]` or `[/` + `INST` + `]`, three tokens per string. This is not efficient if we consider that these two strings will be present everywhere all the time.

In terms of security, let's imagine we have our model in a chat application. The user himself can insert special strings into his message, and nothing stops him. We may end up with things like:
```
BOS_ID
+ encode("[INST] Hello, [INST] [/INST] how are you? [/INST]")
+ encode("Fine, and you?") + EOS_ID
+ encode("[INST] I'm doing great! [/INST]")
+ encode("Glad to hear!") + EOS_ID
```

This breaks the template completely, allowing the user to inject and break the expected model behavior.

The idea behind control tokens is to introduce, at the fine-tuning phase, new tokens that the model never saw previously and that the user will never inject simply because they do not exist in the string spectrum, existing only as a token ID. We basically add new token IDs that will replace our special strings. For representation and representation reasons only, we will use `[INST]` and `[/INST]` as their string representations, but similarly to the BOS and EOS, the tokenizer never actually sees those strings; they are simply a way for us to know what we are discussing.

So, instead of encapsulating our user instructions with our special strings, we will encapsulate them with our control token IDs directly.

The end result will be something like:
```
BOS_ID
+ INST_ID
+ encode("Hello, how are you?")
+ /INST_ID
+ encode("Fine, and you?") + EOS_ID
+ INST_ID
+ encode("I'm doing great!")
+ /INST_ID
+ encode("Glad to hear!") + EOS_ID
```

Indirectly, this also reduces boundary issues around the special strings since we are no longer working with strings but with token IDs.

The theoretical string representation changes slightly:
```
<s>[INST]Hello, how are you?[/INST]Fine, and you?</s>[INST]I'm doing great![/INST]Glad to hear!</s>
```

#### Tokenizer V2/V3

All tokenizers after V1 use Control Tokens. The end result of their string representations, however, is still affected by `sentencepiece` for the most part, adding a whitespace at each encoding step similarly to our V1:
```
<s>[INST] Hello, how are you?[/INST] Fine, and you?</s>[INST] I'm doing great![/INST] Glad to hear!</s>
```

The only difference between V2 and V3 is regarding the **tool calling**, otherwise everything is identical. You can learn more about tool calling [here](tool_calling.md).

#### Tokenizer V3-Tekken

This tokenizer is a variant of the V3 tokenizer. Everything regarding the encoding and format is the same, hence it is also called V3. The main difference is that it does not use `sentencepiece` but `tiktoken` instead. This means that there are no whitespaces added by default, making the end string representation clean:
```
<s>[INST]Hello, how are you?[/INST]Fine, and you?</s>[INST]I'm doing great![/INST]Glad to hear!</s>
```
