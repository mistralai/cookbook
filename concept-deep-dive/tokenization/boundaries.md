# Tokenization

## Boundaries and Token Healing

### Basics - Recap

Tokenization converts a string into a sequence of token IDs that the model uses as input. Tokens can be characters, subwords, or symbols. The tokenizer breaks the string into tokens and converts them into IDs (encoding) and vice versa (decoding). The model then predicts the next token based on the sequence. For example, the sentence "Hello, how are" might be tokenized as: `Hello,` -> `432`, ` how` -> `523`, ` are` -> `87`. The model predicts the next token, such as `75` (decoded as ` you`), allowing it to generate coherent text. To use an LLM as a chatbot, you can fine-tune it with conversations following a template. The model starts with a Beginning of String (BOS) token (`<s>`) and uses an End of String (EOS) token (`</s>`) to stop generation. To avoid confusion and ensure proper stopping, you can use rare strings for formatting and insert EOS tokens after each Assistant response, an example could be:
```
<s>[INST] Hello, how are you? [/INST] Fine, and you?</s>[INST] I'm doing great! [/INST] Glad to hear!</s>
```

### The Boundary Problem

To explain this problem, let's use a different example. Consider a URL like this: `https://mistral.ai`.

Such URLs could potentially be tokenized similarly to something like: `https` + `://` + `mistral` + `.ai`.

With `https` being a very frequent string, it will very likely be a token on its own, and most secure valid URLs will be tokenized with it.

Now, what would happen if we ask our model to generate the next token after `http`? Note that we are using `http` and not `https`.

Will it try to predict `s` to make `https`? Very unlikely. The model has never encountered a sequence of tokens such as `http` + `s`. Instead, it has always seen `https` as a single token during training. It will be biased and more likely predict `://` directly, resulting in an insecure URL.

In this case, this boundary problem only introduces a slight bias. However, in some cases, it can lead to completely wrong outputs.

Let's say we give it `https:` and ask it to predict the next token.

The tokenizer will encode this as `https` + `:`.

Now, what will be the next token? `//`? You guessed it-it won't. It barely or even never saw such a sequence in its training data. The `:` that was added can disturb the distribution immensely. It finds itself in a situation where it may even avoid using the correct token, prioritizing token sequences that it actually saw in its training data. This can potentially lead to completely wrong outputs, making invalid URLs in this case.

With our previous template in mind, we have no way to know, for example, how `...[/INST]...` will be tokenized. In very bad scenarios, we may end up with `[/INST] Fine` tokenized as `[/` + `INST` + `] Fine`. Creating boundary problems around our special strings.

This highlights the importance of understanding and addressing boundary problems in tokenization. Ensuring that the model is trained on a diverse and comprehensive dataset can help mitigate some of these issues, and other techniques may also make the model more robust.

### Solution A
One possible solution involves reworking our template to ensure the last token is a clean expected token resulting in an expected sequence. We can do this by encoding the user message encapsulated with the special strings on its own, and only after we encode the assistant response and concatenate both sequences.

The result would be:
```
BOS_ID
+ encode("[INST] Hello, how are you? [/INST]")
+ encode("Fine, and you?") + EOS_ID
+ encode("[INST] I'm doing great! [/INST]")
+ encode("Glad to hear!") + EOS_ID
```

With this method, we are certain that the sequence provided to the model for completion, which will most of the time end with a clean `[/INST]` properly tokenized, will not have any boundary problems.

#### Tokenizer V1
With all of this in mind, we are actually very close to understanding everything needed to know about our very first tokenizer used for our very first model Mistral 7B v0.1 and Mistral 7B v0.2, as well as Mixtral 8x7B v0.1. The only difference concerns the string representation. Previously, we mentioned that the string representation could look like:
```
<s>[INST] Hello, how are you? [/INST] Fine, and you?</s>[INST] I'm doing great! [/INST] Glad to hear!</s>
```
However, our first tokenizers used `sentencepiece`, which encodes by default by prepending with a whitespace. This means that at each encoding step, we add a new whitespace at the start. Meaning, for example, that `encode("[INST] Hello, how are you? [/INST]")` will actually encode `" [INST] Hello, how are you? [/INST]"`.

With this in mind, we end up with:
```
<s> [INST] Hello, how are you? [/INST] Fine, and you?</s> [INST] I'm doing great! [/INST] Glad to hear!</s>
```

### Solution B - Token Healing

The previous solution does not fully solve the problem for more general cases, but only around the special strings. This means that if we desire to prefix/prefill a string for the assistant response, we will once more encounter boundary issues.

This problem also affects even more basic models that are not fine-tuned for chat completions. <u>This means that you are constantly a victim of boundary issues when using a base model.</u>

Token Healing addresses the problem by simply removing the last token of the sequence and then constraining the next token generation to start with a string corresponding to the removed token. This completely solves our boundary problem at the cost of one token generation.

Let's take our URL example, when with a sequence such as `https` + `:`, we remove our last token `:` and then constrain the next token generation to start with ":", allowing our model to properly predict `://`.
