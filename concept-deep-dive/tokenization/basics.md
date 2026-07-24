# Tokenization

## Basics

### LLM - Recap
LLMs (Large Language Models) as we know them today are predominantly auto-regressive models based on the transformer architecture. While their applications vary and amaze the world with highly skilled chatbots, the underlying technology follows a simple principle: LLMs can be described as Text Completion Machines. Given a sentence, text, or string, the model splits it into multiple pieces called "Tokens." Trained on sequences of these tokens, the model predicts the next token to complete the sequence.

### What is Tokenization?

Tokenization is the process of converting a request, often a string, into a sequence of token IDs that the model will receive as input. A token can be anything from a single character to a subword or a symbol. The model possesses a vocabulary of tokens with their corresponding IDs. The tokenizer's job is to break apart the string and convert it into a sequence of token IDs, a step called "encoding." It is also the tokenizer's job to convert a sequence of IDs back into a string, a step called "decoding." Once provided with a sequence of token IDs, the model tries to predict the next token.

For simplicity, let's describe tokens as characters and full words.

Imagine a scenario where we have the following sentence: "Hello, how are"

1. The tokenizer will first break apart the sentence into different pieces, or tokens. For example:
   - `Hello,`
   - ` how`
   - ` are`

2. The tokenizer will then get the token ID corresponding to each token. For example:
   - `Hello,` -> `432`
   - ` how` -> `523`
   - ` are` -> `87`

   This sequence of token IDs is then provided to the model.

3. The model will try to predict the next token. Let's say it predicts:
   - `75`

   We will then have a full sequence of token IDs:
   - `432` + `523` + `87` + `75`

4. This sequence can be decoded back into a full string:
   - `Hello,` + ` how` + ` are` + ` you` -> `Hello, how are you`

We can then repeat this process as much as needed.

5. The model will try to predict the next token again. Let's say it predicts:
   - `868`

   We will then have a full sequence of token IDs:
   - `432` + `523` + `87` + `75` + `868`

6. This sequence will be decoded back into a full string:
   - `Hello,` + ` how` + ` are` + ` you` + `?` -> `Hello, how are you?`

This iterative process of encoding, prediction, and decoding allows the model to generate coherent and contextually relevant text.

You can learn more on how to make your own tokenizer [here](tokenizer.md)!

### How to talk with a model?

Now, one might wonder how to discuss with an LLM or use it as a chatbot. The idea is simple: fine-tune the model with conversations and dialogues so that the model learns to complete dialogues with coherent responses.

Let's design what we'll call a Chat Template, a dialogue format that the model will be fine-tuned with.

```
User: Hello, how are you?
Assistant: Fine, and you?
User: I'm doing great!
Assistant: Glad to hear!
```

With a model trained with such a format, it will be able to complete dialogues and give the illusion of talking to you.

Actually, there is something we haven't mentioned: there is always a token added to the sequence of tokens, usually called BOS for Beginning of String. This token essentially starts the engine, acting as the first token. The standard representation of this token is `<s>`. The string representation of this token is not actually seen by the tokenizer; instead, we prepend the token ID of the BOS to the rest of the sequence of tokens. With our previous example in mind, it would be something like:

```
BOS_ID + encode("User: Hello, how are you?
Assistant: Fine, and you?
User: I'm doing great!
Assistant: Glad to hear!")
```

Or, if we use their string representations only:
```
<s>User: Hello, how are you?
Assistant: Fine, and you?
User: I'm doing great!
Assistant: Glad to hear!
```

This results in a long sequence of token IDs starting with the BOS.

However, there are issues with this Chat Template. First, by using such a classic dialogue format, every time the user wants to discuss something similar to this format, the model will get confused. We are essentially forbidden to ask anything similar to this template. A solution could be to use very rare strings that are very unlikely to ever be used in a conversation to set the format of the chat template.

A second issue is how to stop the model. Currently, if we train the model like this, it will never stop completing the text unless you make it stop. A solution to this problem is to use what we call EOS, End of String. This token, when predicted, dictates that the generation should stop. We will represent it as `</s>`, similar to the BOS. The tokenizer never sees this string; the EOS_ID will be inserted each time we want the model to stop, in our case, after each Assistant response.

To summarize, we want:
- To use very rare strings to format and build our chat template. For example, we can encapsulate our User's instructions with `[INST]` and `[/INST]`.
- Insert EOS at the end of each Assistant response to tell the model when to stop.

With these ideas in mind, we can build a template like this:
```
BOS_ID
+ encode("[INST] Hello, how are you? [/INST] Fine, and you?") + EOS_ID
+ encode("[INST] I'm doing great! [/INST] Glad to hear!") + EOS_ID
```

Or, with their string representations only:
```
<s>[INST] Hello, how are you? [/INST] Fine, and you?</s>[INST] I'm doing great! [/INST] Glad to hear!</s>
```

While this is still not close to what we implement with [mistral-common](https://github.com/mistralai/mistral-common), many could stop here, as these are the bare minimum understanding requirements. However, there are a few hidden issues within tokenizers and LLMs that are crucial to understand, specifically **[boundary problems](boundaries.md)**.
