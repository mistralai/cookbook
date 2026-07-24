# Demystifying Mistral's Instruct Tokenization & Chat Templates

> [!IMPORTANT]
> This document is deprecated. Please take a look at the full, in-depth documentation [here](README.md).

From the original tokenizer V1 to the most recent V3 and Tekken tokenizers, Mistral's tokenizers have undergone subtle changes related to how to tokenize for the instruct models. We've come a long way in optimization and research to find the best approach. Today, we'll delve into these tokenizers, demystify any sources of debate, and explore how they work, the proper chat templates to use for each one, and their story within the community!

In this article, we will mostly delve into instruction tokenization and chat templates for simple instruction following. We won't dig into function calling or fill in the middle.

The ground truth for all the information available here can be found by exploring the repos on hugging face as well as on github, specifically [mistral_common](https://github.com/mistralai/mistral-common).

<details>

<summary><b>TL;DR</b></summary>

### Tokenizer V1:
```
"<s> [INST] user message [/INST] assistant message</s> [INST] new user message [/INST]"
```
<sub><sup>With mistral-common, the system prompt is prepended to the first user message by default (feel free to customise it)</sup></sub>

### Tokenizer V2:
```
"<s>[INST] user message[/INST] assistant message</s>[INST] new user message[/INST]"
```
<sub><sup>With mistral-common, the system prompt is prepended to the last user message by default (feel free to customise it)</sup></sub>

**FIM**  
```
"<s>[SUFFIX]suffix[PREFIX] prefix"
```

### Tokenizer V3:
```
"<s>[INST] user message[/INST] assistant message</s>[INST] new user message[/INST]"
```
<sub><sup>V3 is highly similar to V2, the only difference concerns function calling.</sup></sub>

**FIM**  
```
"<s>[SUFFIX]suffix[PREFIX] prefix"
```

### Tokenizer V3 - Tekken (Nemo):
```
"<s>[INST]user message[/INST]assistant message</s>[INST]new user message[/INST]"
```
<sub><sup>With mistral-common, the system prompt is prepended to the last user message by default (feel free to customise it)</sup></sub>

EDIT: New document explaining can be found in our [cookbooks](https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/chat_templates.md)

</details>

## Tokenizer V1

The very first releases, Mistral 7B V1 and V2 as well as Mixtral 8x7B V1, were met with a lot of appreciation and love from the community. However, this did not prevent some disagreements and debates within the community regarding the correct chat templates for the instruct tokenizers. Today, you can reliably rely on `mistral_common` as the ground truth for the tokenization process, but back then, most of the community relied on the available Jinja chat templates!

The community was very quick to notice inconsistencies around the tokenizers and the chat templates, achieving better results with custom versions that adjusted the whitespaces surrounding special strings. It's amazing how the community is very fast and helpful in detecting any issues and bugs related to the models!

The tokenizer used for these first releases, based on `sentencepiece`, uses a similar template to the old Llama models. This tokenizer deals with instruct messages as follows:

### Instruct Chat Template

The chat template was, as previously mentioned, highly similar to the first Llama template, and would result in strings like the following.

```
<s> [INST] user message [/INST] assistant message</s> [INST] new user message [/INST]
```

Here, the only special strings were `[INST]` to start the user message and `[/INST]` to end the user message, making way for the assistant's response. The BOS (beginning of string) was and still is represented with `<s>`, and the EOS (end of string) is `</s>`, used at the end of each completion, terminating any assistant message. 

**The whitespaces are of extreme importance.**

Note that `<s>` and `</s>` are more like strings representing the BOS and EOS than actual strings. Their corresponding IDs are `1` and `2`.

> ⚠️ This template will always cause the model to output a token that starts with a whitespace, always having a leading space. Depending on your inference engine, it may or may not strip that whitespace by default. Hence, it is essential to ensure that a whitespace (and a single one) is present after each assistant response when using the template.

The Jinja template for the HuggingFace `transformers` for this kind of string could look something like this:

```jinja
{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}
```

You can see the current one being used by default by taking a look at the [tokenizer_config.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json#L33).

Here it is formatted to be easier to read, making it easier to understand how they work:
```jinja
{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ ' [INST] ' + message['content'] + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' ' + message['content'] + eos_token }}
    {% else %}
        {{ raise_exception('Only user and assistant roles are supported!') }}
    {% endif %}
{% endfor %}
```

> ⚠️ As you can see, each `message['content']` is always preceded by a whitespace. This assumes that `message['content']` has the first whitespace manually removed by you or your inference library.

If you want to apply the template yourself manually, you can do it in Python:

```py
from jinja2 import Template, Environment, exceptions

JINJA_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

def raise_exception(message):
    raise ValueError(message)

def apply_jinja_template(messages, bos_token='<s>', eos_token='</s>'):
    template_str = JINJA_TEMPLATE

    env = Environment()
    env.globals['raise_exception'] = raise_exception

    template = env.from_string(template_str)

    try:
        result = template.render(messages=messages, bos_token=bos_token, eos_token=eos_token)
        return result
    except exceptions.TemplateError as e:
        raise ValueError(f"Template rendering error: {e}")

print(apply_jinja_template(messages))
```

### Instruct Tokenization Logic

Now let's dig into the original logic behind the tokenization process that you can see in `mistral_common`. You often would not simply provide the text to the tokenizer and encode the entire string as it is, `mistral-common` actually goes directly from `request` -> `int`, while other approaches such as the one used by `transformers` is closer to `request` -> `str` -> `int`. The way it is actually done for the tokenization process is closer to the following logic:

- **Tokenize each message:**  
  You would have user messages and assistant messages. You would take them separately as individual strings, with the user message encapsulated with the special strings, and encode them. In the previous example, this would be something like:  
  `encode("[INST] user message [/INST]")`, `encode("assistant message")`, and `encode("[INST] new user message [/INST]")`, and so on...

- **Concatenate:**  
  Once each message is tokenized and encoded, you would concatenate them all, prepending them with the BOS ID and separating them by pairs with the EOS ID:  
  `BOS_ID + encode("[INST] user message [/INST]") + encode("assistant message") + EOS_ID + encode("[INST] new user message [/INST]")`

You might notice that if followed step by step, this would actually match with a string and whitespaces added like so:
```
<s>[INST]_user message_[/INST]_assistant message</s>[INST]_new user message_[/INST]
```
<sub><sup>"_" represents the added whitespaces when encapsulating with the special strings</sup></sub>

If compared to the chat template mentioned above, this misses some whitespaces after the BOS and EOS. Where do they come from then? They come from `sentencepiece` when encoding. It will, by default, start with a leading whitespace. So every time we `encode`, we are adding a leading whitespace. Therefore:
- `encode("[INST] user message [/INST]")` is actually encoding `encode(" [INST] user message [/INST]")`

This adds the leading whitespaces, making the final string look more like:
```
<s>_[INST] user message [/INST] assistant message</s>_[INST] new user message [/INST]
```
<sub><sup>"_" represents the new added whitespaces by sentencepiece</sup></sub>

Finally, token per token it will give the following:

<div>
<table style="border-collapse: collapse; width: 100%; text-align: center;">
  <tr>
    <td style="white-space: nowrap;">&lt;s&gt;</td>
    <td style="white-space: nowrap;">_[</td>
    <td style="white-space: nowrap;">INST</td>
    <td style="white-space: nowrap;">]</td>
    <td style="white-space: nowrap;">_user</td>
    <td style="white-space: nowrap;">_message</td>
    <td style="white-space: nowrap;">_[</td>
    <td style="white-space: nowrap;">/</td>
    <td style="white-space: nowrap;">INST</td>
    <td style="white-space: nowrap;">]</td>
    <td style="white-space: nowrap;">_assistant</td>
    <td style="white-space: nowrap;">_message</td>
    <td style="white-space: nowrap;">&lt;/s&gt;</td>
    <td style="white-space: nowrap;">_[</td>
    <td style="white-space: nowrap;">INST</td>
    <td style="white-space: nowrap;">]</td>
    <td style="white-space: nowrap;">_new</td>
    <td style="white-space: nowrap;">_user</td>
    <td style="white-space: nowrap;">_message</td>
    <td style="white-space: nowrap;">_[</td>
    <td style="white-space: nowrap;">/</td>
    <td style="white-space: nowrap;">INST</td>
    <td style="white-space: nowrap;">]</td>
  </tr>
  <tr>
    <td>1</td>
    <td>733</td>
    <td>16289</td>
    <td>28793</td>
    <td>2188</td>
    <td>2928</td>
    <td>733</td>
    <td>28748</td>
    <td>16289</td>
    <td>28793</td>
    <td>13892</td>
    <td>2928</td>
    <td>2</td>
    <td>733</td>
    <td>16289</td>
    <td>28793</td>
    <td>633</td>
    <td>2188</td>
    <td>2928</td>
    <td>733</td>
    <td>28748</td>
    <td>16289</td>
    <td>28793</td>
  </tr>
</table>
</div>

<sub><sup>"_" represents any whitespace</sup></sub>

### System Prompt

The first releases did not have a specific methodology for system prompts. By default, we usually prepend the system prompt to the first user message, followed by two new lines:
```
<s> [INST] system prompt

user message [/INST] assistant message</s> [INST] new user message [/INST]
```

## Tokenizer V2

Following the previous tokenizer, and mostly used by the proprietary models, the second version of the tokenizer powered models such as the now old Mistral Small 2402 and Mistral Large 2402. This new tokenizer introduced control tokens, specifically for the previous special strings `[INST]` and `[/INST]`, which became control tokens with the corresponding IDs being `3` and `4`, but also new control tokens for function calling.

### Instruct Chat Template

Due to the introduction of the control tokens, the chat template changed slightly due to the absence of some whitespaces.

```
<s>[INST] user message[/INST] assistant message</s>[INST] new user message[/INST]
```

For comparison, here is a string with the previous chat template of the previous tokenizer:

```
<s> [INST] user message [/INST] assistant message</s> [INST] new user message [/INST]
```

> ⚠️ This template will always cause the model to output a token that starts with a whitespace, always having a leading space. Depending on your inference engine, it may or may not strip that whitespace by default. Hence, it is essential to ensure that a whitespace (and a single one) is present after each assistant response when using the template.

The Jinja Template for the HuggingFace `transformers` for this kind of string could look something like this:

```jinja
{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + '[/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}
```

After being formatted to be easier to read:
```jinja
{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ '[INST] ' + message['content'] + '[/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' ' + message['content'] + eos_token }}
    {% else %}
        {{ raise_exception('Only user and assistant roles are supported!') }}
    {% endif %}
{% endfor %}
```

> ⚠️ As you can see, each `message['content']` is always preceded by a whitespace. This assumes that `message['content']` has the first whitespace manually removed by you or your inference library.

If you want to apply the template yourself manually, you can do it in Python:

```py
from jinja2 import Template, Environment, exceptions

JINJA_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + '[/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

def raise_exception(message):
    raise ValueError(message)

def apply_jinja_template(messages, bos_token='<s>', eos_token='</s>'):
    template_str = JINJA_TEMPLATE

    env = Environment()
    env.globals['raise_exception'] = raise_exception

    template = env.from_string(template_str)

    try:
        result = template.render(messages=messages, bos_token=bos_token, eos_token=eos_token)
        return result
    except exceptions.TemplateError as e:
        raise ValueError(f"Template rendering error: {e}")

print(apply_jinja_template(messages))
```

**FIM**  
For FIM, the template would be as follows:
```
<s>[SUFFIX]suffix[PREFIX] prefix
```

### Instruct Tokenization Logic

Now that we have control tokens, the process to tokenize has changed slightly. Here is the new logic behind the chat template:

- **Tokenize each message:**  
  You have user messages and assistant messages. You now take them separately as individual strings and encode them:  
  `encode(user_message)`, `encode(assistant_message)`, and `encode(new_user_message)`, and so on...

- **Concatenate:**  
  Once each message is tokenized and encoded, you would concatenate them all, prepending them with the BOS ID and separating them by pairs with the EOS ID while encapsulating the user messages with the control tokens, specifically `[INST]` ID and `[/INST]` ID:  
  `BOS_ID + INST_ID + encode("user message") + /INST_ID + encode("assistant message") + EOS_ID + INST_ID + encode("new user message") + /INST_ID`

Following this, the string would be something like so:
```
<s>[INST]user message[/INST]assistant message</s>[INST]new user message[/INST]
```

However, once more, this misses some whitespaces, this time after the control tokens. They come from the same place as before, from `sentencepiece`:
- `encode("user message")` is actually encoding `encode(" user message")`
- `encode("assistant message")` is actually encoding `encode(" assistant message")`
- `encode("new user message")` is actually encoding `encode(" new user message")`

This adds the missing whitespaces, making the final string look more like:
```
<s>[INST]_user message[/INST]_assistant message</s>[INST]_new user message[/INST]
```
<sub><sup>"_" represents the new added whitespaces by sentencepiece</sup></sub>

Finally, token per token it will give the following:

<table style="border-collapse: collapse; width: 100%; text-align: center;">
  <tr>
    <td style="white-space: nowrap;">&lt;s&gt;</td>
    <td style="white-space: nowrap;">[INST]</td>
    <td style="white-space: nowrap;">_user</td>
    <td style="white-space: nowrap;">_message</td>
    <td style="white-space: nowrap;">[/INST]</td>
    <td style="white-space: nowrap;">_assistant</td>
    <td style="white-space: nowrap;">_message</td>
    <td style="white-space: nowrap;">&lt;/s&gt;</td>
    <td style="white-space: nowrap;">[INST]</td>
    <td style="white-space: nowrap;">_new</td>
    <td style="white-space: nowrap;">_user</td>
    <td style="white-space: nowrap;">_message</td>
    <td style="white-space: nowrap;">[/INST]</td>
  </tr>
  <tr>
    <td>1</td>
    <td>3</td>
    <td>2956</td>
    <td>3696</td>
    <td>4</td>
    <td>14660</td>
    <td>3696</td>
    <td>2</td>
    <td>3</td>
    <td>1401</td>
    <td>2956</td>
    <td>3696</td>
    <td>4</td>
  </tr>
</table>
<sub><sup>"_" represents any whitespace</sup></sub>

### System Prompt

The second version of the tokenizer implements a slightly different system prompt approach. By default, it prepends it to the last user message:
```
<s>[INST] user message[/INST] assistant message</s>[INST] system prompt

new user message[/INST]
```

> ⚠️ This is how `mistral_common` and the templates implement system prompts, but this can easily be customized. Feel free to use system prompts in different places, such as the second from the last or simply as the first user message, as before.

## Tokenizer V3
This tokenizer powers models such as Mixtral 8x22B, Codestral 22B, Mathstral 7B, Mamba Codestral 7B, Small 2409 and Large 2 (Large 2407). It is highly similar to the second version, with only tool use being slightly different.

The chat template, tokenization, and system prompt for basic instruct are the same as the previous one.

### Tekken
Tekken is a different version of the V3 tokenizer and powers Mistral Nemo 12B and Pixtral 12B. While the original one and previous tokenizers were based on `sentencepiece`, Tekken is based on `tiktoken`. With a considerably larger vocabulary size, it also deals with encoding differently. The main difference for the chat template is that it does not prepend a whitespace like `sentencepiece`.

This results in a simpler chat template and more intuitive tokenization.

### Tekken Instruct Chat Template
With even fewer intrusive whitespaces, the new template is as simple as the following:
```
<s>[INST]user message[/INST]assistant message</s>[INST]new user message[/INST]
```

For comparison, here is the chat template of the previous tokenizer:

```
<s>[INST] user message[/INST] assistant message</s>[INST] new user message[/INST]
```

The Jinja template for the HuggingFace `transformers` for this kind of string could look something like this:

```jinja
{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST]' + message['content'] + '[/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}
```

Formatted to be easier to read:
```jinja
{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ '[INST]' + message['content'] + '[/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ message['content'] + eos_token }}
    {% else %}
        {{ raise_exception('Only user and assistant roles are supported!') }}
    {% endif %}
{% endfor %}
```

If you want to apply the template yourself manually, you can do it in Python:

```py
from jinja2 import Template, Environment, exceptions

JINJA_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST]' + message['content'] + '[/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

def raise_exception(message):
    raise ValueError(message)

def apply_jinja_template(messages, bos_token='<s>', eos_token='</s>'):
    template_str = JINJA_TEMPLATE

    env = Environment()
    env.globals['raise_exception'] = raise_exception

    template = env.from_string(template_str)

    try:
        result = template.render(messages=messages, bos_token=bos_token, eos_token=eos_token)
        return result
    except exceptions.TemplateError as e:
        raise ValueError(f"Template rendering error: {e}")

print(apply_jinja_template(messages))
```

### Tekken Instruct Tokenization Logic
The logic for the tokenization is still the same as the previous one, which is:

- **Tokenize each message:**  
  You have user messages and assistant messages. You now take them separately as individual strings and encode them:  
  `encode(user_message)`, `encode(assistant_message)`, and `encode(new_user_message)`, and so on...

- **Concatenate:**  
  Once each message is tokenized and encoded, you would concatenate them all, prepending them with the BOS ID and separating them by pairs with the EOS ID while encapsulating the user messages with the control tokens, specifically `[INST]` ID and `[/INST]` ID:  
  `BOS_ID + INST_ID + encode("user message") + /INST_ID + encode("assistant message") + EOS_ID + INST_ID + encode("new user message") + /INST_ID`

And that is all. The resulting string is more intuitive, with no whitespaces added since it has a different behavior from `sentencepiece`, making the final string the following:
```
<s>[INST]user message[/INST]assistant message</s>[INST]new user message[/INST]
```

Token per token::

<table style="border-collapse: collapse; width: 100%; text-align: center;">
  <tr>
    <td style="white-space: nowrap;">&lt;s&gt;</td>
    <td style="white-space: nowrap;">[INST]</td>
    <td style="white-space: nowrap;">user</td>
    <td style="white-space: nowrap;">_message</td>
    <td style="white-space: nowrap;">[/INST]</td>
    <td style="white-space: nowrap;">ass</td>
    <td style="white-space: nowrap;">istant</td>
    <td style="white-space: nowrap;">_message</td>
    <td style="white-space: nowrap;">&lt;/s&gt;</td>
    <td style="white-space: nowrap;">[INST]</td>
    <td style="white-space: nowrap;">new</td>
    <td style="white-space: nowrap;">_user</td>
    <td style="white-space: nowrap;">_message</td>
    <td style="white-space: nowrap;">[/INST]</td>
  </tr>
  <tr>
    <td>1</td>
    <td>3</td>
    <td>3263</td>
    <td>5117</td>
    <td>4</td>
    <td>1503</td>
    <td>19464</td>
    <td>5117</td>
    <td>2</td>
    <td>3</td>
    <td>3080</td>
    <td>3330</td>
    <td>5117</td>
    <td>4</td>
  </tr>
</table>
<sub><sup>"_" represents any whitespace</sup></sub>

### System Prompt

By default, the system prompt is handled similarly to the previous versions:
```
<s>[INST]user message[/INST]assistant message</s>[INST]system prompt

new user message[/INST]
```

## Conclusion

The tokenizers have evolved significantly from V1 to V3 and the Tekken variant, each iteration bringing improvements and changes that have sometimes led to debates and issues within the community. Understanding the nuances of these tokenizers is crucial for optimizing the performance of Mistral models in various applications and properly fine-tuning.

### Key Takeaways:

1. **Tokenizer V1**:
   - Used `sentencepiece` and a template similar to Llama models.
   - Special strings `[INST]` and `[/INST]` with whitespace handling.
   - System prompt prepended to the first user message with two new lines.

2. **Tokenizer V2**:
   - Introduced control tokens for `[INST]` and `[/INST]` as well as other control tokens.
   - Slightly adjusted chat template due to the absence of some whitespaces.
   - System prompt prepended to the last user message.

3. **Tokenizer V3**:
   - Similar to V2 but improved tool use.
   - Same chat template, tokenization, and system prompt approach as V2.

4. **Tokenizer V3 - Tekken**:
    - Based on `tiktoken` with a larger vocabulary.
    - Simpler chat template with no leading whitespaces.

### Best Practices:

- **Whitespace Handling**: Pay close attention to whitespace handling, especially with the `sentencepiece`-based tokenizers, as they prepend leading whitespaces.
- **Control Tokens**: Use control tokens correctly in V2 and V3 tokenizers to ensure proper encapsulation of user messages.
- **System Prompts**: Customize system prompts as needed, but be aware of the default placements in different tokenizer versions.

### Community and Resources:

- **Community Contributions**: The community has been instrumental in identifying and resolving inconsistencies and bugs, showcasing the collaborative nature of open-source development. Join Mistral's [discord server](https://discord.gg/mistralai) and exchange with community members!
- **Ground Truth**: For the most accurate and up-to-date information, refer to the [mistral_common](https://github.com/mistralai/mistral-common) repository on GitHub and the [documentation](https://docs.mistral.ai/).
