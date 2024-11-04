# Tokenization            

For the ground truth, please refer to [mistral-common](https://github.com/mistralai/mistral-common).

## Templates

| Tokenizer               | Based on                 |  Models                | Basic String Chat Template                                                 |
|:-------------------------:|:--------------------------:|:--------------------------:|:---------------------------------------------------------------------:|
| Tokenizer V1            | `sentencepiece`          | Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1, Mistral-7B-v0.2, Mistral-7B-Instruct-v0.2, Mixtral-8x7B-v0.1, Mixtral-8x7B-Instruct-v0.1, Mixtral-8x22B-v0.1          | `<s> [INST] Hello, how are you? [/INST] Fine, and you?</s> [INST] I'm doing great! [/INST] Glad to hear!</s>`          |
| Tokenizer V2         | `sentencepiece`          | *No models with weights available released*          | `<s>[INST] Hello, how are you?[/INST] Fine, and you?</s>[INST] I'm doing great![/INST] Glad to hear!</s>`            |
| Tokenizer V3         | `sentencepiece`          | Mixtral-8x22B-Instruct-v0.1, Mistral-7B-v0.3, Mistral-7B-Instruct v0.3, Codestral-22B-v0.1, Mathstral-7B-v0.3, Mamba-Codestral-7B-v0.1, Mistral-Large-123B-Instruct-2407, Mistral Small 22B Instruct 2407           | `<s>[INST] Hello, how are you?[/INST] Fine, and you?</s>[INST] I'm doing great![/INST] Glad to hear!</s>`            |
| Tokenizer V3-Tekken     | `tiktoken`               | Mistral-Nemo-12B-2407, Mistral-Nemo-12B-Instruct-2407, Pixtral-12B-2409, Ministral-8B-Instruct-2410         | `<s>[INST]Hello, how are you?[/INST]Fine, and you?</s>[INST]I'm doing great![/INST]Glad to hear!</s>`            |

## More

<details>

<summary><b>V1</b></summary>

### Tokenizer V1

The chat template for Tokenizer V1 is as follows:

```
<s> [INST] Hello, how are you? [/INST] Fine, and you?</s> [INST] I'm doing great! [/INST] Glad to hear!</s>
```
<sub><sup>With mistral-common, the system prompt is prepended to the first user message by default (feel free to customise it)</sup></sub>

**Jinja Template:**

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

**Encoding Code Sample:**

```python
BOS_ID
+ encode("[INST] Hello, how are you? [/INST]")
+ encode("Fine, and you?") + EOS_ID
+ encode("[INST] I'm doing great! [/INST]")
+ encode("Glad to hear!") + EOS_ID
```

</details>

<details>

<summary><b>V2/V3</b></summary>

### Tokenizer V2/V3

The basic chat template for Tokenizer V2 and V3 is as follows:

```
<s>[INST] Hello, how are you?[/INST] Fine, and you?</s>[INST] I'm doing great![/INST] Glad to hear!</s>
```
<sub><sup>With mistral-common, the system prompt is prepended to the last user message by default (feel free to customise it)</sup></sub>
<sub><sup>The main difference between V2 and V3 regards tool calling.</sup></sub>

**Jinja Template:**

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

**Encoding Code Sample:**

```python
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

For Tool Calling, please refer to [this section](tool_calling.md).

</details>

<details>

<summary><b>V3-Tekken</b></summary>

### Tokenizer V3-Tekken

The chat template for Tokenizer V3-Tekken is as follows:

```
<s>[INST]Hello, how are you?[/INST]Fine, and you?</s>[INST]I'm doing great![/INST]Glad to hear!</s>
```

**Jinja Template:**

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

**Encoding Code Sample:**

```python
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

For Tool Calling, please refer to [this section](tool_calling.md).

</details>
