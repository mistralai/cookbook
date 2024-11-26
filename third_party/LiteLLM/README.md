# Use LiteLLM Proxy to Call Mistral AI API

Use [LiteLLM Proxy](https://docs.litellm.ai/docs/simple_proxy) for:
- Calling 100+ LLMs Mistral AI, OpenAI, Azure, Vertex, Bedrock/etc. in the OpenAI ChatCompletions & Completions format
- Track usage + set budgets with Virtual Keys

Works for [Mistral AI API](https://docs.litellm.ai/docs/providers/mistral) + [Codestral API](https://docs.litellm.ai/docs/providers/codestral) + [Bedrock](https://docs.litellm.ai/docs/providers/bedrock)

## Sample Usage

### Step 1. Create a Config for LiteLLM proxy

LiteLLM Requires a config with all your models define - we can call this file `litellm_config.yaml`

[Detailed docs on how to setup litellm config - here](https://docs.litellm.ai/docs/proxy/configs)

```yaml
model_list:
  - model_name: mistral-small-latest ### MODEL Alias ###
    litellm_params: # all params accepted by litellm.completion() - https://docs.litellm.ai/docs/completion/input
      model: mistral/mistral-small-latest ### MODEL NAME sent to `litellm.completion()` ###
      api_key: "os.environ/MISTRAL_API_KEY" # does os.getenv("MISTRAL_API_KEY")
  - model_name: mistral-nemo
    litellm_params: 
      model: mistral/open-mistral-nemo 
      api_key: "os.environ/MISTRAL_API_KEY"

```

### Step 2. Start litellm proxy

```shell
docker run \
    -v $(pwd)/litellm_config.yaml:/app/config.yaml \
    -e MISTRAL_API_KEY=<your-mistral-api-key>
    -p 4000:4000 \
    ghcr.io/berriai/litellm:main-latest \
    --config /app/config.yaml --detailed_debug
```

### Step 3. Test it! 

[Use with Langchain, LlamaIndex, Instructor, etc.](https://docs.litellm.ai/docs/proxy/user_keys)



## Basic Chat Completion

```python
import os

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

client = MistralClient(
  api_key="sk-1234",             # set api_key to litellm proxy virtual key
  endpoint="http://0.0.0.0:4000" # set endpoint to litellm proxy endpoint
)
chat_response = client.chat(
    model="mistral-small-latest",
    messages=[
        {"role": "user", "content": "this is a test request, write a short poem"}
    ],
)
print(chat_response.choices[0].message.content)

```

## Tool Use
```python
import os

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

client = MistralClient(
  api_key="sk-1234",             # set api_key to litellm proxy virtual key
  endpoint="http://0.0.0.0:4000" # set endpoint to litellm proxy endpoint
)

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          },
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
      },
    }
  }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
completion = client.chat.completions.create(
  model="mistral-small-latest",
  messages=messages,
  tools=tools,
  tool_choice="auto"
)

print(completion)

```

## Vision Example

```python

import os

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

client = MistralClient(
  api_key="sk-1234",             # set api_key to litellm proxy virtual key
  endpoint="http://0.0.0.0:4000" # set endpoint to litellm proxy endpoint
)

response = client.chat.completions.create(
    model="mistral-small-latest",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                }
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])
```

## Supported Mistral AI API Models

**ALL MODELS SUPPORTED**. 

Just add `mistral/` to the beginning of the model name.

Example models: 
| Model Name     | Usage                                              |
|----------------|--------------------------------------------------------------|
| Mistral Small  | `completion(model="mistral/mistral-small-latest", messages)` |
| Mistral Medium | `completion(model="mistral/mistral-medium-latest", messages)`|
| Mistral Large  | `completion(model="mistral/mistral-large-latest", messages)` |
| Mistral 7B     | `completion(model="mistral/open-mistral-7b", messages)`      |
| Mixtral 8x7B   | `completion(model="mistral/open-mixtral-8x7b", messages)`    |
| Mixtral 8x22B  | `completion(model="mistral/open-mixtral-8x22b", messages)`   |
| Codestral      | `completion(model="mistral/codestral-latest", messages)`     |
| Mistral NeMo      | `completion(model="mistral/open-mistral-nemo", messages)`     |
| Mistral NeMo 2407      | `completion(model="mistral/open-mistral-nemo-2407", messages)`     |
| Codestral Mamba      | `completion(model="mistral/open-codestral-mamba", messages)`     |
| Codestral Mamba    | `completion(model="mistral/codestral-mamba-latest"", messages)`     |


## Supported Bedrock Mistral AI Models
| Model Name       | Usage                   |
|----------------|--------------------------------------------------------------|
| Mistral 7B Instruct        | `completion(model='bedrock/mistral.mistral-7b-instruct-v0:2', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| Mixtral 8x7B Instruct      | `completion(model='bedrock/mistral.mixtral-8x7b-instruct-v0:1', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
