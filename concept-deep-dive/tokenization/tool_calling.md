# Tokenization

## Tool Calling

Tool calling is a feature that allows the model to interact with external tools or APIs. The model is provided a list of tools it can pick from, the tool restul is then provided to the model. This can be useful for tasks that require information retrieval, computation, or other functionalities that the model cannot perform on its own. The tool calling feature is implemented using control tokens to ensure efficiency, security, and proper boundary handling.

### LLM - Recap
LLMs (Large Language Models) are predominantly auto-regressive models based on the transformer architecture. They are essentially text completion machines, converting a string into a sequence of token IDs that the model uses as input. The model then predicts the next token to complete the sequence. This process allows LLMs to generate coherent and contextually relevant text.

### Tokenization - Recap
Tokenization is the process of converting a string into a sequence of token IDs. Tokens can be characters, subwords, or symbols. The tokenizer breaks the string into tokens and converts them into IDs (encoding) and vice versa (decoding). The model then predicts the next token based on the sequence. For example, the sentence "Hello, how are" might be tokenized as: `Hello,` -> `432`, ` how` -> `523`, ` are` -> `87`. The model predicts the next token, such as `75` (decoded as ` you`), allowing it to generate coherent text.

### Control Tokens - Recap
Control tokens tackle efficiency, security, and boundary issues. They introduce new tokens that the model never saw previously and that the user will never inject. These tokens replace special strings, making the encoding process more efficient and secure. For example, instead of encapsulating user instructions with special strings, we use control token IDs directly.

### Tokenizer V2

The new tokenizer introduces control tokens for tool calling. These tokens include:

- `BEGIN_AVAILABLE_TOOLS` and `END_AVAILABLE_TOOLS`: Control tokens for the beginning and end of available tools.
- `BEGIN_TOOL_RESULTS` and `END_TOOL_RESULTS`: Control tokens for the beginning and end of tool results.
- `TOOL_CALLS`: Control token for tool calls.

#### Scenario: User Asks for a Calculation

Let's consider a scenario where a user asks the model to perform a calculation using an external tool.

**User Message:**  
```
What's 2+2?
```
**Tools:**  
```
[
    {
        "type": "function", 
        "function": {
            "name": "calculator",
            "description": "Performs mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to be done in python format."
                        }
                    }, 
                "required": ["operation"]
            }
        }
    }
]
```
**String Representation:**  
```
<s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "calculator", "description": "Performs mathematical calculations", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "description": "The operation to be done in python format."}}, "required": ["operation"]}}}][/AVAILABLE_TOOLS][INST] What's 2+2?[/INST]
```

**Assistant Tool Call:**  
```
[{"name": "calculator", "arguments": {"operation": "2+2", "format": "celsius"}}]
```

**Tool Call String Representation:**
```
<s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "calculator", "description": "Performs mathematical calculations", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "description": "The operation to be done in python format."}}, "required": ["operation"]}}}][/AVAILABLE_TOOLS][INST] What's 2+2?[/INST][TOOL_CALLS] [{"name": "calculator", "arguments": {"operation": "2+2", "format": "celsius"}}]</s>
```
**Tool Result:**  
```
[{"name": "calculator", "content": 4}]
```

**Tool Result String Representation:**
```
<s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "calculator", "description": "Performs mathematical calculations", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "description": "The operation to be done in python format."}}, "required": ["operation"]}}}][/AVAILABLE_TOOLS][INST] What's 2+2?[/INST][TOOL_CALLS] [{"name": "calculator", "arguments": {"operation": "2+2", "format": "celsius"}}]</s>[TOOL_RESULTS] [{"name": "calculator", "content": 4}][/TOOL_RESULTS]
```

**Final Assistant Response:**
```
2+2=4
```

**Final String Representation:**
```
<s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "calculator", "description": "Performs mathematical calculations", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "description": "The operation to be done in python format."}}, "required": ["operation"]}}}][/AVAILABLE_TOOLS][INST] What's 2+2?[/INST][TOOL_CALLS] [{"name": "calculator", "arguments": {"operation": "2+2", "format": "celsius"}}]</s>[TOOL_RESULTS] [{"name": "calculator", "content": 4}][/TOOL_RESULTS] 2+2=4</s>
```

In this scenario, the control tokens `TOOL_CALLS`, `BEGIN_TOOL_RESULTS`, and `END_TOOL_RESULTS` are used to encode the tool call and the tool result. The model can then generate a coherent response based on the tool result.

### Tokenizer V3

The tokenizer V3 is overall similar to V2, but introduces a slightly different way of encoding tool messages. The main differences are:

1. **Tool Results Encoding:** In V3, tool results are not wrapped in a list, and the history of tool calls is tokenized as well.
2. **Tool Call Encoding:** In V3, tool calls include `id` field, which can be used to track the history of tool calls.

#### Scenario: User Asks for a Calculation

Let's consider a scenario where a user asks the model to perform a calculation using an external tool.

**User Message:**  
```
What's 2+2?
```
**Tools:**  
```
[
    {
        "type": "function", 
        "function": {
            "name": "calculator",
            "description": "Performs mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to be done in python format."
                        }
                    }, 
                "required": ["operation"]
            }
        }
    }
]
```
**String Representation:**  
```
<s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "calculator", "description": "Performs mathematical calculations", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "description": "The operation to be done in python format."}}, "required": ["operation"]}}}][/AVAILABLE_TOOLS][INST] What's 2+2?[/INST]
```

**Assistant Tool Call:**  
```
[{"name": "calculator", "arguments": {"operation": "2+2", "format": "celsius"}, "id": "VvvODy9mT"}]
```

**Tool Call String Representation:**
```
<s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "calculator", "description": "Performs mathematical calculations", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "description": "The operation to be done in python format."}}, "required": ["operation"]}}}][/AVAILABLE_TOOLS][INST] What's 2+2?[/INST][TOOL_CALLS] [{"name": "calculator", "arguments": {"operation": "2+2", "format": "celsius"}, "id": "VvvODy9mT"}]</s>
```
**Tool Result:**  
```
[{"name": "calculator", "content": 4}]
```

**Tool Result String Representation:**
```
<s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "calculator", "description": "Performs mathematical calculations", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "description": "The operation to be done in python format."}}, "required": ["operation"]}}}][/AVAILABLE_TOOLS][INST] What's 2+2?[/INST][TOOL_CALLS] [{"name": "calculator", "arguments": {"operation": "2+2", "format": "celsius"}, "id": "VvvODy9mT"}]</s>[TOOL_RESULTS] {"content": 4, "call_id": "VvvODy9mT"}[/TOOL_RESULTS]
```

**Final Assistant Response:**
```
2+2=4
```

**Final String Representation:**
```
<s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "calculator", "description": "Performs mathematical calculations", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "description": "The operation to be done in python format."}}, "required": ["operation"]}}}][/AVAILABLE_TOOLS][INST] What's 2+2?[/INST][TOOL_CALLS] [{"name": "calculator", "arguments": {"operation": "2+2", "format": "celsius"}, "id": "VvvODy9mT"}]</s>[TOOL_RESULTS] {"content": 4, "call_id": "VvvODy9mT"}[/TOOL_RESULTS] 2+2=4</s>
```

In this scenario, the control tokens `TOOL_CALLS`, `BEGIN_TOOL_RESULTS`, and `END_TOOL_RESULTS` are used to encode the tool call and the tool result. The model can then generate a coherent response based on the tool result.

### Tokenizer V3 - Tekken
The main difference with tekken is once more whitespacing, its overall the same tokenization, the only difference being that its not based on `sentencepiece`, the final string representation of the previous scenario would look like so:
```
<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "calculator", "description": "Performs mathematical calculations", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "description": "The operation to be done in python format."}}, "required": ["operation"]}}}][/AVAILABLE_TOOLS][INST]What's 2+2?[/INST][TOOL_CALLS][{"name": "calculator", "arguments": {"operation": "2+2", "format": "celsius"}, "id": "VvvODy9mT"}]</s>[TOOL_RESULTS]{"content": 4, "call_id": "VvvODy9mT"}[/TOOL_RESULTS]2+2=4</s>
```