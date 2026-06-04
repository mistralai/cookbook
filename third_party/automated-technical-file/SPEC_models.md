# Mistral AI Model & SDK Specification (2026)

This document records the exact model IDs and SDK signatures for the Mistral AI features used in the Automated Technical File (ATF) cookbook demo.

## 1. Core Model IDs

| Feature | Model ID (Latest) | Specific Version (Stable) |
| :--- | :--- | :--- |
| **Agents / Reasoning** | `mistral-large-latest` | `mistral-large-2411` |
| **Structured Outputs** | `mistral-large-latest` | `mistral-large-2411` |
| **Efficient Reasoning** | `mistral-small-latest` | `mistral-small-2503` |
| **On-Device (Edge)** | `ministral-3b-latest` | `ministral-3b-2410` |
| **Document OCR** | `mistral-ocr-latest` | `mistral-ocr-2505` |
| **STT (Transcription)** | `voxtral-mini-latest` | `voxtral-mini-2602` |
| **TTS (Speech)** | `voxtral-tts-latest` | `voxtral-mini-tts-2603` |

## 2. SDK Signatures (Python)

### Initialization
```python
from mistralai import Mistral
client = Mistral(api_key="MISTRAL_API_KEY")
```

### Agents API
```python
# Create an agent
agent = client.agents.create(
    name="ATF Assistant",
    model="mistral-large-latest",
    instructions="Process technical files.",
    tools=[{"type": "document_library", "library_ids": ["lib_id"]}]
)

# Completion
response = client.agents.complete(
    agent_id="ag_id",
    messages=[{"role": "user", "content": "Query"}]
)
```

### Document Library (Beta)
```python
# Create library
lib = client.beta.libraries.create(name="ATF_Corpus")

# Upload document
with open("file.pdf", "rb") as f:
    doc = client.beta.libraries.documents.upload(
        library_id="lib_id",
        file=f,
        purpose="document_library"
    )
```

### Structured Outputs
```python
from pydantic import BaseModel

class Analysis(BaseModel):
    patterns: list[str]
    anomalies: list[str]

response = client.chat.parse(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Analyze log slice"}],
    response_format=Analysis
)
```

### Voxtral STT (Transcribe 2)
```python
with open("audio.mp3", "rb") as f:
    response = client.audio.transcriptions.complete(
        model="voxtral-mini-latest",
        file={"file_name": "audio.mp3", "content": f.read()},
        diarize=True
    )
```

### Voxtral TTS
```python
response = client.audio.speech.complete(
    model="voxtral-tts-latest",
    input="Text to speak",
    voice_id="en_paul_neutral",
    response_format="mp3"
)
# Zero-shot cloning: pass base64 to 'ref_audio'
```

