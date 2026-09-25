# Build a real-time audio fact-checking agent
As you speak into the microphone, this agent transcribes your speech, checks
what you say against a source of truth, and reports its conclusions.

This project uses a Python server for the logic and to access LLMs, plus a small
web UI to capture microphone input and display results. Follow the instructions
here to create the files you need - or visit [the GitHub repo](https://github.com/mistralai/cookbook/tree/main/mistral/audio/realtime-fact-checker) if
that's easier.

Here's how it works:

1. On startup, the server generates embeddings for each fact in a source of truth.
1. The web app uses an AudioWorklet to read audio from the microphone and streams it to
   your server over a WebSocket.
1. The server forwards audio to a real-time transcription model and watches the
   transcript for complete sentences.
1. Each sentence is checked to see if it contains a factual claim. If so,
   this claim is embedded and matched against the embeddings for the source of truth.
1. If a close-enough match is found, an LLM judges the assertion against it
   and returns a verdict plus its reasoning, which is pushed back to the browser and
   rendered as a card.

## Prerequisites

To complete this cookbook, you will need:
- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/), or your favorite package manager
- A Mistral API key
- A browser with microphone access

## Environment setup

### Install
#### Python
Install the packages globally with `pip`, or create a project with `uv` and add the packages you'll need:
create a project with `uv` and add the packages you'll need:

```sh
uv init
uv add fastapi "mistralai[realtime]" numpy python-dotenv "uvicorn[standard]"
```

This creates a `pyproject.toml` listing these dependencies and a `uv.lock`
pinning their exact versions. Any `uv run` command syncs your environment
automatically. Or, sync it yourself:

```sh
uv sync
```

#### JavaScript
This project uses vanilla, plain JavaScript. No build process or packages needed!

### Required environment variables
To complete this cookbook, you'll need a Mistral API key. In [Studio](https://console.mistral.ai), navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys) and create a new API key.

Create a `.env` at the root of your project and add your Mistral API key:

```
MISTRAL_API_KEY=your-mistral-api-key
```

## Create the server
Create `server.py` in your project directory.

```sh
touch server.py
```

Open `server.py` and add these imports from built-in Python libraries.

```py
import asyncio
import json
import os
import re
```

Next, add imports from the packages you installed above.
To enable streaming audio transcription, include `WebSocket` objects from `FastAPI`,
as well as the `Transcription` events Mistral's Realtime model emits.

```py
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from mistralai.client import Mistral
from mistralai.client.models import (
    AudioFormat,
    RealtimeTranscriptionError,
    TranscriptionStreamDone,
    TranscriptionStreamTextDelta,
)
```

Finally, load in `MISTRAL_API_KEY` from the `.env` file you just made.
```py
load_dotenv()
```

### Define constants

Set the names of the models you wish to use.

```py
EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-medium-3.5"
TRANSCRIPTION_MODEL = "voxtral-mini-transcribe-realtime-2602"
```

Choose a threshold for cosine similarity. Below this value, determine that a
claim is unrelated to any fact in the source of truth.

```py
MATCH_THRESHOLD = 0.5
```

Use this regular expression to determine the end of a sentence.

More sophisticated ways exist to evaluate whether a chunk of text contains a
complete sentence - for example, NLP-based systems like [NLTK's sentence
tokenizer](https://www.nltk.org/api/nltk.tokenize.html) and
[spaCy](https://spacy.io/). For this demo, a regular expression to seek sentence
ends is sufficient - even though this will fail, for example, on input which
includes abbreviations like "St. James."

```py
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
```

The `FACTS` list is the source of truth.
One might also find these on the Internet or elsewhere, but even in a real-world scenario,
for speed, you'd probably want to source such facts in advance.

```py
FACTS = [
    "Venus rotates once every 243 Earth days and completes one orbit around the Sun in 225 Earth days",
    "The mythical unicorn is Scotland's official national animal",
    "France is slightly larger than Spain in total area - about 551,500 km², compared to 505,400 km² for Spain",
    "The Great Wall of China is not visible to the naked eye from the Moon",
    "An octopus has three hearts: two pump blood through the gills, and one pumps it through the rest of the body",
    "The capital of New York State is Albany",
    "The Red Sox won the World Series in 2004 for the first time since 1918"
]
```

### Generate embeddings for facts in the source of truth

Instantiate the Mistral API client, generate embeddings, and store embeddings
in a `numpy` [ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html).

```py
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

_response = client.embeddings.create(model=EMBEDDING_MODEL, inputs=FACTS)
FACT_EMBEDDINGS = np.array([row.embedding for row in _response.data])
```

### Define functions which process the transcription

For consistency, ensure the first letter of each assertion is capitalized.

```py
def capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]
```

Use your regular expression to split a chunk of transcribed text into sentences.
```py
def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END_RE.split(text.strip()) if s.strip()]

def extract_complete_sentences(buffer: str, already_processed: int) -> tuple[list[str], int]:
    sentences = split_into_sentences(buffer)
    complete = sentences if buffer.rstrip()[-1:] in ".!?" else sentences[:-1]
    return complete[already_processed:], len(complete)
```

### Define the functions which use LLMs to analyze text

Define an agent which decides whether a string contains a factual assertion.
Make this `async` so that you can call it from the asynchronous websocket
service. By making all of these asynchronous, transcription can continue,
streaming live on screen, while the models do their work.

```py
async def detect_assertion(sentence: str) -> str | None:
    response = await client.chat.complete_async(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    'You detect factual assertions in a sentence. Reply with JSON: '
                    '{"contains_assertion": bool, "assertion_text": string}. '
                    "assertion_text is the claim itself, or an empty string if there isn't one."
                ),
            },
            {"role": "user", "content": sentence},
        ],
    )
    result = json.loads(response.choices[0].message.content)
    return result["assertion_text"] if result["contains_assertion"] else None
```

Now, set up an agent which weighs a claim against one known fact and returns a verdict.
You may want to play with the prompt here, depending on the model you use, the types
of claims you want evaluated, and your desired behavior.

```py
async def judge_assertion(assertion: str, fact: str) -> dict:
    response = await client.chat.complete_async(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    'You fact-check a claim against a known fact. Reply with JSON: '
                    '{"verdict": "true"|"false"|"likely_true"|"likely_false"|"unknown", "reasoning": string}. '
                    'Use "unknown" if the known fact doesn\'t actually relate to the claim. '
                    'If you use "unknown", don\'t mention the irrelevant known fact in your reasoning. '
                    'Never say "the known fact" or "the context" in your reasoning, since the reader '
                    'won\'t know what that means - when the known fact confirms or denies the claim, '
                    'just state the correct information directly, e.g. "In fact, ...".'
                ),
            },
            {"role": "user", "content": f"Claim: {assertion}\n\nKnown fact: {fact}"},
        ],
    )
    return json.loads(response.choices[0].message.content)
```

This function finds the known fact that's most similar to a claim.
Pass it the embedding generated for that claim, and it will
return the index and similarity score of the fact that matches best.

```py
def most_similar_fact(claim_embedding: np.ndarray) -> tuple[int, float]:
    a = claim_embedding / np.linalg.norm(claim_embedding)
    f = FACT_EMBEDDINGS / np.linalg.norm(FACT_EMBEDDINGS, axis=1, keepdims=True)
    similarities = f @ a
    best_index = int(np.argmax(similarities))
    return best_index, float(similarities[best_index])
```

Now, you're ready for the core of the application's logic. Given a factual claim,
find the closest known fact, and ask your judging agent to compare them.

```py
async def check_claim(assertion: str) -> dict | None:
    response = await client.embeddings.create_async(model=EMBEDDING_MODEL, inputs=[assertion])
    claim_embedding = np.array(response.data[0].embedding)
    index, similarity = most_similar_fact(claim_embedding)
    if similarity < MATCH_THRESHOLD:
        return None
    return await judge_assertion(assertion, FACTS[index])
```

### Create the API
Your server has one main service: `/ws`, which uses a WebSocket.
This service bridges your browser's microphone audio into Mistral's real-time transcription,
fact-checking each sentence as it completes and pushing verdicts back as JSON.

This is the one service your JavaScript will call.

```py
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    # The browser's AudioContext picks its own sample rate; its first message
    # tells us what it chose so you can tell Mistral to expect the same thing.
    hello = json.loads(await websocket.receive_text())
    audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=hello["sample_rate"])

    async def receive_pcm_chunks():
        try:
            while True:
                yield await websocket.receive_bytes()
        except WebSocketDisconnect:
            return

    transcript_so_far = ""
    processed_count = 0
    in_flight: set[asyncio.Task] = set()  # holds refs so tasks aren't GC'd mid-flight

    async def fact_check_one(sentence: str) -> None:
        assertion = await detect_assertion(sentence)
        if assertion is None:
            return
        assertion = capitalize(assertion)
        result = await check_claim(assertion)
        if result is None:
            await websocket.send_json({"type": "verdict", "text": assertion, "matched": False})
        else:
            await websocket.send_json(
                {
                    "type": "verdict",
                    "text": assertion,
                    "matched": True,
                    "verdict": result["verdict"],
                    "reasoning": result["reasoning"],
                }
            )

    async for event in client.audio.realtime.transcribe_stream(
        receive_pcm_chunks(), model=TRANSCRIPTION_MODEL, audio_format=audio_format
    ):
        if isinstance(event, TranscriptionStreamTextDelta):
            transcript_so_far += event.text
            new_sentences, processed_count = extract_complete_sentences(transcript_so_far, processed_count)
        elif isinstance(event, TranscriptionStreamDone):
            # Final flush - the API's own transcript is authoritative once there's no more audio coming.
            transcript_so_far = event.text
            new_sentences = split_into_sentences(transcript_so_far)[processed_count:]
        elif isinstance(event, RealtimeTranscriptionError):
            raise RuntimeError(f"Realtime transcription error: {event}")
        else:
            continue

        await websocket.send_json({"type": "transcript", "text": transcript_so_far})

        for sentence in new_sentences:
            task = asyncio.create_task(fact_check_one(sentence))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

    if in_flight:
        await asyncio.gather(*in_flight)  # let any still-running fact-checks finish
```

Include a general service which catches everything `/ws` doesn't.
You'll use this to serve static files to the browser, like `index.html` and `app.js`.
```py
app.mount("/", StaticFiles(directory="web", html=True), name="web")
```

Finally, use `uvicorn` to run the server you've just defined.
```py
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

## Create the web app
Create a directory for your web assets. Inside that, create one file for the
`AudioWorklet`, and another for the JavaScript that will run your app.
```sh
mkdir web
cd web
touch audio-processor.js
touch app.js
```

### Your AudioWorklet module
You'll make a little `AudioWorkletProcessor` which will work in the
`AudioWorklet` thread. Its purpose is to grab audio samples as they come through
and pass those along to the main thread.

This module will live in its own file. Open up `audio-processor.js` and add this code.
```js
class PCMCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0][0];
    if (channel) {
      // The underlying buffer gets reused next quantum, so copy it with slice().
      this.port.postMessage(channel.slice());
    }
    return true; // keep the processor alive
  }
}

registerProcessor("pcm-capture-processor", PCMCaptureProcessor);
```

### Your HTML
Now make the HTML for your web app. You'll include markup for
* a title
* a button to start/stop recording audio
* an area for the streaming transcript
* an area for fact-checking verdicts

Create a new file called `index.html`. Add the following.
To make this look nice, use the `style.css` provided [in the GitHub repo](web/style.css).

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Fact Checker Demo</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <header>
    <div class="wordmark">
      <img class="logo-mark" src="img/cat.svg" alt="" width="36" height="36" />
      <span class="wordmark-text">Fact Checker Demo</span>
    </div>
    <button id="start-button">Start</button>
  </header>

  <main>
    <section class="panel">
      <h2>Transcript</h2>
      <p id="transcript">(nothing yet - click Start and speak into your microphone)</p>
    </section>

    <section class="panel">
      <h2>Results</h2>
      <div id="results"></div>
    </section>
  </main>

  <script src="app.js"></script>
</body>
</html>
```

### Your client-side logic
Now, write the JavaScript which runs your web app.
* Use an `AudioWorklet` to capture microphone input
* Convert the raw audio data to a standard format - 16-bit PCM
* Stream the processed audio to the server over a `WebSocket`

When the server sends back a message over the `WebSocket`, it has either an
updated transcription or a new fact-check for you to display.

Change this constant to buffer more or less audio. Lower values mean less latency,
but can also cause some audio to get lost. 
```js
const CHUNK_MS = 200; // how much audio we buffer client-side before sending
```

Now locate the DOM elements you need to work with, and initialize the `audioContext`
and the current websocket session. Wire up the start button event listener to
start or stop recording.

```js
const startButton = document.getElementById("start-button");
const transcriptEl = document.getElementById("transcript");
const resultsEl = document.getElementById("results");

let audioContext = null;
let ws = null; // the current session, or null when stopped

startButton.addEventListener("click", toggleRecording);

async function toggleRecording() {
  if (ws) {
    stopRecording();
  } else {
    await startRecording();
  }
}
```

Set up microphone capture to stream audio chunks to the server.
You'll buffer incoming audio quanta until you have `CHUNK_MS` worth,
then send those all at once.

```js
async function startRecording() {
  startButton.disabled = true;

  if (!audioContext) {
    audioContext = new AudioContext();
    await audioContext.audioWorklet.addModule("audio-processor.js");
  }
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioContext.createMediaStreamSource(stream);
  const captureNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");
  source.connect(captureNode);

  const samplesPerChunk = Math.floor((audioContext.sampleRate * CHUNK_MS) / 1000);
  let buffered = [];
  let bufferedLength = 0;

  const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  ws = socket;
```

Wire up three event listeners for this WebSocket session:
* `"open"`: when the user clicks "Start", the session starts. Tell the server what
  sample rate your microphone audio is using, and switch the button text to
  "Stop".
* `"message"`: Whenever a message arrives, render it. It'll be either an
updated transcript or a new fact-check verdict
* `"close"`: when the connection closes, release the microphone and reset the UI
  back to its starting state

```js
  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ sample_rate: audioContext.sampleRate }));
    startButton.textContent = "Stop";
    startButton.disabled = false;
  });

  socket.addEventListener("message", (event) => {
    renderMessage(JSON.parse(event.data));
  });

  socket.addEventListener("close", () => {
    stream.getTracks().forEach((t) => t.stop());
    captureNode.disconnect();
    source.disconnect();
    if (ws === socket) ws = null;
    startButton.textContent = "Start";
    startButton.disabled = false;
  });
```

Closing the WebSocket is all `stopRecording()` needs to do directly. That
triggers the `close` listener you just wrote above, which handles the
actual cleanup (releasing the microphone, resetting the button).

```js
function stopRecording() {
  if (!ws) return;
  ws.close(); // triggers the "close" listener above, which does the actual cleanup
}
```

Every message from the server is one of two things: a live transcript
update, or a fact-check verdict.
* For a new transcript, replace the on-screen text.
* For a verdict, build a small card, styled according to whether the claim
matched the source of truth, and add it to the results panel.

```js
function renderMessage(message) {
  // Update the transcript or add a fact-check result card to the page.
  if (message.type === "transcript") {
    transcriptEl.textContent = message.text;
    return;
  }

  const card = document.createElement("div");
  card.className = "card";
  if (!message.matched) {
    card.innerHTML = `
      <div class="claim">${escapeHtml(message.text)}</div>
      <div class="badge">No match</div>
    `;
  } else {
    card.innerHTML = `
      <div class="claim">${escapeHtml(message.text)}</div>
      <div class="badge badge-${message.verdict}">${message.verdict.replace("_", " ")}</div>
      <div class="reasoning">${escapeHtml(message.reasoning)}</div>
    `;
  }
  resultsEl.prepend(card);
}
```

`audio-processor.js` posts a message here every audio "quantum" (a fixed
batch of ~128 samples, delivered many times per second) with the
microphone's raw audio for that instant. Each piece gets appended
to buffered below. Once there's `CHUNK_MS` worth, they're stitched into
one array, converted to the PCM format the server expects, and sent -
then the buffer resets to start collecting the next chunk.

```js
  captureNode.port.onmessage = (event) => {
    buffered.push(event.data);
    bufferedLength += event.data.length;
    if (bufferedLength >= samplesPerChunk) {
      const chunk = mergeFloat32(buffered, bufferedLength);
      buffered = [];
      bufferedLength = 0;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(floatTo16BitPCM(chunk).buffer);
      }
    }
  };
}
```

Web Audio delivers audio in small fixed-size pieces as the microphone
captures it, not as one continuous stream. This stitches a batch of
those pieces back together into a single contiguous array right before
you send it over the network.

```js
function mergeFloat32(chunks, totalLength) {
  // Combine buffered Float32 audio chunks into one contiguous array.
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}
```

Web Audio represents each audio sample internally as a 32-bit float
between -1 and 1. Do a little math to convert those to 16-bit PCM,
the standard format most audio APIs use, and the format commonly used in
the ubiquitous `.wav` file.

```js
function floatTo16BitPCM(float32) {
  // Convert normalized Float32 samples into 16-bit PCM samples for transport.
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}
```

Finally, a basic helper function which inserts text into HTML safely.
```js
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
```

## Run it
Run `uv run server.py`, then open http://127.0.0.1:8000, click Start, and allow microphone access when your browser asks.

## Summary
You've just built a real-time audio fact-checking agent. A Python server streams  microphone audio to Mistral's real-time transcription API over a WebSocket, checks any factual claims it hears against a small set of known facts, and shows the live transcript and each verdict in a little web UI.

**What you built**
- An `AudioWorklet` that captures microphone audio in the browser and streams it to a Python server over a WebSocket
- A server that forwards that audio to Mistral's real-time transcription API and splits the growing transcript into sentences
- An assertion detector and fact-checker built on Mistral's chat and embeddings APIs
- A small web UI that renders the live transcript and each fact-check verdict as it arrives

**Mistral features used**
- Real-time audio transcription (`voxtral-mini-transcribe-realtime-2602`)
- Chat Completions API (`mistral-medium-3.5`)
- Embeddings API (`mistral-embed`)

[View the documentation](https://docs.mistral.ai/studio/audio/overview)