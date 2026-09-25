# Audio fact-checking agent

A simple demo which checks an audio stream for factual assertions, checking
each against a source of truth. Click Start, talk, and enjoy!

We use Mistral's real-time transcription API, plus its chat and embeddings APIs. 
We capture microphone audio continuously via an `AudioWorklet`, streaming it to
our little server in 200ms chunks over a `WebSocket`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies install automatically
on first run, or install them up front with:

```
uv sync
```

Create a `.env` file in this folder:

```
MISTRAL_API_KEY=your-mistral-api-key
```

## Run

```
uv run server.py
```

Then open http://127.0.0.1:8000, click Start, and allow microphone access
when your browser asks.

## How it works

- `web/audio-processor.js` runs on the browser's own audio thread,
  capturing raw microphone samples as they arrive.
- `web/app.js` buffers those into 200ms chunks and streams them to the
  server over a WebSocket, then renders whatever JSON messages come back.
- `server.py` forwards the audio to Mistral's real-time transcription API,
  splits the growing transcript into sentences, and checks each one that
  looks like a factual claim against `FACTS` - pushing the live transcript
  and each verdict back to the browser as JSON as soon as it's ready.
- The `FACTS` list at the top of `server.py` is our source of truth.
