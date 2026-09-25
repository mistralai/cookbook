"""
Stream microphone audio to this server via a WebSocket.
As the transcription streams in, we check it for sentences that look like
factual claims. We check each such claim against a source of truth and display
the result.

Usage:
    uv run server.py

Then open http://127.0.0.1:8000, click Start, and allow microphone access.
"""

import asyncio
import json
import os
import re

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

load_dotenv()

EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-medium-3.5"
TRANSCRIPTION_MODEL = "voxtral-mini-transcribe-realtime-2602"

# Below this cosine similarity, a claim is treated as unrelated to anything we know.
MATCH_THRESHOLD = 0.5

# The source of truth. In a real-world scenario, you'd probably want to source facts in advance anyway.
FACTS = [
    "Venus rotates once every 243 Earth days and completes one orbit around the Sun in 225 Earth days",
    "The mythical unicorn is Scotland's official national animal",
    "France is slightly larger than Spain in total area - about 551,500 km², compared to 505,400 km² for Spain",
    "The Great Wall of China is not visible to the naked eye from the Moon",
    "An octopus has three hearts: two pump blood through the gills, and one pumps it through the rest of the body",
    "The capital of New York State is Albany",
    "The Red Sox won the World Series in 2004 for the first time since 1918"
]

SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    """Split a chunk of transcribed text into individual sentences."""
    return [s.strip() for s in SENTENCE_END_RE.split(text.strip()) if s.strip()]


def extract_complete_sentences(buffer: str, already_processed: int) -> tuple[list[str], int]:
    """Split the transcript so far into sentences, holding back a trailing partial one.

    There's no way to tell a finished sentence from one still being spoken
    except by checking whether the buffer itself ends in sentence punctuation.
    """
    sentences = split_into_sentences(buffer)
    complete = sentences if buffer.rstrip()[-1:] in ".!?" else sentences[:-1]
    return complete[already_processed:], len(complete)


def capitalize(text: str) -> str:
    """Capitalize just the first letter, leaving the rest of the text untouched."""
    return text[:1].upper() + text[1:]


client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Embedded once at startup - just a handful of facts, no need to cache this to disk.
_response = client.embeddings.create(model=EMBEDDING_MODEL, inputs=FACTS)
FACT_EMBEDDINGS = np.array([row.embedding for row in _response.data])


async def detect_assertion(sentence: str) -> str | None:
    """Ask the model whether a sentence contains a factual claim, and return it if so."""
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


def most_similar_fact(claim_embedding: np.ndarray) -> tuple[int, float]:
    """Return the index and similarity score of the known fact closest to claim_embedding."""
    a = claim_embedding / np.linalg.norm(claim_embedding)
    f = FACT_EMBEDDINGS / np.linalg.norm(FACT_EMBEDDINGS, axis=1, keepdims=True)
    similarities = f @ a
    best_index = int(np.argmax(similarities))
    return best_index, float(similarities[best_index])


async def judge_assertion(assertion: str, fact: str) -> dict:
    """Ask the model to weigh a claim against one known fact and return a verdict."""
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


async def check_claim(assertion: str) -> dict | None:
    """Embed a claim, find the closest known fact, and judge the claim against it."""
    response = await client.embeddings.create_async(model=EMBEDDING_MODEL, inputs=[assertion])
    claim_embedding = np.array(response.data[0].embedding)
    index, similarity = most_similar_fact(claim_embedding)
    if similarity < MATCH_THRESHOLD:
        return None
    return await judge_assertion(assertion, FACTS[index])


app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Bridge one browser's microphone audio into Mistral's real-time transcription,
    fact-checking each sentence as it completes and pushing results back as JSON.
    """
    await websocket.accept()

    # The browser's AudioContext picks its own sample rate; its first message
    # tells us what it chose so we can tell Mistral to expect the same thing.
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


# Mounted last so it only catches what /ws doesn't (index.html, app.js, ...).
app.mount("/", StaticFiles(directory="web", html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
