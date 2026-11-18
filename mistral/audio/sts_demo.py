"""
# Designing a Speech-to-Speech Assistant
From Audio Input to Spoken Response: Building a Triggerable Assistant with Voxtral and Mistral Small
**Blog:** [Mistral AI Learn Hub](https://learn.mistral.ai/home/blogs)

## Overview
This script demonstrates a real-time, triggerable speech-to-speech assistant using:
- **Mistral AI's APIs**: For transcription (`voxtral-mini-latest`) and text-to-speech (`voxtral-mini-tts-2603`).
- **Keyboard Trigger**: Press `Right Shift` to start/stop recording.
- **Web Search Agent**: Uses Mistral's `web_search` tool to fetch up-to-date information.
- **Audio Processing**: Captures input via `sounddevice`, streams responses with low latency.

### Key Features:
1. **Voice Activation**: Toggle recording with a keypress.
2. **Transcription**: Converts speech to text with speaker diarization.
3. **Agent Interaction**: Queries Mistral's conversational API, maintaining context.
4. **Audio Playback**: Streams synthesized speech in real-time.

### Dependencies:
- `pynput` (keyboard listener)
- `sounddevice` (audio I/O)
- `mistralai` (API client)
- `scipy` (WAV file handling)

**Note**: Requires a Mistral API key (`MISTRAL_API_KEY` env var).

To run this cript:
pip install pynput sounddevice queue scipy mistralai

For more information, visit the [Blog Post](https://learn.mistral.ai/home/blogs/designing-a-speech-to-speech-assistant-2026-04-02).
"""

from pynput.keyboard import Listener, Key
import sounddevice as sd
import numpy as np
import queue
from scipy.io.wavfile import write

from mistralai.client import Mistral
import os

import base64

# Settings
SAMPLE_RATE = 44100
CHANNELS = 1
OUTPUT_FILE = "output.wav"
is_recording = False
audio_queue = queue.Queue()
recording_stream = None

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)
transcription_model = "voxtral-mini-latest"
assistant_model = "mistral-small-latest"
model_tts = "voxtral-mini-tts-2603"
conversation_id = None

websearch_agent = client.beta.agents.create(
    model=assistant_model,
    description="Agent able to search information over the web, such as news, weather, sport results...",
    name="Websearch Agent",
    instructions="You are a general assistant. You have the ability to perform web searches with `web_search` to find up-to-date information. Keep your replies concise and to the point, with no markdown or code blocks.",
    tools=[{"type": "web_search"}],
    completion_args={
        "temperature": 0.3,
        "top_p": 0.95,
    }
)

play_queue = queue.Queue()
voice_id = "gb_jane_neutral"

def query(transcription, conversation_id=None):
    if conversation_id:
        response = client.beta.conversations.append(
            conversation_id=conversation_id,
            inputs=transcription
        )
    else:
        response = client.beta.conversations.start(
            agent_id=websearch_agent.id,
            inputs=transcription
        )
    conversation_id = response.conversation_id
    content = response.outputs[-1].content
    if isinstance(content, list):
        content = "".join([c.text for c in content if c.type == "text"])
    return conversation_id, content

def transcribe(file_path):
    with open(file_path, "rb") as f:
        transcription_response = client.audio.transcriptions.complete(
            model=transcription_model,
            file={
                "content": f,
                "file_name": "audio.mp3",
            },
            diarize=True,
            timestamp_granularities=["segment"],
        )
    transcription = "\n".join([f"[{s.start} -> {s.end}] {s.speaker_id} : {s.text}" for s in transcription_response.segments])
    return transcription

def play_callback(outdata, frames, time, status):
    """Callback for audio playback."""
    try:
        data = play_queue.get_nowait()
    except:
        outdata.fill(0)
        return
    outdata[:len(data), 0] = data
    if len(data) < len(outdata):
        outdata[len(data):, 0] = 0

def play_audio(text: str) -> None:
    """Stream and play audio for the given text."""
    with sd.OutputStream(
        samplerate=24000,
        channels=1,
        dtype=np.float32,
        callback=play_callback,
        blocksize=960,
        latency="low",
    ):
        with client.audio.speech.complete(
            model=model_tts,
            input=text,
            voice_id=voice_id,
            response_format="pcm",
            stream=True,
        ) as stream:
            for event in stream:
                if event.event == "speech.audio.delta":
                    audio_data = base64.b64decode(event.data.audio_data)
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    for i in range(0, len(audio_array), 960):
                        block = audio_array[i:i + 960]
                        play_queue.put(block)
                elif event.event == "speech.audio.done":
                    break

        while not play_queue.empty():
            sd.sleep(100)

def audio_callback(indata, frames, time, status):
    """Callback to collect audio data."""
    if is_recording:
        audio_queue.put(indata.copy())

def on_press(key):
    """Start/stop recording on Right Shift press."""
    global is_recording, recording_stream, conversation_id
    if key == Key.shift_r:
        if not is_recording:
            print("Recording started... Press Right Shift again to stop.")
            is_recording = True
            recording_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=audio_callback
            )
            recording_stream.start()
        else:
            print("Recording stopped. Saving to 'output.wav'...")
            is_recording = False
            recording_stream.stop()
            recording_stream.close()
            audio_data = []
            while not audio_queue.empty():
                audio_data.append(audio_queue.get())
            if audio_data:
                audio_array = np.concatenate(audio_data, axis=0)
                write(OUTPUT_FILE, SAMPLE_RATE, audio_array)
                print(f"Saved to {OUTPUT_FILE}")
                trancription = transcribe(OUTPUT_FILE)
                print(f"Transcription: {trancription}")
                conversation_id, content = query(trancription, conversation_id)
                print(f"Assistant: {content}")
                play_audio(content)

print("Press 'Right Shift' to start/stop recording. Exit with Ctrl+C.")
with Listener(on_press=on_press) as listener:
    listener.join()
