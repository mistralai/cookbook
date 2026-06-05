"""
listen.py — Microphone input + Voxtral Transcribe 2 speech-to-text for Robot Ross.

Records until silence is detected, then transcribes with Voxtral (Mistral API).

Usage (standalone test):
    python listen.py                     # record and print transcript
    python listen.py --threshold 0.03    # raise threshold for noisy rooms

As a module:
    from listen import VoxtralListener
    listener = VoxtralListener()
    text = listener.listen()
    print(text)
"""

import sys, os, argparse, tempfile
import numpy as np
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAMPLE_RATE       = 16000   # Voxtral expects 16kHz
CHUNK_S           = 0.4     # seconds per audio chunk for VAD
SILENCE_S         = 0.8     # stop after this many seconds of silence
MIN_SPEECH_S      = 0.5     # ignore clips shorter than this (avoid false triggers)
MAX_RECORD_S      = 30      # hard cap on recording length
DEFAULT_THRESHOLD = 0.015   # RMS energy threshold for speech vs silence
DEFAULT_MODEL     = "voxtral-mini-2602"


class VoxtralListener:
    def __init__(self, model_id=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD):
        self.model_id  = model_id
        self.threshold = threshold
        self._client   = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY environment variable not set")
            try:
                from mistralai import Mistral
            except ImportError:
                print("  [Voxtral] ERROR: mistralai package not installed.", flush=True)
                raise
            self._client = Mistral(api_key=api_key)
        return self._client

    def listen(self, prompt="  🎤 Listening...", max_seconds=None):
        """
        Record from microphone until silence, return transcript string.
        Returns "" if nothing was captured or timeout reached.
        max_seconds: override MAX_RECORD_S for this call (e.g. short confirmations).
        """
        import sounddevice as sd

        client = self._get_client()

        chunk_samples     = int(SAMPLE_RATE * CHUNK_S)
        silence_needed    = int(SILENCE_S / CHUNK_S)
        min_speech_chunks = int(MIN_SPEECH_S / CHUNK_S)
        max_chunks        = int((max_seconds or MAX_RECORD_S) / CHUNK_S)

        if prompt:
            print(prompt, flush=True)

        audio_chunks    = []
        silence_count   = 0
        speech_started  = False

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_samples)
                chunk = chunk.flatten()
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms >= self.threshold:
                    if not speech_started:
                        print("  🔴 Recording...", flush=True)
                    speech_started = True
                    silence_count  = 0
                    audio_chunks.append(chunk)
                elif speech_started:
                    audio_chunks.append(chunk)   # include trailing silence
                    silence_count += 1
                    if silence_count >= silence_needed:
                        break

        if not speech_started or len(audio_chunks) < min_speech_chunks:
            return ""

        audio  = np.concatenate(audio_chunks)
        
        # Save to temporary file for Voxtral API
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        
        try:
            sf.write(tmp_path, audio, SAMPLE_RATE)
            
            with open(tmp_path, "rb") as f:
                response = client.audio.transcriptions.complete(
                    model=self.model_id,
                    file={"file_name": "audio.wav", "content": f.read()}
                )
            
            text = response.text.strip()
            print(f"  💬 Heard: {text}", flush=True)
            return text
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Ross — Voxtral Transcribe 2 test")
    parser.add_argument("--model",     default=DEFAULT_MODEL, help="Voxtral model ID")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"RMS silence threshold (default {DEFAULT_THRESHOLD}; raise for noisy rooms)")
    args = parser.parse_args()

    try:
        listener = VoxtralListener(model_id=args.model, threshold=args.threshold)
        print("Speak something (Ctrl+C to quit)...")
        while True:
            try:
                text = listener.listen()
                if text:
                    print(f"Transcript: {text}\n")
            except KeyboardInterrupt:
                print("\nDone.")
                break
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
