import os
import sys
import subprocess
import tempfile
import platform
import base64

_IS_WINDOWS = platform.system() == "Windows"
_IS_MAC     = platform.system() == "Darwin"

# ── Voxtral TTS (Mistral API) ─────────────────────────────────────────────────
VOXTRAL_VOICE = "en_paul_neutral"
VOXTRAL_MODEL = "voxtral-mini-tts-2603"


def _play_wav(path):
    """Play a WAV file using the best available method for this OS."""
    if _IS_WINDOWS:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
    elif _IS_MAC:
        subprocess.run(["afplay", path], check=True)
    else:
        # Linux fallback
        subprocess.run(["aplay", path], check=True)


def _speak_voxtral(text):
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print("  Voxtral: MISTRAL_API_KEY not set — skipping", file=sys.stderr)
        return
    try:
        from mistralai import Mistral
    except ImportError:
        print("  Voxtral: mistralai not installed — skipping", file=sys.stderr)
        return
    client = Mistral(api_key=api_key)
    
    try:
        response = client.audio.speech.complete(
            model=VOXTRAL_MODEL,
            input=text,
            voice_id=VOXTRAL_VOICE,
            response_format="wav",
        )
        
        # Standard Mistral SDK returns audio content in response.content
        audio_bytes = response.content

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            with open(tmp, "wb") as f:
                f.write(audio_bytes)
            _play_wav(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception as e:
        print(f"  Voxtral TTS failed: {e}", file=sys.stderr)


# ── Public API ────────────────────────────────────────────────────────────────
def speak(text):
    """
    Speak text using Voxtral TTS (Mistral API).
    """
    print(f"Narrating: {text}")
    _speak_voxtral(text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Robot Ross — Voxtral TTS test")
    parser.add_argument("text", nargs="?",
                        default="Hello, I am Robot Ross, and today we will make some happy little drawings.")
    args = parser.parse_args()
    speak(args.text)
