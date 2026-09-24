# Narration Subsystem

## 1. Role
The Narration subsystem generates Bob Ross-style poetic commentary for each drawing job, providing a unique "personality" to the robot.

## 2. Key Components
- **Local LLM**: Mistral (hosted) with Ministral local fallback.
- - **Voice Output**: Voxtral (hosted) with Ministral local fallback.

## 3. Workflow
1. **Prompt Generation**: `bob_ross.py` constructs a prompt based on the drawing job (e.g., "a happy little lighthouse").
2. **Narration JSON**: The narration model generates a JSON structure containing:
    - `intro`: Spoken before drawing starts.
    - `commentary[]`: A list of short phrases spoken during drawing at regular intervals.
    - `outro`: Spoken after the job completes.
3. **Playback**: Commentary is spoken in parallel with the arm movements.

## 4. Uncertainty & Contradictions
- **Commentary Timing**: `bob_ross.py` uses a hard-coded `COMMENTARY_INTERVAL` (default 6 seconds). If the drawing is very simple, some commentary might be cut off. If it is very complex, there might be long periods of silence if the list of phrases is short.
- **TTS Personality**: While the text is generated in the style of Bob Ross, the voice (Voxtral) may lack the specific intonation of the artist, creating a mismatch between text and voice.

---
**Sources:**
- `skills/robot-ross/bob_ross.py`
- `AGENTS/CONTEXT/agentegra.md`
- `AGENTS/CONTEXT/robot_ross_artist.md`
