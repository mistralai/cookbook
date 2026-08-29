# Job Orchestration Subsystem

## 1. Role
The Job Orchestration subsystem is the central "brain" of the RobotRoss artist side. It is primarily implemented in `bob_ross.py` and coordinates all other subsystems to fulfill drawing orders.

## 2. Main Orchestrator: `bob_ross.py`
This script manages the high-level workflow of a drawing job:
1. **Order Intake**: Accepts commands like `sketch`, `write`, `draw`, or `svg`.
2. **Readiness Check**: Verifies hardware connection, calibration state, and local model (Ollama) availability.
3. **Locking Mechanism**: Uses `/tmp/robot_ross_running.lock` to ensure only one job runs at a time.
4. [[Narration]] **Generation**: Calls Mistral to generate poetic commentary based on the job prompt.
5. [[VideoProof]] **Control**: Triggers OBS Studio recording via WebSocket.
6. **Execution Control**: Spawns hardware-level scripts (`huenit_svg.py`, etc.) to move the arm.
7. **Cleanup**: Releases the job lock and triggers post-processing of the video proof.

## 3. Job Lifecycle
The full job lifecycle is documented in [[Overview#Full Job Lifecycle]].

## 4. Uncertainty & Contradictions
- **Error Recovery**: The current implementation of `bob_ross.py` uses a job lock but does not explicitly document how it handles mid-job hardware failures or power loss. If a job is interrupted, the lock might remain, requiring manual intervention.
- **Hyphenation Logic**: `bob_ross.py` includes a custom `_hyphenate_word` function for text wrapping. It is unclear if this logic is optimized for all languages or only English/German.

---
**Sources:**
- `skills/robot-ross/bob_ross.py`
- `AGENTS/CONTEXT/robot_ross_artist.md`
