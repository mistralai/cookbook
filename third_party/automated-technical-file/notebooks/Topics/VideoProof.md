# Video Proof Subsystem

## 1. Role
The Video Proof subsystem automatically records drawing sessions and uploads them to YouTube as "proof-of-work" for the customer.

## 2. Key Components
- **Recording**: OBS Studio (WebSocket port 4455) records the drawing job.
- **Scene Switching**: `obs_manager.py` switches between "Workspace" (intro/outro) and "Artist" (arm camera + status) scenes.
- **Post-Processing**: `ffmpeg` is used to trim dead air and remux the video (MOV to MP4) for optimal YouTube upload.
- **Upload**: `youtube_upload.py` uses the Google OAuth2 API to upload the video and retrieve the public URL.

## 3. Workflow
1. **Start Recording**: `bob_ross.py` triggers OBS at the start of the job.
2. **Scene Control**: `obs_manager.py` handles scene transitions at specific job milestones.
3. **Stop Recording**: `bob_ross.py` stops OBS after the job and narration finish.
4. **Processing**: `ffmpeg` remuxes the file to `.mp4`.
5. **Publish**: `youtube_upload.py` uploads the video and records the URL in the [[OrderManagement#Local Ledger]].

## 4. Uncertainty & Contradictions
- **Storage Management**: The source code does not mention a pruning strategy for local MOV/MP4 files in `~/Movies/`. Over time, this could fill the Mac Mini's disk.
- **Upload Reliability**: If the YouTube upload fails, there is no documented automatic retry mechanism. The job URL in the Salesman API might remain blank or incorrect.

---
**Sources:**
- `skills/robot-ross/obs_manager.py`
- `skills/robot-ross/youtube_upload.py`
- `AGENTS/CONTEXT/robot_ross_artist.md`
