# Hardware Interface Subsystem

## 1. Role
The Hardware Interface subsystem provides the low-level communication and control for the Huenit robotic arm via a serial (USB) G-code interface.

## 2. Key Components
- **Serial Interface**: Communication occurs over `/dev/cu.usbserial-310` at 115200 baud.
- **G-code Controller**: `huenit_svg.py` and `huenit_draw.py` parse SVG or shape commands into G-code for the arm's onboard controller.
- **Z-axis Control**: Manages pen-up (`Z_UP=6.0mm`) and pen-down movements.
- **Tilt Correction**: `TILT_SLOPE` in `huenit_svg.py` provides Z-height correction per mm of Y travel, indicating the drawing surface may not be perfectly level.

## 3. Calibration
- **Required**: Must be run at the start of each session.
- **Tool**: `huenit_draw.py calibrate`
- **Output**: Generates `calibration.json` in the huenit skill directory.

## 4. Uncertainty & Contradictions
- **Calibration Persistence**: Code checks for `/tmp/huenit_ready.flag`. Since `/tmp` is cleared on reboot, calibration must be performed every session. However, some documentation implies a more permanent state could be achieved.
- **Safety**: The arm's G-code parser does not appear to have complex obstacle avoidance. It relies on the user ensuring the drawing area (125mm x 125mm) is clear.

---
**Sources:**
- `~/.openclaw/workspace/skills/huenit/huenit_svg.py`
- `~/.openclaw/workspace/skills/huenit/huenit_draw.py`
- `AGENTS/CONTEXT/robot_ross_artist.md`
