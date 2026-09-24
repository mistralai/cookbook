# Calibration Topic

## 1. Importance
Calibration is critical for the Huenit robotic arm to ensure consistent pen pressure and accurate drawing within the 125mm x 125mm area. Without calibration, the pen may drag or fail to reach the paper.

## 2. Process
- **Tool**: `huenit_draw.py calibrate`
- **Steps**:
    1. Manually jog the arm to the home position.
    2. The arm touches several points on the surface to determine the plane and tilt.
    3. The Z-height for pen-up (`Z_UP=6.0mm`) and pen-down is established.
- **Output**: Writes `calibration.json` which is consumed by all hardware control scripts.

## 3. Operations
Calibration must be run after every physical restart of the arm. It is triggered by the operator using a desktop shortcut: `Calibrate Robot Ross.command`.

## 4. Uncertainty & Contradictions
- **Surface Leveling**: Software calibration can compensate for some tilt (`TILT_SLOPE`), but physical documentation emphasizes that the table should be as level as possible. The limit of software-based tilt compensation is not explicitly documented.
- **Auto-Calibration**: There is no mention of a "re-calibration" trigger during long-running sessions, which might be necessary if the arm drifts or the surface shifts.

---
**Sources:**
- `skills/huenit/huenit_draw.py`
- `AGENTS/CONTEXT/robot_ross_artist.md`
