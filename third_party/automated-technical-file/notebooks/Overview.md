# RobotRoss System Overview

## 1. Introduction
RobotRoss is an autonomous robotic artist platform designed to bridge AI creativity with physical execution. It uses a Huenit robotic arm to draw plotter-optimized SVGs or write calligraphy while providing poetic narration in the style of Bob Ross.

## 2. Core Architecture
The system follows a three-layer architecture:
- **Commerce Layer**: [[CommerceLayer]] handles order intake, queue management, and the competitive bidding system.
- **Orchestration Layer**: [[JobOrchestration]] (primarily `bob_ross.py`) manages the job lifecycle, narration, recording, and uploading.
- **Hardware Layer**: [[HardwareInterface]] provides low-level G-code control and calibration for the robot arm.

## 3. Key Subsystems
- [[CommerceLayer]]: Salesman API and order queue management.
- [[JobOrchestration]]: The brain of the artist side.
- [[HardwareInterface]]: Serial G-code control of the Huenit arm.
- [[OrderManagement]]: Polling and logging of drawing jobs on the Artist side.
- [[Narration]]: Local LLM-based poetic commentary.
- [[VideoProof]]: Automated OBS recording and YouTube uploading.

## 4. Key Integration Topics
- [[BiddingRules]]: Competitive overwrite rules and the 8x8 Wall of Fame.
- [[ShopifyIntegration]]: Webhooks, metadata write-back, and human e-commerce.
- [[VirtualsACP]]: Agentic commerce protocol for autonomous hiring.
- [[Calibration]]: Necessary startup procedures for hardware accuracy.
- [[Compliance]]: EU AI Act mapping and architectural traceability.

## 5. Hardware Requirements
- **Computer**: Mac Mini M4 (Apple Silicon)
- **Robot**: Huenit Robotic Arm
- **Cameras**: Reolink 4K (Main) + macOS Screen Capture (Board)
- **Audio**: BlackHole 2ch for internal routing

## 6. Software Stack
- **OS**: macOS (darwin) for Artist; Ubuntu (Linux) for Salesman.
- **LLM**: Apertus 8B (local via Ollama)
- **Agent Framework**: OpenClaw
- **Utilities**: OBS Studio, ffmpeg, Python 3.12, Node.js

## 7. Uncertainty & Contradictions
- **Calibration Persistence**: Source code indicates calibration is required after every restart (`READY_FLAG` in `/tmp`), but some docs suggest it might be semi-persistent.
- **Pen Pressure**: Manual leveling of the table is mentioned as a physical requirement that software cannot currently compensate for.
- **Narration Latency**: There is an inherent delay while Apertus 8B generates narration, which `bob_ross.py` handles with a wait period, but the impact on "live" feel is a point of ongoing optimization.

---
**Sources:**
- `AGENTS/CONTEXT/robot_ross_artist.md`
- `AGENTS/CONTEXT/robot_ross_salesman.md`
- `~/.openclaw/workspace/skills/robot-ross/bob_ross.py`
