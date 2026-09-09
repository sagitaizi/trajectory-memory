# trajectory-memory

SNN-based trajectory memory from event-camera input: a spiking network watches a target move
in a repetitive path, memorises it, predicts it a short time ahead, and flags when the motion
stops matching. Offline. M.Sc. thesis work (paper 4c), targeted at IEEE MCSoC 2026.

- **Start:** `PLAN.md`, then `TIMELINE.md`.
- **Architecture:** `docs/implementation-plan.md`.
- **Context for Claude Code:** `CLAUDE.md`.

## Setup

```
conda activate thesis          # shared with D:\Projects\Thesis
pip install -r requirements.txt
```

Reuses `recording`, `camera`, `pipeline` from `D:\Projects\Thesis` (imported, not copied).
Override its location with `THESIS_REPO` if it is not the sibling directory.
