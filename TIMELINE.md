# Timeline

Camera away until **2026-10-10** (laptop only). Deadlines are fixed; the work plan is the
order things must land to make the draft.

## Deadlines

| Date | Milestone | |
|---|---|---|
| 2026-10-01 | Draft to Dr. Ezra Tsur for review | ⬜ |
| 2026-10-10 | **MCSoC 2026 submission** (EDAS, hard), 8 pages | ⬜ |
| 2026-10-31 | Notification | ⬜ |
| 2026-11-08 | Camera-ready + author registration | ⬜ |
| 2026-12-14…17 | Conference, Shanghai Jiao Tong University | ⬜ |

## Work plan

| Window | Goal | |
|---|---|---|
| Sep 9–16 | Recording; simulator matched to real; sim corpus; frontend, baselines, metrics, evaluation loop | ✅ |
| Sep 17–20 | Framework decision (G-F); SNN localiser + memory core; pretrain on sim; Stage 1 numbers on development clips | ✅ |
| Sep 21–27 | Deviation thresholds on development clips; label held-out clips; Stage 1 result frozen; Stage 2 if Stage 1 is solid | ⬜ |
| Sep 28–Oct 1 | Held-out run, results table, draft to supervisor | ⬜ |
| Oct 2–9 | Supervisor comments, Stage 2 push, final 8 pages | ⬜ |
| Oct 10 | Submit | ⬜ |

## Fallback ladder

Report the highest rung reached, labelled honestly:

1. Memory + prediction + deviation on the **classical** front-end (safety net).
2. **Stage 1** — same, SNN from accumulated frames (the goal).
3. **Stage 2** — same, SNN from raw events (upside).
