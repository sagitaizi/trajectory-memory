# Timeline

Deadlines are fixed. The camera-ready to-do list is in `PAPER_PROGRESS.md`.

## Deadlines

| Date | Milestone | |
|---|---|---|
| 2026-10-01 | Draft to Dr. Ezra Tsur for review | ✅ |
| 2026-10-10 | **MCSoC 2026 submission** (EDAS, hard), 8 pages | ✅ |
| 2026-10-31 | Notification: accepted as a regular paper | ✅ |
| 2026-11-08 | **Camera-ready + copyright form + author registration** (IEEE CPS, hard) | ⬜ |
| 2026-12-14…17 | Conference, Shanghai Jiao Tong University | ⬜ |

## Work plan

| Window | Goal | |
|---|---|---|
| Sep 9–16 | Recording; simulator matched to real; sim corpus; frontend, baselines, metrics, evaluation loop | ✅ |
| Sep 17–20 | Framework decision (G-F); SNN localiser + memory core; pretrain on sim; Stage 1 numbers on development clips | ✅ |
| Sep 21–27 | Deviation thresholds on development clips; label and score the held-out pendulum clips; Stage 1 result frozen | ✅ |
| Sep 28–Oct 1 | Held-out run, results table, draft to supervisor | ⬜ |
| Oct 2–9 | Supervisor comments, final 8 pages | ⬜ |
| Oct 10 | Submit | ✅ |
| Oct 25–Nov 8 | Answer the reviewers, camera-ready, copyright, registration | ⬜ |

## Fallback ladder

Report the highest rung reached, labelled honestly:

1. Memory + prediction + deviation on the **classical** front-end (safety net).
2. **Stage 1** — same, SNN from accumulated frames (the goal).
3. **Stage 2** — same, SNN from raw events. Dropped for the submission; future work.
