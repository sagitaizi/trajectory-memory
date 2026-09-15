# Timeline

Today: **2026-09-16**. Camera leaves **2026-09-10** (Sagi away ~1 month, laptop only, returns
2026-10-10).

## Fixed deadlines

| Date | Milestone | Done |
|---|---|---|
| 2026-09-09 (tonight) | Recording session — all four setups (`RECORDING_PLAN.md`) | ✅ |
| 2026-10-01 | Draft to Dr. Ezra Tsur for review | ⬜ |
| 2026-10-10 | **MCSoC 2026 submission** (EDAS, hard, final extension), 8 pages | ⬜ |
| 2026-10-31 | Notification | ⬜ |
| 2026-11-08 | Camera-ready + author registration | ⬜ |
| 2026-12-14…17 | Conference, Shanghai Jiao Tong University | ⬜ |

## Work plan

| Window | Goal | Done |
|---|---|---|
| Tonight | 4 recording setups; verify each clip's duration after capture | ✅ |
| Tonight | Repo skeleton runnable (`_thesis_path`, imports, empty modules import clean) | ✅ |
| Sep 10–13 | `trajectories.py` + `simulate.py`; sim-vs-real eyeball check on one real clip | ✅ |
| Sep 10–13 | Framework research (G-F) — decide reservoir vs snnTorch for the memory core | 🟨 slipped; facts gathered in `PLAN.md` G-F, decision pending |
| Sep 14–20 | `frontend.frames`; provisional model + `baseline.py`; pretrain on sim | 🟨 frontend + baselines done 09-16; sim corpus generated; SNN core open |
| Sep 14–20 | Stage 1 prediction results on real clips; `metrics.py` (error, lock-on) | ⬜ |
| Sep 21–27 | Deviation detection working + scored (ROC, latency); Stage 1 result frozen | ⬜ |
| Sep 21–27 | Begin Stage 2 (raw events) if Stage 1 is solid | ⬜ |
| Sep 28–Oct 1 | Consolidate results, write draft, **send to supervisor Oct 1** | ⬜ |
| Oct 2–9 | Stage 2 push + address supervisor comments + finalise 8 pages | ⬜ |
| Oct 10 | Submit | ⬜ |

## Fallback ladder

Report whichever is the highest rung reached, honestly labelled:

1. Memory + prediction + deviation on a **classical** front-end (safety net).
2. **Stage 1** — same, SNN from accumulated frames (the goal).
3. **Stage 2** — same, SNN from raw events (upside).
