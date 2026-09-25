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

## Running things

All scripts run with the `thesis` env's interpreter (`conda run` swallows flags like `--n`):

```
python scripts/make_sim_dataset.py --n 100 --duration 15      # the simulated corpus -> corpus/sim/
python scripts/make_sim_dataset.py --kind pendulum --n 100 --duration 15 --out corpus/sim_pendulum
python scripts/match_sim_real.py corpus/real/fan/fan_brush_slow_02   # sim-vs-real check
python scripts/evaluate.py --set development                   # every paper table -> runs/eval/development/
python scripts/evaluate.py --set development --inputs centroid snn   # add the spiking-localiser rows
python scripts/paper_results.py run --ablations                 # the paper's runs + traces -> runs/paper/development/
python scripts/paper_figures.py                                  # results figures -> paper/figures/, tables -> runs/paper/tables.tex
python scripts/run_experiment.py --set development             # scorecard for every memory
python scripts/run_experiment.py corpus/sim/sim_003.npz --memory kalman
python scripts/check_localiser.py --set development            # centroid vs hand-labels
python scripts/replay_gt.py corpus/real/fan/fan_brush_slow_02 --model kalman   # watch it live
python scripts/make_tracks.py                                  # position tracks for pretraining
python scripts/make_frames.py --downsample 8                   # frame sets for the localiser
python scripts/train_localiser.py                              # train the spiking localiser -> runs/localiser/snn.pt
python scripts/train_localiser.py --corpus corpus/sim_pendulum --init runs/localiser/snn.pt --lr 1e-3 --epochs 10 --out runs/localiser/pend_ft.pt
python scripts/check_localiser.py --set development --localiser snn   # localiser vs hand-labels
python scripts/run_experiment.py --set development --memory snn_phasemap --localiser snn --localiser-checkpoint runs/localiser/pend_ft.pt
python scripts/train_memory.py                                 # ablation: pretrain the two-layer SNN memory -> runs/memory/snn.pt
python scripts/run_experiment.py --set development --memory snn --subtract-offset   # ablation
python -m pytest                                               # ~2 min
```

`corpus/sets.yaml` names the development and held-out real clips. Results land in
`runs/results/`, replays in `runs/replay/`; both directories are git-ignored.
