# Decisions — 2026-09-09

The reasoning behind the plan, so it's recoverable offline. `PLAN.md` has the current state;
this has the "why".

## The pivot
The main thesis's ego-motion / texture-rejection work can't progress without the physical
camera, which is unavailable for ~a month. Trajectory memory is a direction from the
2026-09-03 Ezra Tsur meeting (`future-directions.md` §5a) that can be done entirely offline
from recorded + simulated data. So: shift effort to a trajectory-memory paper for **IEEE
MCSoC 2026** (submission 2026-10-10, draft to supervisor 2026-10-01, 8 pages).

## What the system does
A spiking network watches a target move in a repeating path, builds a memory of it, predicts
its position a short horizon ahead, and flags when the motion stops matching. Offline only.

## Scope
- **In:** memorise + predict + deviation ("report a break") detection.
- **Out (future work):** re-learning a *new* pattern after a break; the attention-span /
  habituation mechanism across multiple objects; feeding predictions to the pan-tilt rig.

## "Online" — what it means here
Three separate things get called "online". The system needs #1 and #3, not necessarily #2:
1. **Streaming inference** — process the event stream step by step. Every SNN gives this.
2. **Online weight updates** — synapses change during deployment via a local rule.
3. **Fast acquisition of the specific instance** — end up with a working model of *this*
   path, which can live in network state / a fast readout with frozen recurrent weights.

Chosen approach: **pretrain on simulation, freeze, adapt in state** at deployment. Dropping
"re-learn after a break" removed most of the pressure toward #2.

## The funnel (revised)
The position-stream stage was dropped so the SNN always does target extraction.
1. **Stage 1 (goal):** SNN takes accumulated event frames → localises + memorises + predicts
   + flags deviation.
2. **Stage 2 (upside):** raw events end-to-end.
- Fallback: classical centroid → SNN memory core (reported with an asterisk).

## Framework — decided (decision gate G-F)
Localiser: a spiking conv net in snnTorch. Memory: the clock-and-map network, its weights
built by Nengo (NEF) and its LIF neurons run in PyTorch, the PES map rule by hand. The
snnTorch learned memories and the Nengo LMU are ablations. Details in `PLAN.md`.

## Data
- **Own corpus, recorded 2026-09-09:** fan-mounted brush, string pendulum, hand-moved printed
  square, and blank-wall clips (4a motor-swept camera = exact encoder ground truth; 4b
  hand-moved square, clean background). Objects: brush, printed square, ball. The 4a encoder
  ground truth was abandoned; every clip is hand-labelled (`PAPER_PROGRESS.md`).
- **Simulation (v2e):** unlimited labelled trajectories through the DVXplorer camera model,
  for pretraining. Stand it up before the camera leaves; validate sim-vs-real day one.
- **Ground truth:** analytic where the motion is driven; sparse hand-labels + interpolation
  for the hand clips.
- Public event datasets: **generalisation section only**, not training.

## Contribution framing (Ezra Tsur)
The spiking dynamics should *solve* the prediction problem — predictive inference — not "the
same task on a different network". For clean periodic motion a classical method will likely
beat the SNN on raw accuracy, so the contribution rests on: event-native operation, the
dynamics-as-solver framing, online acquisition, and being the substrate a low-latency
neuromorphic tracker would deploy. Loihi/Lava unavailable → energy/latency **cited, not
measured**.

## Repo
New repo `D:\Projects\trajectory-memory`, imports `recording`/`camera`/`pipeline` from the
Thesis repo (no copy). Shares the `thesis` conda env + `snntorch`, `v2e`.
