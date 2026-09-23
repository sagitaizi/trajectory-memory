# What our numbers can and cannot be compared to

For §II and §V-E. The short answer: **no published system does this task**, so there is no
number to beat. The comparison the paper can make is against our own baselines on our own
corpus; everything else is positioning, and must be written as positioning.

## Why there is no comparable number

The task is: an event camera watches a target repeating a path, a spiking network learns
that path online, predicts it a short time ahead, and flags when the motion stops matching.
Published event-based prediction work differs in at least one of these ways every time:

| Work | What it predicts | Why its numbers are not ours |
|---|---|---|
| Debat et al., 2021 (the nearest prior art) | Where a thrown ball will land | One-shot ballistic flight, not a repeated path; trained offline on a labelled set; no deviation signal. Its error is a landing-point error in a different geometry; a pixel number from it is not our pixel number. |
| Egocentric ping-pong prediction, 2025 | 3-D ball trajectory, real time | Same shape of task as Debat, with 3-D ground truth; reports 3-D errors in mm against a motion-capture rig. No repetition, no online path learning. |
| N-DriverMotion, 2024 | Driver motion class / next pose, on Loihi 2 | Classification-style targets and a different sensor geometry; useful as evidence the full stack is publishable, not as an error number. |
| EventVOT (2024), FE108 (2021) | Bounding boxes, frame by frame | Tracking benchmarks: they measure "where is it now", which is our *localiser's* job, not the memory's. Their metrics (success rate, precision plots) could be run against our localiser if we adopt their protocol -- see "the one comparison we could add" below. |
| Vacuum Spiker, 2025 | Anomalies in time series, SNN | Closest in spirit for Phase D, but on generic time series, not on a learned spatial path; its AUC is over other datasets. |
| Rhythmic DMPs (Ijspeert 2002; Saveriano 2023 survey) | A learned rhythmic trajectory, non-neuromorphic | The nearest *functional* relative: learn a periodic movement, reproduce it, adapt its phase. Not event-based, not spiking, and normally driven by clean joint-angle input rather than a noisy event centroid. Good for framing the clock-and-map design, not for an error comparison. |

## What the paper compares against instead

Our own baselines, on the same clips, with the same input and the same metrics -- which is
the honest comparison and the one a reviewer can check:

- **Periodic Kalman filter** (`baseline.PeriodicKalman`): the classical answer to "predict a
  repetitive path", given the same centroid input.
- **Harmonic fit** (`baseline.HarmonicFit`): a least-squares cycle fitted to the observed
  track -- a strong offline reference that cannot run causally.
- **Arithmetic clock-and-map** (`phasemap.PhaseMap`): our own design without spikes. This is
  the ablation that says what the spiking implementation costs.
- **Classical centroid vs the spiking localiser** as the memory's input: what Stage 1 adds.

## The one comparison we could add, if there is time

EventVOT or FE108 give public event clips with bounding-box ground truth. Running our
*localiser* on a subset and reporting their tracking metric would place one of our
components on an external dataset. It would not test the memory (their targets do not
repeat a path), and it needs their evaluation protocol implemented. Listed in `PLAN.md`
as the "external-dataset generalisation check"; worth doing only if §V lands early.

## How to word it in §II

Say plainly that event-based trajectory prediction exists (Debat; ping-pong; N-DriverMotion)
but always for one-shot motions with offline training, and that rhythmic movement primitives
learn repeated paths but from clean state input and without spikes or deviation detection.
Then state the gap this paper fills -- a repeated path learned online, from events, in
spiking neurons, with a deviation signal -- and that the evaluation is therefore against
classical baselines on a purpose-recorded corpus, not against a published benchmark number.
Do not write "outperforms the state of the art"; write "beats the classical baselines on
this corpus" and name them.
