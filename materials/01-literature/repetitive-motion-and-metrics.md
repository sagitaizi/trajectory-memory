# Repetitive motion in vision, and the metrics each field reports

The question: who else watches a repeating motion, and what numbers do they print? Five
communities each own one piece of our task and none owns the join, so every column of our
results table has to be compared against a different specialist. This note is that map,
and the naming our tables adopt from it.

Companion notes: `classical-baselines.md` (the methods), `deviation-and-novelty.md` (the
mechanism), `closest-prior-art.md` (event + SNN prediction).

## 1. Repetition counting — "is it repetitive, how many cycles?"

The largest active community, and the one that will read "repetitive motion" in our title.

- **Dwibedi et al. 2020** (*CVPR*), RepNet — temporal self-similarity matrix over frame
  embeddings, with heads for period and for periodicity. Introduced **Countix**.
- **Hu et al. 2022** (*CVPR*), TransRAC — multi-scale temporal correlation; introduced
  **RepCount**, the only benchmark annotating the start and end of *each* cycle.
- **UCFRep** — 526 videos, 23 cyclic classes. Later work: Every Shot Counts (exemplars,
  arXiv:2403.18074), Dynamic Queries (arXiv:2403.01543), IVAC-P²L (arXiv:2403.11959),
  CountLLM (arXiv:2503.17690).

**Their two metrics**, both whole-clip integers:
- **MAE** = mean over clips of `|count_pred − count_true| / count_true`
- **OBO** (off-by-one) = fraction of clips within ±1 of the true count

Neither applies to a running prediction, so we do not report them; related work says why in
one line. **arXiv:2411.08878** is the caveat to cite if a reviewer asks for counting
numbers: the field's results do not reproduce across implementations and protocols.

## 2. Periodicity and frequency estimation — the period alone

- **Cutler & Davis 2000** (*TPAMI* 22(8)) — track the object, build the self-similarity
  matrix over time, read the period from the peak of the average power spectral density.
  Still the standard learning-free estimator. A topological variant exists
  (persistent homology, arXiv:1704.08382).
- **Frequency Cam** (arXiv:2211.00198) — per-pixel period straight from the event stream,
  real time. The event-native version of the above.
- Event-based vibration and machine-fault work reports **dominant-frequency error in Hz**
  against an accelerometer or tachometer (e.g. 31.6 Hz on a fan).

**What this means for us:** period from events is cheap and solved. Our claim is the
*path*, not the period. These give a learning-free comparator for the period half of the
memory, scored as period error in ms or %, and a candidate row in `baseline.py`.

## 3. Trajectory prediction — where ADE/FDE come from

- **ADE** (average displacement error) — mean distance between prediction and truth over
  the predicted horizon. **FDE** (final displacement error) — the distance at its end.
  Standardised by **Alahi et al. 2016** (Social LSTM, *CVPR*) and universal since.
- **Schöller et al. 2020** (*RA-L* 5(2)) — the constant-velocity model beats most learned
  predictors on the standard benchmarks. The reason a naive floor is *reported*, not
  assumed to be bad.
- Event-native points of comparison: **arXiv:2609.00839** (UAV forecasting; its own
  baselines are constant-velocity and constant-acceleration filters, the latter slightly
  ahead) and **arXiv:2302.13796** (fast end-point prediction, FDE-only, on a robot).

**Adopted.** `metrics.displacement_error` is FDE at one horizon (pooled over a clip's
steps rather than over a set of trajectories); `metrics.ade_fde` combines the horizons;
the tables read `FDE 25 ms … FDE 200 ms` plus `ADE px`.

## 4. Phase — the gait convention

The one published convention for scoring a learned cycle's *phase*, which is what our
clock produces and what a half-period lock corrupts.

- Error is reported as **RMSE in percent of the cycle**. Published real-time systems land
  at **2.5–5 %**: 3.48 % normal / 4.31 % asymmetric gait (ICORR 2022); 5.00 ± 1.65 %
  spatial and 2.78 ± 0.97 % temporal at heel strike (IMU multi-resolution nets).
- Event timing is reported separately as MAE at a named cycle landmark.

Not yet in `metrics.py`; it is the natural way to score the clock apart from the map.

## 5. Deviation detection — three literatures, three metric sets

**(a) Time-series anomaly detection, and its metric argument.** Point-wise F1 is
meaningless on segment anomalies, and the **point-adjust F1** that replaced it is inflated
to the point that random scores beat published detectors (**Kim et al. 2022**, *AAAI*).
Current replacements: **range-based precision/recall** (**Tatbul et al. 2018**, *NeurIPS*
— explicit terms for existence, size, position, cardinality of an overlap),
**affiliation-based F** (temporal proximity rather than overlap), and **VUS-ROC / VUS-PR**
(**Paparrizos et al. 2022**, *PVLDB* — threshold-free with a tolerance buffer, robust to
fuzzy label boundaries, which ours have). Surveys: arXiv:2303.01272, arXiv:2409.13053.

**(b) Streaming detection with a latency reward — NAB.** **Ahmad et al. 2017**
(*Neurocomputing* 262) defines a window per true event; only the first detection inside it
counts, scored higher the earlier it lands, and false positives are penalised by distance
from a window. Our AUC + latency + false-alarms-per-minute triple is a hand-rolled version
of this — NAB is what to cite for it. Its detector, HTM, is also our nearest cousin in
kind: an online sequence memory that flags what it did not predict.

**(c) Video anomaly detection.** **Liu et al. 2018** (*CVPR*), future-frame prediction —
the field's statement of our principle, anomaly *is* prediction error. Frame-level AUC on
UCSD Ped2 / CUHK Avenue / ShanghaiTech: 92.9 / 90.6 / 74.7 %. Caveat to cite alongside:
arXiv:2606.29506 shows same-dataset frame AUC 0.704 falling to 0.499 across datasets,
which is the argument for scoring held-out clips once and for reporting false alarms per
minute next to AUC.

**(d) Robot execution monitoring** — the application-side cousin. arXiv:2107.14206 (model
the nominal motion of a repeated task, flag deviation, from vision); on-line
expectation-based novelty detection for mobile robots (*RAS* 2016); arXiv:2404.12134
(abnormal *cycles* in a repetitive task, with warping between them). Closest published
problem statements to ours, all on frames or proprioception.

## What the floor actually measures

`baseline.Extrapolator` is the naive comparator this scan argued for: a least-squares
polynomial over the last `fit_s` of observations, extended to t + horizon. No period, no
remembered path — it fills the prediction and deviation columns only.

Development set, 8 clips, centroid input, offset removed, median px
(`scripts/evaluate.py --set development`):

| memory | FDE 25 ms | FDE 50 ms | FDE 100 ms | FDE 200 ms | ADE | path px | AUC | latency s |
|---|---|---|---|---|---|---|---|---|
| kalman | 10.7 | 11.9 | 13.7 | 16.2 | 13.1 | 14.1 | 0.86 | 0.42 |
| harmonic | 21.4 | 22.1 | 23.4 | 24.5 | 22.9 | 10.0 | 0.72 | 0.14 |
| constant velocity | 10.1 | 12.2 | 23.7 | 47.4 | 23.3 | — | 0.41 | inf |
| constant acceleration | 12.7 | 15.5 | 26.9 | 74.3 | 32.4 | — | 0.45 | inf |

Three things follow.

**The horizon has to be argued.** At 25 ms the naive floor is the best predictor in the
table; at 50 ms it is level. The memory only pays from 100 ms and wins decisively at
200 ms. Any result quoted at 25 or 50 ms is a result a constant-velocity line matches, so
the headline horizon is 200 ms and the crossover is a finding, not a footnote.

**Deviation detection is where the memory is unambiguous.** Both extrapolators sit below
chance and never flag a break — they have no expectation to violate. That is the clean
zero-point for the detection column.

**Constant acceleration loses here, unlike in the UAV paper.** Even at its own best window
it is behind constant velocity at every horizon. The cause is a real difference in setup,
not a bug: that work filters a full bounding-box state, while our position is a raw event
centroid on hand-labelled clips, so the acceleration term is mostly noise and h² multiplies
it. `fit_s` is swept on the development clips and set per order in `experiment.make_memory`
— 0.1 s for order 1, 0.2 s for order 2:

| | 25 ms | 50 ms | 100 ms | 200 ms | ADE |
|---|---|---|---|---|---|
| order 1, fit 0.05 | 11.6 | 13.6 | 22.8 | 58.1 | 26.6 |
| order 1, **fit 0.10** | 10.1 | 12.2 | 23.7 | 47.4 | **23.3** |
| order 1, fit 0.20 | 13.3 | 19.6 | 26.4 | 64.8 | 31.0 |
| order 2, fit 0.05 | 21.7 | 45.7 | 129.3 | 429.3 | 156.5 |
| order 2, fit 0.10 | 15.4 | 22.4 | 46.4 | 131.8 | 54.0 |
| order 2, **fit 0.20** | 12.7 | 15.5 | 26.9 | 74.3 | **32.4** |

## Open

- A phase-error-in-%-of-cycle metric, per §4, to score the clock apart from the map.
- A second detection metric from §5(a) — range-based or affiliation F — so the deviation
  section is not resting on a bare AUC.
- Cutler–Davis period estimation as a learning-free row, per §2.
- A rhythmic-DMP row: its usual front end is the same adaptive-frequency oscillator our
  clock already uses, so it is mostly built (see `classical-baselines.md`).
