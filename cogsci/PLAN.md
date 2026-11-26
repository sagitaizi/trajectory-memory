# CogSci 2027 plan

The CogSci 2027 paper: a spiking model of how people follow a repeating motion with their
eyes, an **inner clock** with a learned map, driving a simulated eye, and compared with what
people's eyes do in published experiments.
Submission rules and dates are in `ROADMAP.md` track 2. Deadline **2027-02-01** (UTC-11);
we aim to submit by **2027-01-22**. The one-page question is
`cogsci/materials/question.md`.

## The claim

People's eyes fall into step with a repeating motion within a few cycles, with almost no lag.
The account modelled: an inner rhythm falls into step with the motion, and what to do at each
point of the rhythm is learned.

The question is how close a spiking model of this account gets to the way people's eyes
predict motion. It is built from what is known of the eye and brain, adjusted on some human
data and tested on the rest (D6). It goes through the situations people were tested in
(start of the motion, target hidden, target stops, rhythm changes, one odd cycle, motion that
cannot be predicted). The result is a table: situation × (people, model, reaction alone),
where reaction alone is the same eye without the memory.

What makes it new against the MCSoC paper: the memory works in eye velocity inside a closed
eye loop, the damped clock, the situations, and the comparison with people. The MCSoC memory
is cited (in the third person) as where the clock design came from. Other accounts of
prediction (Barnes's stored velocity released on time, Kalman-style models) are discussed,
not built.

Goals: a 6-page paper a cognitive scientist reads as a test of a model of the mind; units a
pursuit researcher uses (gain, lag in ms, cycles).

Non-goals: new recordings; training on real recordings; human experiments of our own;
catch-up saccades; real motors (the eye is simulated; the camera stays fixed); energy, Loihi
and real-time claims.

Anti-goals: claiming the model *is* how the brain does it; picking the human findings that
suit it; counting a situation it was adjusted on as evidence for it.

## The system

```
camera → localiser (unchanged) → target position p
                                      │
           gaze g ←── simulated eye ←─┤  retinal position r = p − g
             │        (delay, plant)  │  retinal slip      ṙ, both seen 100 ms late
             └─ eye velocity copy ────┤
                                      ▼
                     memory: clock and map
                     clock input = target position (retina + gaze)
                     stores      = target velocity rebuilt as slip + eye velocity
                     output      = eye velocity, read ahead to make up the delay
                     reaction    = retinal slip, added to the output
```

- **Localiser:** unchanged, frozen. The paper calls its output a population code over image
  position and makes no claim about it.
- **Simulated eye** (`pursuit/eye.py`): a gaze point over the image, a visual delay and
  an eye plant, with values taken from the pursuit literature.
- **Clock memory** (`pursuit/oscillator.py`, `pursuit/clock.py`, from
  `trajmem/snn_phasemap.py`): the clocks and a ring of phase cells; the map now holds target
  velocity at each phase, learned by PES from the rebuilt target velocity, paired with the
  clock's present phase (D3). The oscillator is
  damped, and the output is scaled by its swing (D3, gaps).
- **Surprise signal:** the rebuilt target velocity far from what a slow copy of the map
  expects, with the MCSoC threshold rule (`trajmem/metrics.py`) (D5).
- **Loop:** `pursuit/loop.py`, modelled on `trajmem.experiment.evaluate_clip` (which stays
  the MCSoC loop). It already has what the hidden-target test needs: `blank=(t0, t1)` hides the
  position for a stretch of time.

## Design questions

D1-D4 decided together, one at a time; D5 and D6 are Claude's calls at Sagi's request,
checked by him. Nothing in the SNN is built before its question is settled.

| # | Question |
|---|---|
| D1 | The eye: delay and plant values; does the eye also have a plain reactive path (follow the slip) beside the memory, as pursuit models do? |
| D2 | Rebuilding target velocity from noisy positions: how much smoothing, and where. |
| D3 | Clock memory: what the clock locks on to (target velocity or position), what the map learns from (slip or rebuilt velocity), which MCSoC position-only parts stay. |
| D4 | What is spiking and what is computed directly. |
| D5 | The surprise signal in velocity, and its threshold. |
| D6 | Settings: which are adjusted, on what, and which runs set them. |

Decided:

- **D1, what moves the eye:** a reaction to the retinal slip (always on, one visual delay
  late) plus the memory's predicted eye velocity, summed. The reaction without
  the memory is the baseline.
- **D1, visual delay:** 100 ms, fixed (human pursuit onset 100-130 ms in reviews, models
  80-100 ms). The free human onset number is 141 ± 15 ms (Brostek et al. 2017), which may
  include more than the delay. The main comparison is re-run at 80, 120 and 140 ms as a
  robustness check.
- **D1, eye response:** the eye moves at the commanded speed at once. The eyeball itself is
  slow (time constants ~170 and 13 ms, Robinson 1976), but the cerebellar drive cancels it
  (Krauzlis & Lisberger 1994), so all the slowness sits in the delay. Sources to verify.
- **D1, the reaction:** the delayed slip changes the eye's speed gradually (eye acceleration
  ∝ slip), as in pursuit models; a direct "speed ∝ slip" loop is stable only below half the
  target's speed. Its strength comes from published measurements of the eye's first ~100 ms
  after a target starts moving, before feedback or prediction act, so no situation's data
  sets it. Until that number is sourced: the strongest value that does not make the eye
  overshoot and ring at the 100 ms delay.
- Delay, response and reaction strength are measured human values, not adjusted (D6).
- **D2, velocity:** the localiser's raw 5 ms velocity is noisy, 0.85-2.1x the pendulum's own
  speed (`scripts/cogsci/velocity_noise.py`; 50 ms of smoothing brings it to 0.3-0.6x at ~50
  ms of extra delay). Velocity is smoothed lightly (~20 ms) and fed to the memory, whose
  slow learning over many cycles averages the rest. The reaction takes raw slip, since it
  changes the eye's speed gradually. Checked early in the build; if the learned velocity is
  too noisy, the memory takes positions instead and reads eye speed from the learned path.
- **D3, what the clock locks on to:** the target's position along its main direction, rebuilt
  as retinal position + gaze, as in the MCSoC clock. Not slip, which vanishes once the eye
  keeps up. Its peaks are the turnarounds, where tapping studies suggest people take a
  visual rhythm from (Iversen et al.; Gan et al. 2015; Hove et al. 2013; to verify), which
  the paper can point to. Turnaround pulses (closest to that evidence) stay a possible
  follow-up.
- **D3, what the map learns from:** the rebuilt target velocity (slip + the eye's own
  speed), paired with the clock's phase at that moment. The map holds the whole motion and
  the reaction fixes what is left. Learning from slip (the cerebellar picture) was set
  aside: the work between map and reaction could split arbitrarily.
- **D3, the delay is made up once, by reading ahead:** slip and retinal position arrive 100 ms
  late, so the eye-velocity copy and the gaze they are added to are delayed by the same
  100 ms. The clock locks to that late position and the map pairs late velocity with late
  phase, so the two agree; the output reads the map at phase + rate × delay, as the MCSoC
  memory predicts. The model is taken to know its own delay (the paper says so); the 80, 120 and
  140 ms re-runs change both together. Pairing with an earlier phase was set aside: the
  shift it learns belongs to the old rhythm (about 20 ms late for two cycles after a 2 s to
  1.4 s change, in a noise-free check with an ideal clock). An undelayed eye copy breaks
  tracking in the same check.
- **D3, which MCSoC parts stay:** kept: six clocks started 0.8-7 s apart (a small bank of
  oscillators, as in neural resonance models; interval-tuned cells in SMA/putamen, Merchant
  et al.; to verify), fast learning at the start (values under D6), the start kick.
  Dropped: the outlier gate (it would fake overshoot and late alarms when the target stops
  or breaks), the map stretching (back only if the pendulum's shrinking swing needs it),
  the short-term correction term (the reaction does it). The slow copy is decided in D5.
- **D3, picking a clock:** the clock whose map predicts best (`elect` other than
  `zero_crossings` in `SpikingPhaseMap`), like the most resonant oscillator in a bank. The
  MCSoC rule, timing the period from crossings, is interval timing, a
  different account. Its lock-on speed and accuracy are checked in the build: when the slow
  copy is saved, the best-fit clock sits at 0.87-0.88x the period on two of seven development
  runs (the MCSoC rule: 0.91 and 0.95x on the same two).
- **D3, saving the slow copy:** the clock's own settle rule (at least three cycles on the
  chosen clock and its error no longer falling, or the time limit), without the crossing-timed
  period the MCSoC clock also waits for (`_settled` in `trajmem/snn_phasemap.py`). With
  best-fit clocks, dropping it changes nothing on six of seven development runs; on the
  seventh (a loop) the slow copy is saved at 8 s instead of at the 20 s limit. Rechecked in
  the eye loop.
- **D3, the damped clock in spiking cells:** the target's drive pushes the oscillator along one
  axis and the swing decays by itself (a forced damped oscillator, as in Large's models), so
  the push (its strength is set by the damping) also sets how strongly the phase follows. At
  the published 1500 cells and 20 ms synapse the firing noise hides the damping; 3000 cells and
  a 50 ms synapse fade to 0.29-0.42 after one damping time (0.37 wanted). The undamped setting
  is the MCSoC clock itself.
- **D3, reading ahead follows the clock's own dynamics** over the delay (pushed as expected, or
  fading if the target is unseen). Turning the phase alone made the velocity 10-20% too fast,
  since the push makes the swing ripple within a cycle.
- **D4, what is spiking:** the clock's oscillator and phase cells, and the map. Computed
  directly: the main direction of motion, the clock's rate update, choosing a clock, reading
  the cells ahead by the delay. The paper lists these and notes each could be built from
  neurons. The eye, the delay and the rebuilt velocity stand for body and early vision and are
  not neurons.
- **D3, gaps and stops:** the clock is a damped oscillator that keeps its swing only while
  the target drives it, and the map's output is scaled by the swing, so with no target the
  drive fades with the clock's damping time (as in Large's resonance models). People slow to
  a reduced speed in a gap (Becker & Fuchs 1985). No free human number exists for a hidden
  repeating target, so the damping is not adjusted: S3 and S4 are run with the swing falling
  to about a third in half a cycle, one cycle, two cycles and never (the MCSoC clock), and
  all four are reported. One cycle is the value everywhere else, fixed before any run. S3
  tests a trend: the eye goes on through a turnaround, slower as the gap lasts. Gaps start
  at eight evenly spaced points of the cycle; the average and each start are reported. Eye
  velocity in a gap is one ratio over the window against the eye's own velocity at the same
  points of the cycle before, not point by point (velocity passes zero at turnarounds). The
  damping departs from the MCSoC clock; its lock-on is rechecked in the build.
- **D5, the surprise signal** (Claude's call, checked by Sagi): the memory keeps a slow copy
  of its map, as the MCSoC memory does, consolidating once the map has settled. The score is
  how far the rebuilt target velocity is from what the slow copy expected, smoothed over 0.2
  s as in MCSoC. The slow copy matters because the map itself re-learns within a few cycles
  and would absorb the surprise; it matches the slowly updated memory trace that mismatch
  responses are compared against. The alarm is the MCSoC ratcheting rule
  (`trajmem.metrics.RatchetAlarm`): the bar tightens as the memory settles and never
  loosens; its constant k and floor are set in D6. Slip was set aside as the score: it mixes
  in the reaction, which hides the memory's own error. Velocity is noisy, so false alarms on
  steady runs are checked in the build; if there are too many, the score's smoothing goes
  up.
- **D6, settings** (Claude's call, checked by Sagi):
  - Fixed: the eye's delay, response and reaction strength (D1), the 20 ms velocity smoothing
    (D2), the score smoothing (D5), the clock's damping (one cycle; D3, gaps) and cells (D3),
    its rate constant and period range as published in MCSoC.
  - Adjusted: the map's learning-rate schedule (start rate, end rate, how fast it falls), the
    alarm's k and floor.
  - Human data in two parts, split before any run. *Adjust on:* steady following (S2, gain
    and lag at one period, 4 s, in simulation; Brostek et al. 2017). *Test on:* S1, S2's other
    periods, S3-S8.
  - The score: closeness to the adjust-on human values, with no false alarms on steady runs.
  - Result (`scripts/cogsci/tune.py`; to check): all 27 learning-rate schedules tried give an
    eye gain within the human spread at 4 s, and a lag under about 10 ms once the localiser's
    own lag (21-29 ms on slow swings, ~3 ms on real runs) is taken out. The data cannot tell
    them apart, so the published schedule stays. k = 8 and no floor: the smallest k with no
    false alarm on the steady development runs, over three neuron builds (unchanged with the
    start-up wait removed). One 4 s simulated
    swing is never quiet: the localiser loses the target at its slow turnarounds.
  - Set on the MCSoC development pendulum runs and a few simulated clips; frozen before any
    test-on run. Grids and chosen values go in the paper or its supplement, and the results
    table marks which situations were adjusted on.

## Phases

### 1. Question and first design — Nov 21 – Nov 26

- [ ] Sagi signs off `cogsci/materials/question.md`.
- [x] D1-D6 settled.
- [ ] Download the CogSci template; Sagi and Elishai create PCS accounts (by 2027-01-08,
      two weeks before our submission).

### 2. Reading — Nov 22 – Dec 6

Notes in `cogsci/materials/`, one file per topic; every paper used goes into
`BIBLIOGRAPHY.md` in the same change. Author lists and years below are from memory and are
**verified before citing**. Paywalled papers are not obtained: a number is used only if it comes
from a paper readable in full for free, or from a free paper that reports it; abstract-only
sources are used for trends, not numbers.

**Sagi's list: to understand the account and its rivals** (about 8 papers; can share time with the
learning sessions in `ROADMAP.md` track 3):

| Topic | Start with |
|---|---|
| How pursuit works, and its models | Robinson et al. 1986; Lisberger et al. 1987; a recent review (Kowler et al. 2019) |
| Prediction in pursuit, Barnes's account | Barnes 1991 (cited); Barnes & Asselman 1991; Barnes 2008 |
| Pursuit when the target vanishes | Becker & Fuchs 1985 |
| The clock account: rhythm as an entrained oscillator | Large & Jones 1999; Lakatos et al. 2008 |
| Keeping time with a beat | Repp 2005 |
| Learning from slip in the cerebellum | Lisberger's work on pursuit learning; Marr–Albus–Ito in a review |
| Prediction errors and surprise | Rao & Ballard 1999; Clark 2013 (cited) |
| A spiking model presented to this audience | Eliasmith et al. 2012 (Spaun) |

**Claude's list: the human numbers.** For each, the value, its spread, the stimulus (period,
amplitude, speed) and the source:

| Situation | Number wanted |
|---|---|
| Eye loop | visual delay (141 ± 15 ms, Brostek et al. 2017); eye plant time constant; eye acceleration per unit target velocity in the first ~100 ms of pursuit (sets the reaction, D1) |
| Start of motion | cycles until gain is near 1 and lag near 0 for a sinusoid; dependence on period |
| Steady motion | gain and lag across periods; where prediction stops working |
| Target hidden | how eye velocity decays through a gap, and over what time |
| Target stops | how long and how far the eye carries on after an unexpected stop |
| Rhythm changes | cycles to adapt after a step change in period |
| One odd cycle | whether the next cycle copies it |
| Unpredictable motion | gain and lag when the motion cannot be predicted (the floor) |

Also: CogSci papers from 2022–2026 modelling pursuit or rhythm prediction.

**Done when** each situation has at least one sourced number, or is dropped with the
reason noted, and the adjust-on values (D6) are sourced from free papers.

### 3. Build — Nov 24 – Dec 10

In order, each one tested before the next:

- [x] `pursuit/eye.py` and `pursuit/loop.py`, tested with a "perfect" memory that outputs the true
      target velocity. On the development pendulum runs (`scripts/cogsci/check_eye.py`, 100 ms
      delay): perfect memory gain 0.99, lag within 5 ms; reaction alone gain 0.80, lag ~260 ms.
- [x] Velocity rebuilt from the localiser (D2; `pursuit/velocity.py`). The learned velocity is
      clean (error 0.14-0.18x the speed on the development pendulum runs, against ~1x going in),
      so the positions-in fallback is not needed.
- [x] Clock memory (D3). On the three steady development pendulum runs the clock sits at
      0.98-0.99x the period; memory gain 0.97-1.01, lag 2-8 ms; eye gain 1.01-1.03
      (`scripts/cogsci/check_clock.py`). The velocity smoothing (+17 ms) and the map's own lead
      cancel, so nothing is added to the read-ahead.
- [x] Surprise signal (D5), NaN until the slow copy is saved (3-4 s on these runs).
- [x] Settings (D6).
- [x] No start-up wait (the MCSoC memory waited 3 s, which would have set S1's answer).
      Instead the memory moves the eye only while its clock's read-ahead one visual delay
      earlier beats assuming the target kept its speed (decided with Sagi). Steady runs: on
      99-100% of the time, gain and lag unchanged. Clean sinusoids, first cycles: eye gain at
      most 1.24 (1.36 without the check), but the memory drops out for part of cycles 3-6 and
      the eye falls to 0.45-0.8 then (Open).
- [ ] Sagi checks D6.

Code goes in `pursuit/`, tests in `tests/pursuit/`; `trajmem/` and `runs/paper/` stay as
the MCSoC paper left them (`cogsci/CLAUDE.md`).

### 4. Runs — Dec 7 – Dec 23 (paused Dec 14–17 for MCSoC)

The model and the reaction alone, same eye, same input. Outputs to `runs/cogsci/`, one script per situation
(`scripts/cogsci/`), every quoted number in `runs/cogsci/numbers.csv`.

| # | Situation | On | Measure |
|---|---|---|---|
| S1 | Start of motion | steady real runs | cycles until gain and lag settle |
| S2 | Steady motion (one period adjusts) | steady real runs; sim across periods | gain and lag; the period range where prediction holds |
| S3 | Target hidden 0.5, 1, 2 cycles, from 8 starts, 4 dampings | steady pendulum runs (`blank=`) | eye velocity through the gap, as a fraction of before |
| S4 | Target stops, 4 dampings | sim (no real run has a clean stop) | how long and how far the eye carries on |
| S5 | Rhythm changes | the pendulum push run; sim period steps | cycles to adapt |
| S6 | One odd cycle | sim | how much of it shows up in the next cycle |
| S7 | Unpredictable motion | sim (random path) | gain and lag against the reactive path alone |
| S8 | Surprise | the six deviation runs | alarm latency, ms and fraction of a cycle; false alarms |

**Done when** every kept situation has a human number, a model number and a reaction-alone number.

### 5. Skeleton — Dec 18 – Dec 23

`cogsci/paper/` in the CogSci LaTeX template, anonymised from the first line. Written for
this audience, not copied from MCSoC. Six pages, references extra:

| Section | Content | Pages |
|---|---|---|
| Abstract | the account, the model, the comparison with people, the outcome | — |
| 1 Introduction | following a repeating motion; prediction by an inner clock; why a working model helps | 0.6 |
| 2 Background | pursuit and its delay; prediction in pursuit; the clock account; rival accounts; what each implies | 0.8 |
| 3 Model | the eye loop; the front end in one paragraph (cited); the clock memory; one figure | 1.3 |
| 4 Method | the recordings and simulations; each situation and its human counterpart; units and match rule | 0.6 |
| 5 Results | the comparison table; one subsection per situation; figures for the gap and the rhythm change | 1.6 |
| 6 Discussion | where the model fits people, where it does not, and why; rival accounts; predictions for new experiments; limits (simulated eye, no saccades, one target) | 0.9 |
| AI-use statement, acknowledgements | required by CogSci | 0.2 |

**Done when** Sagi has approved the outline with bullet points and sources per section.

### 6. Writing — Dec 24 – Jan 10

- Section by section; Claude writes a section only when Sagi names it, as with `paper/`.
  Status per section in `cogsci/PAPER_PROGRESS.md`.
- Suggested order: 3 Model → 4 Method → 5 Results → 2 Background → 6 Discussion →
  1 Introduction → Abstract.
- [ ] Full draft to Elishai by **2027-01-11**.

### 7. Revision and submission — Jan 11 – Jan 22

- [ ] Elishai's comments in.
- [ ] Anonymity check: no names, lab, university, funding, repo links or "our previous
      work"; file named `<first two title words>_<date>.pdf`; PDF metadata clean.
- [ ] Page check in the unchanged template: 6 pages, appendices inside them, fonts embedded.
- [ ] AI-use statement written.
- [ ] Decide the travel grant (only at submission) and who attends.
- [ ] Optional anonymised supplement (OSF) for what does not fit.
- [ ] Submit through PCS by **2027-01-22**.

## Calendar

| Dates | Main work | Also on |
|---|---|---|
| Nov 21–26 | question; design | reading starts |
| Nov 24–Dec 10 | build, reading | talk video due Dec 5 |
| Dec 7–13 | first runs | — |
| Dec 14–17 | — | MCSoC conference |
| Dec 18–23 | remaining runs, skeleton | — |
| Dec 24–Jan 10 | writing | draft to Elishai Jan 11 |
| Jan 11–22 | revision, submission | PCS accounts by Jan 8 |

## Open

- The memory switching in and out in cycles 3-6 of a clean sinusoid (see Build). Two causes:
  the clock is fixed when the slow copy is saved, three cycles after the *first* choice (as in
  MCSoC, which chose only after its 3 s wait), so now at 3-4 s, before it has settled; and the
  keep-speed error swings within a cycle (0.2 s smoothing). Counting from the latest switch
  instead never saves the slow copy on real runs (two clocks of nearly the same period trade
  places). Both are D3 choices for Sagi.

- S3: with the target hidden there is no slip, so the reaction keeps the speed it had. Beside a
  memory 10% too slow that is 1-11% of peak speed; the reaction alone glides on at 16-68%.
  Looked at again once the clock runs in the loop.
- Who attends Bilbao; the travel grant.
- Whether any real recording has a clean stop (decides whether S4 is real or sim).
