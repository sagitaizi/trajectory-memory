# CogSci 2027 plan

The CogSci 2027 paper: two spiking models of how people follow a repeating motion with
their eyes, an **inner clock** and a **replay of the last cycle**, each driving the same
simulated eye, and compared with what people's eyes do in published experiments.
Submission rules and dates are in `ROADMAP.md` track 2. Deadline **2027-02-01** (UTC-11);
we aim to submit by **2027-01-22**. The one-page question is
`cogsci/materials/question.md`.

## The claim

Two accounts of how the brain predicts a repeating motion:

- **Clock.** An inner rhythm falls into step with the motion; what to do at each point of
  the rhythm is learned.
- **Replay.** The brain keeps a record of the last cycle, times how long a cycle takes, and
  plays the record again on time (close to Barnes's model of predictive pursuit).

Each is built as a spiking memory that tells a simulated eye how fast to move. Both are put
through the same situations people were tested in (start of the motion, target hidden,
target stops, rhythm changes, one odd cycle, motion that cannot be predicted), and the eye's
behaviour is set against the human numbers. The result is a table: situation × (people,
clock, replay).

What makes it new against the MCSoC paper: the memories work in eye velocity inside a
closed eye loop, the replay model, the situations, and the comparison with people. The MCSoC
memory is cited (in the third person) as where the clock design came from.

Goals: a 6-page paper a cognitive scientist reads as a test of two accounts of the mind;
comparisons in units that mean the same for both (pursuit gain, lag in ms, cycles).

Non-goals: new recordings; training on real recordings; human experiments of our own;
catch-up saccades; real motors (the eye is simulated; the camera stays fixed); energy, Loihi
and real-time claims.

Anti-goals: claiming either model *is* how the brain does it; picking the human findings
that suit one model; tuning one model more carefully than the other.

## The system

```
camera → localiser (unchanged) → target position p
                                      │
           gaze g ←── simulated eye ←─┤  retinal position r = p − g
             │        (delay, plant)  │  retinal slip      ṙ, seen ~100 ms late
             └─ eye velocity copy ────┤
                                      ▼
                     memory: clock  or  replay
                     input  = target velocity rebuilt as slip + eye velocity
                     output = eye velocity, read ahead to make up the delay
                     learns from = retinal slip
```

- **Localiser:** unchanged, frozen. The paper calls its output a population code over image
  position and makes no claim about it.
- **Simulated eye** (`pursuit/eye.py`): a gaze point over the image, a visual delay and
  an eye plant, with values taken from the pursuit literature. Shared by both memories.
- **Clock memory** (`pursuit/clock.py`, from `trajmem/snn_phasemap.py`): the same clock
  and ring of phase cells; the map now holds eye velocity at each phase, learned by PES with
  the retinal slip as the error. The position-only parts (stretching the map, the outlier
  gate) are re-decided.
- **Replay memory** (`pursuit/replay.py`): a spiking LMU (`trajmem/lmu.py`, already built)
  holds the last stretch of target velocity; the period is timed from repeated crossings;
  the output reads the record one period back, shifted by the delay. No training.
- **Surprise signal:** slip much larger than the memory expects, with the same threshold
  rule (`trajmem/metrics.py`) for both memories.
- **Loop:** `pursuit/loop.py`, modelled on `trajmem.experiment.evaluate_clip` (which stays
  the MCSoC loop). It already has what the hidden-target test needs: `blank=(t0, t1)` hides the
  position for a stretch of time.

## Design questions, decided together

Asked one at a time, in this order, each with the options in plain words. Nothing in the
SNN is built before its question is settled.

| # | Question |
|---|---|
| D1 | The eye: delay and plant values; does the eye also have a plain reactive path (follow the slip) beside the memory, as pursuit models do, and is it the same for both memories? |
| D2 | Rebuilding target velocity from noisy positions: how much smoothing, and where. |
| D3 | Clock memory: what the clock locks on to (target velocity or position), what the map learns from (slip or rebuilt velocity), which MCSoC position-only parts stay. |
| D4 | Replay memory: how long a record it keeps, how it times the period, spiking throughout or not. |
| D5 | The surprise signal in velocity, and its threshold. |
| D6 | Fairness: the same settings budget for both memories, and which runs are used to set them. |

Decided:

- **D1, what moves the eye:** a reaction to the retinal slip (always on, one visual delay
  late) plus the memory's predicted eye velocity, summed. The same reactive path for both
  memories.
- **D1, visual delay:** 100 ms, fixed (human pursuit onset 100–130 ms, models 80–100 ms;
  sources to verify). The main comparison is re-run at 80 and 120 ms as a robustness check.
- **D1, eye response:** the eye moves at the commanded speed at once. The eyeball itself is
  slow (time constants ~170 and 13 ms, Robinson 1976), but the cerebellar drive cancels it
  (Krauzlis & Lisberger 1994), so all the slowness sits in the delay. Sources to verify.
- **D1, the reaction:** the delayed slip changes the eye's speed gradually (eye acceleration
  ∝ slip), as in pursuit models; a direct "speed ∝ slip" loop is stable only below half the
  target's speed. Its strength comes from published measurements of the eye's first ~100 ms
  after a target starts moving, before feedback or prediction act, so all eight situations
  stay tests. Until that number is sourced: the strongest value that does not make the eye
  overshoot and ring at the 100 ms delay. One value, shared by both memories.
- Delay, response and reaction strength are human values, never tuned.
- **D2, velocity:** the localiser's raw 5 ms velocity is noisy, 0.85-2.1x the pendulum's
  own speed (`scripts/cogsci/velocity_noise.py`; 50 ms of smoothing brings it to 0.3-0.6x
  at ~50 ms of extra delay). Velocity is smoothed lightly (~20 ms) and fed to both
  memories, whose slow learning over many cycles averages the rest; the map still learns
  from slip. The reaction takes raw slip, since it changes the eye's speed gradually.
  Checked early in the build; if the learned velocity is too noisy, both memories take
  positions instead and read eye speed from the learned path.

## Phases

### 1. Question and first design — Nov 21 – Nov 26

- [ ] Sagi signs off `cogsci/materials/question.md`.
- [x] D1 and D2 settled.
- [ ] Download the CogSci template; Sagi and Elishai create PCS accounts (by 2027-01-08,
      two weeks before our submission).

### 2. Reading — Nov 22 – Dec 6

Notes in `cogsci/materials/`, one file per topic; every paper used goes into
`BIBLIOGRAPHY.md` in the same change. Author lists and years below are from memory and are
**verified before citing**.

**Sagi's list: to understand the two accounts** (about 8 papers; can share time with the
learning sessions in `ROADMAP.md` track 3):

| Topic | Start with |
|---|---|
| How pursuit works, and its models | Robinson et al. 1986; Lisberger et al. 1987; a recent review (Kowler et al. 2019) |
| Prediction in pursuit, the replay account | Barnes 1991 (cited); Barnes & Asselman 1991; Barnes 2008 |
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
| Eye loop | visual delay; eye plant time constant; eye acceleration per unit target velocity in the first ~100 ms of pursuit (sets the reaction, D1) |
| Start of motion | cycles until gain is near 1 and lag near 0 for a sinusoid; dependence on period |
| Steady motion | gain and lag across periods; where prediction stops working |
| Target hidden | how eye velocity decays through a gap, and over what time |
| Target stops | how long and how far the eye carries on after an unexpected stop |
| Rhythm changes | cycles to adapt after a step change in period |
| One odd cycle | whether the next cycle copies it |
| Unpredictable motion | gain and lag when the motion cannot be predicted (the floor) |

Also: CogSci papers from 2022–2026 modelling pursuit or rhythm prediction.

**Done when** each situation has at least one sourced number, or is dropped with the
reason noted.

### 3. Build — Nov 24 – Dec 10

In order, each one tested before the next:

- [ ] `pursuit/eye.py` and `pursuit/loop.py`, tested with a "perfect" memory that outputs the true
      target velocity (the eye should then track with only the delay's error).
- [ ] Velocity rebuilt from the localiser (D2), and the check that a memory's learned
      velocity is clean enough (else the positions-in fallback).
- [ ] Clock memory (D3); it should lock on to the steady pendulum runs.
- [ ] Replay memory (D4); the same check.
- [ ] Surprise signal (D5) for both.
- [ ] Settings for both under the same budget (D6), on a few development runs only.

Code goes in `pursuit/`, tests in `tests/pursuit/`; `trajmem/` and `runs/paper/` stay as
the MCSoC paper left them (`cogsci/CLAUDE.md`).

### 4. Runs — Dec 7 – Dec 23 (paused Dec 14–17 for MCSoC)

Both memories, same eye, same input. Outputs to `runs/cogsci/`, one script per situation
(`scripts/cogsci/`), every quoted number in `runs/cogsci/numbers.csv`.

| # | Situation | On | Measure |
|---|---|---|---|
| S1 | Start of motion | steady real runs | cycles until gain and lag settle |
| S2 | Steady motion | steady real runs; sim across periods | gain and lag; the period range where prediction holds |
| S3 | Target hidden 0.5, 1, 2 cycles | steady pendulum runs (`blank=`) | eye velocity through the gap, as a fraction of before |
| S4 | Target stops | sim (no real run has a clean stop) | how long and how far the eye carries on |
| S5 | Rhythm changes | the pendulum push run; sim period steps | cycles to adapt |
| S6 | One odd cycle | sim | how much of it shows up in the next cycle |
| S7 | Unpredictable motion | sim (random path) | gain and lag against the reactive path alone |
| S8 | Surprise | the six deviation runs | alarm latency, ms and fraction of a cycle; false alarms |

**Done when** every kept situation has a human number, a clock number and a replay number.

### 5. Skeleton — Dec 18 – Dec 23

`cogsci/paper/` in the CogSci LaTeX template, anonymised from the first line. Written for
this audience, not copied from MCSoC. Six pages, references extra:

| Section | Content | Pages |
|---|---|---|
| Abstract | the two accounts, the two models, the comparison, the outcome | — |
| 1 Introduction | following a repeating motion; clock against replay; why a working model helps settle it | 0.6 |
| 2 Background | pursuit and its delay; prediction in pursuit; the clock account; the replay account; what each implies | 0.8 |
| 3 Models | the eye loop; the shared front end in one paragraph (cited); the clock memory; the replay memory; one figure with both | 1.3 |
| 4 Method | the recordings and simulations; each situation and its human counterpart; units and match rule | 0.6 |
| 5 Results | the comparison table; one subsection per situation; figures for the gap and the rhythm change | 1.6 |
| 6 Discussion | which account fits where and why; mixed results; predictions for new experiments; limits (simulated eye, no saccades, one target) | 0.9 |
| AI-use statement, acknowledgements | required by CogSci | 0.2 |

**Done when** Sagi has approved the outline with bullet points and sources per section.

### 6. Writing — Dec 24 – Jan 10

- Section by section; Claude writes a section only when Sagi names it, as with `paper/`.
  Status per section in `cogsci/PAPER_PROGRESS.md`.
- Suggested order: 3 Models → 4 Method → 5 Results → 2 Background → 6 Discussion →
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
| Nov 21–26 | question; D1, D2 | reading starts |
| Nov 24–Dec 10 | build (D3–D6 along the way), reading | talk video due Dec 5 |
| Dec 7–13 | first runs | — |
| Dec 14–17 | — | MCSoC conference |
| Dec 18–23 | remaining runs, skeleton | — |
| Dec 24–Jan 10 | writing | draft to Elishai Jan 11 |
| Jan 11–22 | revision, submission | PCS accounts by Jan 8 |

## Open

- Who attends Bilbao; the travel grant.
- Whether any real recording has a clean stop (decides whether S4 is real or sim).
