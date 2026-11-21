# CogSci 2027 plan

The CogSci 2027 paper: the trajectory memory read as a **model of how people predict
repetitive motion**, tested against published findings from human and animal studies.
Submission rules and dates are in `ROADMAP.md` track 2. Deadline **2027-02-01** (UTC-11);
we aim to submit by **2027-01-12**, since the model, recordings and most results
come from the MCSoC paper.

## The claim

Working question (to agree with Elishai):

> Can a clock that locks on to a rhythm, plus a map of position at each phase of the cycle,
> account for the main signatures of human prediction of periodic motion: how fast it
> locks on, what it does when the target vanishes or stops, how it adapts when the rhythm
> changes, and how fast it signals a surprise?

What makes it a CogSci paper rather than MCSoC again:

- The **result is a comparison table**, human finding against model behaviour, not
  prediction error in pixels against Kalman filters.
- **New material:** the gap, stop and change experiments and the per-cycle measures
  below; none are in the MCSoC paper.
- **Testable predictions** for human experiments that follow from the model (for example,
  how prediction through a gap should depend on how many cycles were seen before it).
- The MCSoC paper is cited in the third person for the engineering details.

Goals: a 6-page paper that a cognitive scientist reads as a model of the mind; every
comparison stated in units that mean the same thing for eyes and for our system (cycles,
fractions of a period, milliseconds), never pixels against degrees.

Non-goals: new recordings; training on real recordings; human experiments of our own;
energy, Loihi and real-time claims (MCSoC material).

Anti-goals: claiming the model *is* how the brain does it; choosing findings that fit and
leaving out ones that don't; a biology section bolted onto the MCSoC paper.

## Phases

### 1. Question and agreement — Nov 21 – Nov 25

- [ ] Sagi and Claude draft the question and the list of signatures (above) into one page:
      `materials/05-cogsci/question.md`.
- [ ] Send it to Elishai with the overlap rule spelled out; agree on the claim, the
      authors (same as MCSoC), and who attends Bilbao.
- [ ] Download the CogSci template; Sagi and Elishai create PCS accounts (by
      2026-12-28, two weeks before our submission).

**Done when** Elishai has agreed the question in writing.

### 2. Reading — Nov 22 – Dec 6

Two lists. Notes go in `materials/05-cogsci/`, one file per topic; every paper used goes
into `BIBLIOGRAPHY.md` in the same change. Author lists and years below are from memory
and are **verified before citing**.

**Sagi's list: to understand the biology** (about 8 papers; can share time with the
learning sessions in `ROADMAP.md` track 3):

| Topic | Start with |
|---|---|
| Prediction in smooth pursuit | Barnes 1991 (already cited); Barnes & Asselman 1991; a review of predictive pursuit (Barnes 2008, or Kowler et al. 2019) |
| Pursuit when the target vanishes | Becker & Fuchs 1985 |
| Rhythm and attention as an entrained oscillator | Large & Jones 1999 ("The dynamics of attending"); Lakatos et al. 2008 |
| Keeping time with a beat, period and phase correction | Repp 2005 (review of sensorimotor synchronisation) |
| Prediction errors and surprise signals | Rao & Ballard 1999; Clark 2013 (cited); a mismatch-negativity review (Näätänen et al. 2007) |
| Phase codes in the brain | O'Keefe & Recce 1993 (phase precession); Buzsáki's work on theta |
| Forward models | Wolpert, Miall & Kawato 1998 |
| Spiking models at CogSci | Eliasmith et al. 2012 (Spaun), to see how an NEF model is presented to this audience |

**Claude's list: to find the numbers we compare against.** For each signature, the
measured value, its spread, the stimulus (period, amplitude) and the source:

| Signature | Number wanted |
|---|---|
| Lock-on | cycles until pursuit lag drops near zero for a sinusoidal target; how it depends on period |
| Frequency range | periods over which prediction works, and where it fails |
| Target vanishes | how long predictive pursuit continues through a gap, and how it decays |
| Target stops | overshoot after an unexpected stop: how long, how far, as a fraction of the cycle |
| Rhythm changes | cycles to adapt after a step change in period (pursuit and tapping) |
| Surprise | latency of mismatch / omission responses in ms, and against stimulus rate |

Also: CogSci papers from 2022–2026 on oscillator or predictive models of tracking, to
place the paper among them.

**Done when** each signature has at least one sourced number, or is dropped from the
paper with the reason noted.

### 3. Runs and code — Nov 24 – Dec 6

All on existing recordings (and simulation where a recording cannot show it). Outputs to
`runs/cogsci/`, never `runs/paper/`. Same frozen localiser and settings as MCSoC. Anything
that changes the SNN itself (new parts, new parameters) is decided together first.

| # | Experiment | On | Needs |
|---|---|---|---|
| R1 | Lock-on in **cycles**, clock and map separately, per run | every labelled steady run | analysis only: convert the existing lock-on times to cycles of each run's period |
| R2 | **Gap**: delete all events for 0.5, 1 and 2 cycles mid-run; does the prediction carry on, and how does it decay | steady pendulum runs | event-blanking step in `scripts/`; first check what the memory does when the localiser says "not visible" |
| R3 | **Stop**: what the prediction does when the motion stops | sim (no real recording has a clean stop); real if one turns up | a stop deviation in `trajectories.py` if it lacks one |
| R4 | **Change**: cycles for the clock to re-lock after the rhythm changes | the pendulum push run, the loop run that changes shape, sim period steps | analysis + sim clips with step changes in period |
| R5 | **Surprise latency** in ms and as a fraction of the cycle | the six deviation runs | analysis of existing alarm latencies |
| R6 | **Lock range**: the period range the clocks follow | sim period sweep | exists in part (the six clocks cover 0.8–7 s); one sweep script |

Order: R1, R5 (analysis only) → R2 → R4 → R3, R6. A script per experiment under
`scripts/cogsci_*.py`, writing CSVs and figures; a `runs/cogsci/numbers.csv` like the MCSoC
one so every number in the paper has one source.

**Done when** each kept signature has a model number next to its human number.

### 4. Skeleton — Nov 30 – Dec 4 (alongside phase 3)

`paper-cogsci/` in the CogSci LaTeX template, anonymised from the first line. Not a copy of
the MCSoC text: paragraphs are re-written for this audience, borrowing only the model
description. Proposed outline, 6 pages, references extra:

| Section | Content | Pages |
|---|---|---|
| Abstract | the question, the model, the comparison, the predictions | — |
| 1 Introduction | predicting repetitive motion in people; the open question; our answer in one line | 0.6 |
| 2 How the brain predicts periodic motion | pursuit, entrainment, phase codes, surprise; competing accounts (a stored waveform, an oscillator, a forward model) | 0.8 |
| 3 The model | clock (adaptive oscillator), phase cells and map (local learning rule), the slow copy and the surprise signal; the localiser in one paragraph, details cited to MCSoC. One figure | 1.2 |
| 4 Method | the recordings in a paragraph; each experiment and its matching human measure; units | 0.7 |
| 5 Results | one subsection per signature; the comparison table (human vs model); one or two figures (gap, change) | 1.6 |
| 6 Discussion | what the model explains, what it does not, predictions for new experiments, limits (one target, a camera not an eye, no eye movement) | 0.9 |
| AI-use statement, acknowledgements | required by CogSci | 0.2 |

Cut from MCSoC: energy and Loihi, real-time timing, the simulator, ablations, most
localiser results. Baselines shrink to one contrast: a fixed-period predictor (the Kalman
filter) against the adaptive clock, which mirrors the "stored waveform vs oscillator"
question in the pursuit literature.

**Done when** Sagi has approved the outline and each section has its bullet points and
its sources.

### 5. Writing — Dec 5 – Dec 21 (paused Dec 14–17 for MCSoC)

- Written section by section; Claude writes a section only when Sagi names it, as with
  `paper/`. Status per section in a table at the top of `paper-cogsci/README.md`.
- Suggested order: 4 Method → 5 Results → 3 Model → 2 Background → 6 Discussion →
  1 Introduction → Abstract.
- [ ] Full draft to Elishai by **2026-12-22**.

### 6. Revision and submission — Dec 22 – Jan 12

- [ ] Elishai's comments in.
- [ ] Anonymity check: no names, lab, university, funding, repo links or "our previous
      work"; file named `<first two title words>_<date>.pdf`; PDF metadata clean.
- [ ] Page check in the unchanged template: 6 pages, appendices inside them, fonts embedded.
- [ ] AI-use statement written.
- [ ] Decide the travel grant (only at submission) and who attends.
- [ ] Optional anonymised supplement (OSF) for figures or tables that do not fit.
- [ ] Submit through PCS by **2027-01-12**.

## Calendar

| Dates | Main work | Also on |
|---|---|---|
| Nov 21–25 | phase 1; R1 and R5 (analysis only) | reading starts |
| Nov 26–Dec 6 | runs R2–R6, reading, numbers table, skeleton | talk video due Dec 5; learning sessions |
| Dec 7–13 | writing | — |
| Dec 14–17 | — | MCSoC conference |
| Dec 18–21 | writing | draft to Elishai Dec 22 |
| Dec 22–Jan 12 | revision, submission | PCS accounts by Dec 28 |

## Open

- Elishai's agreement on the question and the overlap risk.
- Who attends Bilbao; the travel grant.
- What the memory does when the target is not visible (decides how much R2 needs).
- Whether any real recording has a clean stop (decides whether R3 is real or sim).
