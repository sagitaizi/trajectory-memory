# Roadmap

What comes after the MCSoC acceptance: three tracks, in this order. Dates are in `TIMELINE.md`.

| Track | Goal | By |
|---|---|---|
| 1 | MCSoC camera-ready that answers every review | 2026-11-08 (hard) |
| 3 | Learning sessions: relearn the paper section by section, then the talk | 2026-12-14 (conference) |
| 2 | A CogSci 2027 version: the same system compared with how the brain predicts motion | ~2027-02-01 (deadline unconfirmed) |

Track 3 starts alongside track 1, since knowing each section well is what lets us answer
the reviews properly.

## 1. MCSoC camera-ready

The checklist (reviewer points, upload, copyright, registration) is in `PAPER_PROGRESS.md`.

**Open: new experiments in the camera-ready.** The camera-ready is meant to fix what the
reviewers raised. Clarifications and small analyses that answer them are normal (the
six-clocks explanation, an energy estimate from the operation counts). New results that
change what the paper claims are not fraud, but then the published paper is not the one
that was reviewed. Training on real recordings also breaks the paper's own protocol ("none
are ever trained on"): every result on those runs would have to move to new, unseen
recordings. Recommendation: the camera-ready answers the reviews only, and new recordings
or training go into the CogSci version or a later journal version.

## 2. CogSci version

Elishai's suggestion: learn how the brain predicts trajectories, add a dedicated comparison
with biology, and keep the rest of the paper.

- **Overlap rule.** CogSci "will not consider any paper that substantially overlaps with
  another paper which has been published in a journal or another conference" at submission
  time. The MCSoC paper will be in IEEE proceedings by December, before the CogSci deadline,
  so the CogSci paper cannot be the MCSoC paper with one section added. It needs a
  cognitive-science question of its own, new material, and a citation of the MCSoC paper.
  Raise this with Elishai before starting.
- **Format.** 6 pages including appendices, CogSci's own template. CogSci proceedings are not
  archival, so a journal version can follow.
- **Reading list** (first pass): smooth pursuit and its prediction of periodic targets
  (Barnes 1991 is already cited); the retina's own motion anticipation (Berry 1999);
  predictive processing (Clark 2013); place cells and phase coding in the hippocampus;
  entrainment of neural oscillations to rhythm; cerebellar forward models.
- **Possible comparisons:** lock-on time in cycles against smooth pursuit; behaviour when
  the rhythm stops or changes against the eye's predictive overshoot; the deviation alarm
  against mismatch responses in the brain.

## 3. Learning sessions

One paper section per session, in plain words. Sagi explains it back at the end, and the
session closes with the questions a listener might ask.

| # | Topic | Paper |
|---|---|---|
| 1 | The problem, and why prediction rather than reaction | §I, §II, §III-A |
| 2 | Event cameras and the 5 ms frames | §III-B |
| 3 | The spiking localiser: LIF neurons, convolution, place cells, training on simulation | §III-C, §IV-B |
| 4 | The clock: the adaptive-frequency oscillator, and why six of them | §III-D |
| 5 | The map: phase cells, the PES rule, prediction | §III-D |
| 6 | The deviation score and the alarm | §III-D, §IV-D |
| 7 | Baselines and metrics: the Kalman filter, the harmonic fit, AUC, latency | §IV-C, §IV-E |
| 8 | The results, number by number | §V |
| 9 | What is spiking, energy and hardware, and what comes next | §VI |
| 10 | The talk: slides and a dry run | — |

## Two versions of the paper

- `paper/` stays the MCSoC paper. When the camera-ready is uploaded, the commit is tagged
  `mcsoc-camera-ready` and `paper/` is frozen from then on.
- The CogSci paper lives in `paper-cogsci/`, started from a copy of the final MCSoC text. It
  is a folder rather than a git branch because both papers draw on the same code and runs.
  Like `paper/`, Claude edits it only when Sagi names the section.
- `runs/paper/` and `runs/paper/numbers.csv` are what the MCSoC paper quotes and are never
  overwritten. New runs for CogSci go to `runs/cogsci/`.
