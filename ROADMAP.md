# Roadmap

What comes after the MCSoC acceptance. Dates are in `TIMELINE.md`.

| Track | Goal | By |
|---|---|---|
| 3 | Learning sessions: relearn the paper section by section, then the talk | 2026-12-14 (conference) |
| 2 | A CogSci 2027 version: the clock model compared with how people's eyes predict motion | 2027-02-01 (UTC-11) |

Track 1, the MCSoC camera-ready, is done (uploaded, copyright form and registration in).

## 2. CogSci version

Elishai's suggestion: learn how the brain predicts trajectories, add a dedicated comparison
with biology, and keep the rest of the paper.

- **Overlap rule.** CogSci "will not consider any paper that substantially overlaps with
  another paper which has been published in a journal or another conference" at submission
  time. The MCSoC paper will be in IEEE proceedings by December, before the CogSci deadline,
  so the CogSci paper cannot be the MCSoC paper with one section added. It needs a
  cognitive-science question of its own, new material, and a citation of the MCSoC paper.
- **Plan (Elishai has given the go-ahead).** A spiking clock model of how people follow a
  repeating motion with their eyes, driving a simulated eye, compared with published human
  findings. No new recordings; new runs on the existing ones where the comparison needs them. Authors as on
  MCSoC. Who attends Bilbao, and the travel grant, are open (decide before Feb 1).
- **Format.** 6 pages including appendices, references unlimited, CogSci's own template
  (changing margins or font is a desk reject). Submitted as PDF through PCS; every author needs
  a PCS account at least two weeks before the deadline. Proceedings are CC-BY and not
  archival, with an abstract-only option, so a journal version can follow.
- **Double-blind, strictly.** No author names, no self-identifying references: the MCSoC paper
  is cited in the third person. File named `<first two title words>_<date>.pdf`.
- **Other rules.** The paper must "relate to the study of the mind". AI use must be described
  in the Methods or Acknowledgements. Bilbao, 2027-07-28 to 31, in person; one author must
  register and attend. A student first author can apply for a travel grant, but only at
  submission time, and is eligible for the student-paper and modelling awards.
- **Dates.** Decisions early April 2027; camera-ready 2027-05-07.
- **The plan** — the claim, reading lists, runs, outline and calendar — is in
  `cogsci/PLAN.md`.

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
- The CogSci project lives in `cogsci/` (docs, notes, paper) and `pursuit/` (code), in this
  repo because both papers draw on the same data, localiser and code. `pursuit/` imports
  `trajmem/` and does not change it. Like `paper/`, Claude edits `cogsci/paper/` only when
  Sagi names the section.
- `runs/paper/` and `runs/paper/numbers.csv` are what the MCSoC paper quotes and are never
  overwritten. New runs for CogSci go to `runs/cogsci/`.
