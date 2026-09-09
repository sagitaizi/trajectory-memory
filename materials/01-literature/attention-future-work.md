# Attention span — future work only

Not in the MCSoC paper. Notes so the future-work paragraph is credible and the roadmap is
grounded.

## The idea (`future-directions.md` §5b)
Multiple moving objects; the network attends to the most "interesting" motion. Interest is
dynamic: repetitive motion **habituates** (becomes less interesting); a repetitive motion
that **breaks its pattern** becomes interesting again (the §5a deviation signal is the
salience trigger); a **newly introduced** object is interesting until it becomes predictable.

## Canonical computational model
Itti & Koch 2001, "Computational modelling of visual attention", *Nature Reviews
Neuroscience* 2(3):194–203:
- **Saliency map** over the visual field.
- **Winner-take-all** selects the current focus.
- **Inhibition of return (IOR)** — the just-attended location is transiently suppressed so
  attention moves on. IOR *is* a short memory of where you've been.
- Your "habituate to repetitive motion" is a temporal generalisation of IOR — inhibition
  keyed to a learned pattern, not just a location.

## Related, more recent
- "Look twice: a generalist model predicts return fixations across tasks and species", *PLOS
  Comp. Biol.* 2022 — models when gaze returns to a previously-fixated thing.
- Robotics-inspired scanpath models for dynamic scenes (arXiv 2408.01322, 2024).
- "Objects guide human gaze in dynamic real-world scenes" (biorxiv 2023).
- **BioSpike-Net** — event-driven SNN with a *novelty-gated temporal attention* mechanism
  concentrating compute on surprising input (physiological signals, not vision). Closest
  spiking analogue of the mechanism.

## The human comparison study (Sagi's idea) — practical blockers
Show videos to people, record eye movements, compare gaze-switching to the model's attention
switching. Needs: eye-tracking hardware, participants, and university ethics/consent approval
— all unassessed, all on the critical path. Keep it as a proposed validation, not a promise.
