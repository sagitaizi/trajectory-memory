# The CogSci question (draft)

Working note for the CogSci 2027 paper (`cogsci/PLAN.md`). Not paper text. Sources are
from memory until checked in the reading phase.

## The question

When people follow a repeating motion with their eyes, the eyes soon move in step with it,
with almost no lag, so the brain must predict. One account: an inner **clock** falls into step
with the motion, and what to do at each point of the rhythm is learned.

> Built as a working spiking model that drives a simulated eye, how close does this account
> get to people's eyes, in which situations, and why?

## The situations

| # | What people do (to source) | What we measure in the model |
|---|---|---|
| S1 | **Start.** Lag disappears within a few cycles of a sinusoid (Barnes 1991). | Cycles until gain and lag settle. |
| S2 | **Steady.** Gain near 1, lag near 0, over a limited range of periods. | Gain, lag, and the range where they hold. |
| S3 | **Hidden.** The eye carries on through a short gap, slowing as it lasts (Becker & Fuchs 1985). | Eye velocity through gaps of 0.5, 1, 2 cycles. |
| S4 | **Stop.** The eye carries on briefly past an unexpected stop. | How long and how far. |
| S5 | **Rhythm changes.** Adaptation over a few cycles (pursuit; tapping, Repp 2005). | Cycles to adapt. |
| S6 | **One odd cycle.** Does the next cycle copy it? | How much of it carries over. |
| S7 | **Unpredictable motion.** No prediction; lag is the delay. | Gain and lag against the reactive path alone. |
| S8 | **Surprise.** A broken pattern is flagged within ~100–250 ms (mismatch responses). | Alarm latency, ms and fraction of a cycle. |

Expected (to check, not assumed): after a change of rhythm the clock adapts gradually (S5),
and one odd cycle is mostly smoothed away (S6).

## Match or miss

Decided before the runs:

- **Match:** the model's value falls in the human range in the same units (gain, lag in
  ms, cycles), *or* shows the same trend (for example, slowing more in a longer gap).
- **Miss:** outside the range, or the opposite trend. Misses are reported, and the
  Discussion says which part of the model causes each.
- The model is adjusted on steady following (S2, one period) and tested on everything else.
  Only test situations count as a match or a miss; the split is fixed before the runs.
- Every situation is also run with the reaction alone (the same eye without the memory), to
  show what the memory adds.
- The hidden target (S3) is tested as a trend, over a range of dampings, since no free human
  number exists for a repeating target.
- A situation with no human number we can source is dropped, with the reason noted.

## Predictions for new experiments

Examples, to check on the model first:

- The eye through a gap should improve with the number of cycles seen before it, and stop
  improving once the rhythm is locked on.
- After a change in period, the path's *shape* should carry over while its *timing* adapts.

## What we do not claim

- That the brain uses this mechanism: we show what the account does when built and where it
  agrees with people. A mixed outcome is reported as one.
- A realistic eye: the eye is simulated (a delay and a plant), with no catch-up saccades;
  the camera is fixed and the retinal image is the target's position minus the gaze.
- That the localiser is biological: it stands in for early vision, is trained by
  backpropagation on simulation, and its output is described as a population code over
  image position (not "place cells"). The claims are about the memory only.
