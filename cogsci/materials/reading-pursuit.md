# Reading: pursuit, prediction, gaps and stops

Notes for `cogsci/PLAN.md` phase 2. "Read" says how far each source was read: **full**,
**abstract**, or **via review** (the claim as a review reports it; the primary paper is still
to be checked). Nothing here is quotable in the paper until it is read in full.

## The eye loop (D1)

| Fact | Value | Source | Read |
|---|---|---|---|
| Human pursuit latency | ~100–130 ms | Tychsen & Lisberger 1986; Carl & Gellman 1987 | via review (Fukushima et al. 2013) |
| Monkey pursuit latency | 80–100 ms | Lisberger & Westbrook 1985 | via review |
| Open-loop interval | first ~100 ms of pursuit; eye acceleration there estimates the open-loop response | Carl & Gellman 1987; Krauzlis & Lisberger 1994 | via review; abstract |
| Early acceleration | first 20–30 ms: rises a little with target velocity; 60–80 ms: much more | Lisberger & Westbrook 1985; Tychsen & Lisberger 1986 | via review |
| Open-loop gain as a number | not found yet; Barnes's model uses K0 ≈ 2.4, Te = 0.12 s | Barnes & Collins 2011 (Fig. 2 of the review) | via review |
| Eye plant | fast constant ~13 ms; slow one 150–660 ms depending on study | Robinson 1964 and later fits | search only |
| Model structure | three visual motion pathways, summed and integrated into eye velocity, in a negative-feedback loop | Krauzlis & Lisberger 1994 | abstract |

Open: a sourced open-loop acceleration per unit target velocity for humans (sets the
reaction's strength, D1).

## Prediction on a repeating motion (S1, S5, S6; Barnes's account)

- Square, triangular and sawtooth target waveforms, flashed 40–320 ms per half-cycle.
  Peak eye velocity builds over the **first 2–4 cycles**; time to peak falls from ~300 to
  ~200 ms (triangular). Barnes & Asselman 1991, abstract.
- After an unexpected **change of period**, the predictive movement keeps the **old
  period's timing**. Cycles to adapt: not in the abstract. Same source.
- After an unexpected **change of amplitude or direction**, the next predictive movement's
  peak velocity and direction follow the **preceding sequence**. Same source.
- Their model: magnitude and timing of eye velocity are stored; the prediction is released
  as **a stereotyped volley of constant duration but varying magnitude**, timed by a
  periodicity estimator. Same source.
- The later form (Barnes & Collins 2011, via review): the reconstructed target velocity is
  sampled for ~150 ms and held (S/H), then kept in a working memory (MEM) that can hold
  several velocity levels. Release is timed by external cues **or by direction changes in
  the reconstructed target velocity**, kept in a timing store. An expectation gain β < 1
  scales the predictive drive.
- Barnes 2008 (abstract): velocity and timing from earlier motion are held in short-term
  memory; in periodic pursuit "partial trajectory information is stored and released".

Barnes's account, a velocity per direction released on time, is the rival the paper discusses.

## Target hidden (S3)

- Eye velocity does not stop; it is sustained at a reduced level, and often recovers before
  the target is expected back. Becker & Fuchs 1985; Bennett & Barnes 2003; via review.
- Reported elsewhere (to check in Becker & Fuchs): the drop starts ~190 ms after
  disappearance and levels off by ~280 ms at a residual of ~60% of pursuit velocity.
- Without an expectation that the target comes back, velocity falls towards zero
  (Mitrani & Dimitrov 1978; Bennett & Barnes 2004; via review).
- The residual level rises with target velocity (Barnes & Collins 2008, via review).
- **Motion in these studies is a ramp, not a repeating one.** Becker & Fuchs 1985 hid a spot
  moving at constant speed, at random times, for varying durations (abstract). Bennett &
  Barnes 2003 and Barnes & Collins 2008b (600 ms gaps, constant velocity 5-20 deg/s) too.
- Becker & Fuchs (abstract): the residual velocity is a first-order prediction attenuated by a
  variable gain, which is close to 1 while the target is visible.
- Barnes's expectation gain is not that: it gates both loops by the expectation that the target
  comes back, not by whether it is visible; near 1 when reappearance is likely, near 0 when not;
  0.9 in the simulations (Barnes & Collins 2008b, full text; Fukushima et al. 2013, text).
- Orban de Xivry et al. 2013 (model, partly read): the memory gain falls to 0.6 on a ramp and 0.7
  on a sinusoid (0.4 Hz) while the target is hidden; the decay to the residual has time
  constant 200 ms. Simulations, not human data.

### Hidden during repeating motion

No human number found yet for a half-cycle gap.

| Study | Motion | Finding | Read |
|---|---|---|---|
| Fukushima et al. 2002 | sinusoid 0.5 Hz (2 s period), hidden about half a cycle; **macaque** | eye gain 0.65 ± 0.16 SD of normal | abstract |
| Whittaker & Eaholtz 1982 | sinusoid 0.8 and 1.0 Hz; 2 human subjects | eye motion goes on after the target vanishes and follows the trajectory ("expected" in one abstract copy, "unexpected" in another; to check) | abstract |
| Kveraga et al. 2001 | predictable trajectories, humans | slow eye movements decelerate, **reverse** and accelerate again as predicted | abstract |

Kveraga and Whittaker bear on the clock: it goes on through a turnaround unseen, as eyes seem
to, and fades while it does.

Paywalled papers are not obtained. Free but not opened automatically (a browser can):
Barnes & Asselman 1991 (PMC1180117), Barnes et al. 1987 (PMC1192073), Ackerley & Barnes
(PMC3099020).

### A gap in a ramp, free source

Dowiasch et al. 2023 (eight people, 10 deg/s ramp, target always back after 300 ms): pursuit gain
0.96 without a gap, 0.89 with one, 200 ms into it; the drop starts ~112 ms after the target
vanishes and is lowest at ~227 ms (the last from a summary of the page, to check). Open raw data,
CC BY (Zenodo 8099040). Other ramp studies give 60% (Becker & Fuchs, unverified) and 29% after
600 ms (an fMRI study, abstract only): the level depends on speed and on expecting the target back.

## Target stops (S4)

- On a sinusoid stopped unpredictably after a few cycles, predictive pursuit goes on for
  150-200 ms, in patients and controls alike. Trillenberg et al. 1998, search snippet of the
  abstract. Nearest human number to S4.
- After an unexpected stop of a ramp: ~100 ms latency, then eye velocity decays to zero with a time
  constant of ~100 ms, no overshoot (Robinson et al. 1986; Pola & Wyatt 1997; via review).
  The response depends on whether the stop is expected.

## Free human numbers by situation

| # | Source | Number | Read |
|---|---|---|---|
| S1 | Barnes & Asselman 1991 | peak eye velocity builds over 2-4 cycles | abstract |
| S2 | Brostek et al. 2017 | 12 people, sinusoid 0.25 Hz, ±15°: gain 1.03 ± 0.09, "zero-latency tracking" | text, figures not |
| S3 | none on a repeating target; Dowiasch et al. 2023 on a ramp | see "Target hidden" | text in part |
| S3, trend | Whittaker & Eaholtz 1982, as cited by Brostek et al. 2017 | eyes keep oscillating at the old frequency after a sinusoid vanishes | via Brostek |
| S4 | Robinson et al. 1986; Pola & Wyatt 1997, as cited by Fukushima et al. 2013 | ramp: ~100 ms latency, ~100 ms decay | via review |
| S5, S6 | Barnes & Asselman 1991 | old timing kept after a period change; next movement follows the preceding sequence | abstract |
| S7 | Soechting et al. 2010 | 6 people, sums of sines from 0.22 Hz: gain 0.8-0.9 at the lowest frequency, ~0.5 at 3 Hz; lag grows to ~180° at 3.33 Hz | text, tables not |
| Delay | Brostek et al. 2017 | pursuit onset 141 ± 15 ms | text |
| S8 | none yet | | |

Sinusoid against Brostek et al. needs a 4 s period (0.25 Hz), so S2's adjust-on run is simulated.

## Steady motion and unpredictable motion (S2, S7)

- Over 0.2–2 Hz: gain is higher for sinusoids than for pseudo-random motion at low
  frequencies, equal at high ones; phase lag grows with frequency for sinusoids;
  pseudo-random motion shows a phase lead at low frequencies. Source of this study not yet
  identified (search abstract only).
- About 75% of the response during sinusoidal pursuit is predictive (perturbation study;
  source to identify).

## The clock account's evidence (D3)

- People tap in time with a **bouncing ball** about as well as with a metronome when its
  motion is realistic (Gan et al. 2015); with a less realistic ball, sound wins (Hove et
  al. 2013). The beat is the bounce, an abrupt reversal; a pendulum turns round smoothly,
  so this supports turnarounds as timing marks less directly. Abstracts.
- Deaf people synchronise to visual flashes better than hearing people (Iversen et al.
  2015). Abstract.
- Cells tuned to particular intervals in medial premotor cortex (Merchant et al. 2013) and
  interval-tuned beta activity in the putamen, preferring ~800 ms (Bartolo et al. 2014).
  Abstracts.
