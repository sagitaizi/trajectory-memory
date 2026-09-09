# Non-neuromorphic approaches to the same problem

These are the comparison points. For **clean periodic motion they will likely beat an SNN on
raw prediction accuracy** — which is why the SNN's contribution has to be event-native
operation, the dynamics-as-solver framing, online acquisition, and deployment substrate, not
"better numbers". Say this explicitly in the paper.

## Rhythmic / periodic Dynamic Movement Primitives (DMP)
Ijspeert, Nakanishi & Schaal 2002; survey Saveriano, Abu-Dakka, Kramberger, Peternel,
*IJRR* 2023.
- A **phase oscillator** `φ̇ = Ω` drives a nonlinear **forcing term** `f(φ)` (weighted sum of
  von Mises basis functions) added to a stable linear system that pulls toward a baseline.
- Learn the weights by **locally weighted regression** from a single demonstrated period.
- Reproduce the path; modulate frequency `Ω` and amplitude online.
- Anomaly = residual between the observed motion and the DMP rollout.
- Equations in `02-methods/rhythmic-dmp.md`. This is the strongest classical baseline for
  "learn and reproduce a repetitive trajectory".

## Kalman filter with a periodic model
- State = position + velocity (constant-velocity), optionally augmented with harmonic
  coefficients for a known/estimated period `ω`; or estimate `ω` itself with an EKF, or run
  an **IMM** bank over candidate periods.
- Prediction horizon `h`: iterate the transition model.
- Deviation = **normalised innovation squared** / Mahalanobis distance of the residual vs a
  chi-square threshold.
- Equations in `02-methods/kalman-periodic.md`.

## Harmonic / Fourier regression
Fit `x(t) ≈ a0 + Σ_k [a_k cos(kωt) + b_k sin(kωt)]` over a sliding window (least squares),
predict forward analytically. Trivial, fast, hard to beat on a clean sinusoid-like path.

## Deep-learning trajectory prediction (context, not baselines you must run)
Social-LSTM, Trajectron++, trajectory transformers. Surveys: Rudenko et al. 2020 "Human
Motion Trajectory Prediction: A Survey" (*IJRR*); "MobilityDL: a review of deep learning from
trajectory data" 2024; "Deep Learning for Vision-based Prediction: A Survey" 2020.

## Trajectory anomaly detection (context)
"A Survey on Deep Learning Models for Anomaly Trajectory Detection" 2025; GeoTrackNet
(RNN + probabilistic, maritime); seq2seq anomaly detection in human trajectories (arXiv
1907.05813); VAE-based abnormality scoring. Mostly GPS / surveillance scale, offline,
non-event — situate your real-time event-based method against them.
