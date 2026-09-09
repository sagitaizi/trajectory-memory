# Deviation / novelty detection

The paper's deviation signal = "the motion stopped matching the learned path". Cheapest
correct form: **prediction error against the learned model**, thresholded.

## Prediction-error-as-novelty (the mechanism you'll use)
- Run the memory network in generative / prediction mode against the incoming position.
- `e(t) = || predicted(t) - observed(t) ||`.
- On-pattern → `e` low and stationary. Pattern breaks → `e` jumps.
- `deviation_score(t)` = a smoothed / normalised `e(t)` (e.g. z-scored against the running
  distribution, or a CUSUM on `e`).
- Report: ROC/AUC over a threshold sweep, median detection latency `t_flag − t_break`,
  false-positive rate on no-deviation clips, and whether the score relaxes after a transient.

## Biological grounding (for the discussion / future-work framing)
- **Habituation via short-term synaptic depression** — repeated input depresses synapses,
  so a familiar pattern drives weaker responses; a change restores them. "Neural habituation
  enhances novelty detection" (biorxiv 2019).
- **Inhibitory-plasticity novelty responses** — Schulz, Miehl, Berry, Gjorgjieva, *eLife*
  2021: STDP on inhibitory→excitatory synapses raises inhibition onto neurons tuned to
  familiar stimuli, leaving novel-stimulus responses intact. A spiking mechanism for
  "familiar becomes quiet, novel stays loud".
- **Mismatch negativity (MMN)** models — spiking integrate-and-fire networks with depressing
  synapses detect multiple kinds of auditory novelty, including stimulus omission.

## Engineered SNN anomaly detection (comparison points)
- "Vacuum Spiker" (arXiv:2510.06910, 2025) — SNN for efficient time-series anomaly detection,
  one-class training.
- "A Deep Spiking Neural Network Anomaly Detection Method" (Comp. Intell. Neurosci. 2022) —
  spike-sequence modelling for anomaly discovery. *(Note: this article was later retracted —
  cite with care or not at all.)*
- SNN autoencoders for anomaly detection (various).

These are generic time-series AD, not trajectory-specific — position your method as "novelty
detection specialised to a learned spatial motion pattern".
