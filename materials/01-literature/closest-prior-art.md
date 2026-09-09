# Closest prior art

## Debat et al. 2021 — "Event-Based Trajectory Prediction Using Spiking Neural Networks"
*Frontiers in Computational Neuroscience* 15:658764. Authors: Debat, Chauhan, Cottereau,
Masquelier, Paindavoine, Baures (verify).

**Task.** Predict where a thrown ball will land from a partial view of its flight.

**Pipeline.**
- Event camera (NeuroSoc, 128×120), **stationary**, perpendicular to the ball plane,
  controlled indoor lighting.
- 3-layer feedforward **LIF** SNN, 60 / 80 / 100 filters, 5×5×d convolutional patches, stride
  1, lateral inhibition at matched retinotopic positions, homeostatic threshold adaptation,
  no pooling.
- **Unsupervised STDP** trains the filters — neurons become selective to motion direction and
  (more broadly) speed.
- Readout: **second-degree polynomial regression** per last-layer filter, mapping neuron
  position → predicted landing point; predictions combined weighted by regression reliability.
- Trained **offline** on 208 of 297 recorded trajectories; 30% held out. Vicon ground truth.

**Results.** Mean absolute error 7.7 px at 15% trajectory visibility, 2.2 px at 90%.
Significantly outperformed 12 human participants. Learning curve plateaus ~80% of the data.

**What it does NOT do (your deltas):**
- Static camera; no ego-motion.
- Constrained, near-identical ballistic trajectories with minimal background motion —
  authors call it "a rather easy task".
- Offline batch training only — no online / streaming acquisition.
- No deviation / novelty detection.
- No attention.
- Prediction is a spatial regression off motion-selective filters, not a learned dynamical
  model of a *repeated* path.

## "Egocentric Event-Based Vision for Ping Pong Ball Trajectory Prediction" 2025
arXiv:2506.07860. RPG, University of Zurich (verify authors). **Public code + data.**

Current-generation restatement of Debat: event camera + 3-D ground-truth ball trajectories,
real-time prediction, egocentric viewpoint (Project Aria + event stream). Still ballistic,
still one-shot (no learned repeated pattern), no deviation detection. Cite alongside Debat as
"the problem has moved on since 2021, and the gap is still repetition + break detection".

## N-DriverMotion 2024
arXiv:2408.13379. Event camera + **directly-trained SNN on Intel Loihi 2**, learning and
predicting in-car driver motion. Different domain, but concrete evidence that
event-camera → directly-trained SNN → motion prediction → neuromorphic hardware is a current,
publishable stack. Useful for the related-work paragraph on neuromorphic deployment (which
you can only cite, not measure).

## The gap, in one sentence
No published system does event-driven, spiking, *online* memory of a **freely-moving real
target's repetitive trajectory**, with a **deviation signal**, framed as **predictive
inference**.
