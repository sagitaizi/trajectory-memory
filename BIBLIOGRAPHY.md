# Bibliography

Every paper referenced anywhere in this project — to justify a design decision, a code change,
or a section of the paper — gets an entry here, with a note on *why* and *where*. Update this
file in the same change that leans on the paper.

Author lists and venues from a 2026-09-09 literature scan; confirm each before citing in the
manuscript.

---

## Closest prior art

### Debat et al., 2021 — "Event-Based Trajectory Prediction Using Spiking Neural Networks"
- Debat, Chauhan, Cottereau, Masquelier, Paindavoine, Baures. *Frontiers in Computational
  Neuroscience* 15:658764.
- **Why:** the nearest existing system — event camera + 3-layer LIF SNN + unsupervised STDP →
  motion-selective neurons → polynomial readout predicts where a thrown ball lands. Static
  camera, constrained ballistic motion, offline batch training, no deviation detection, no
  attention.
- **Where:** `PLAN.md` novelty framing; the baseline every result is positioned against.

### "Egocentric Event-Based Vision for Ping Pong Ball Trajectory Prediction", 2025
- arXiv:2506.07860. Authors unverified (RPG, University of Zurich). Public code.
- **Why:** current-generation restatement of Debat — event camera → ball trajectory prediction
  with 3-D ground truth, real-time. Sharpens the required delta: repetitive learned path +
  deviation, not one-shot ballistic prediction.
- **Where:** `PLAN.md` novelty framing.

### N-DriverMotion, 2024
- "Driver motion learning and prediction using an event-based camera and directly trained
  spiking neural networks on Loihi 2." arXiv:2408.13379.
- **Why:** evidence the full stack (event camera → directly-trained SNN → motion prediction →
  neuromorphic hardware) is a current, publishable combination.
- **Where:** paper related-work; motivation for the event-native goal (Stage 2).

## Sequence memory and dynamical prediction in spiking networks

### Voelker, Kajić & Eliasmith, 2019 — "Legendre Memory Units"
- *NeurIPS 2019*.
- **Why:** NEF-derived, provably optimal rolling memory of a time signal; beats LSTM on chaotic
  time-series prediction; runs on Loihi. The memory core of `snn_lmu.LmuMemory`: the Legendre
  matrices, the delay readout, and the window-holds-the-cycle argument.
- **Where:** `lmu.py`; `PLAN.md` §C; the paper's method section.

### Voelker & Eliasmith, 2018 — "Improving spiking dynamical networks: accurate delays, higher-order synapses, and time cells"
- *Neural Computation* 30(3):569–609.
- **Why:** the spiking implementation of the delay network (the LMU's predecessor) with the NEF:
  how a population of LIF neurons with a first-order synapse realises the linear dynamics, and
  that its neurons show hippocampal time-cell tuning — the biological reading of the window
  memory.
- **Where:** `lmu.py` (`build_population`: NEF principle 3 with a lowpass synapse); the
  plausibility argument in `docs/superpowers/specs/2026-09-19-lmu-memory-design.md`.

### Righetti, Buchli & Ijspeert, 2006 — "Dynamic Hebbian learning in adaptive frequency oscillators"
- *Physica D* 216(2):269–281.
- **Why:** an oscillator whose frequency adapts to its input by a local rule (the input pulls
  the phase and the rate through F·sin φ) locks onto the rhythm of an arbitrary periodic
  signal without any search. The clock of the clock-and-map memory; its input coupling is
  computed by the clock population's neurons.
- **Where:** `phasemap.py` (`_Clock.step`), `snn_phasemap.py` (`clock_dynamics`); `PLAN.md` §C.

### Bekolay et al., 2014 — "Nengo: a Python tool for building large-scale functional brain models"
- *Frontiers in Neuroinformatics* 7:48.
- **Why:** the Neural Engineering Framework tooling that solves encoders, decoders and the
  recurrent weights for a population that computes a given dynamics; used at construction
  time for the LMU population and the clock populations, which torch then simulates. Also the
  home of the PES rule (error-driven decoder learning) the map uses.
- **Where:** `lmu.py` (`build_population`, `build_dynamics`), `snn_phasemap.py`.

### MacDonald, Lepage, Eden & Eichenbaum, 2011 — "Hippocampal 'time cells' bridge the gap in memory for discontiguous events"
- *Neuron* 71(4):737–749.
- **Why:** neurons firing in sequence at successive lags after an event — the brain's version
  of a delay line, which the LMU population reproduces.
- **Where:** plausibility of the hand-set window memory.

### Sussillo & Abbott, 2009 — "Generating Coherent Patterns of Activity from Chaotic Neural Networks"
- *Neuron* 63(4):544–557. FORCE learning.
- **Why:** foundational method for training a fixed recurrent network to generate/predict
  arbitrary periodic patterns; basis for the reservoir approach with an online-trained readout.
- **Where:** `model.py` (`ReservoirMemory`); `PLAN.md` G-F.

### Nicola & Clopath, 2017 — "Supervised learning in spiking neural networks with FORCE training"
- *Nature Communications* 8:2208.
- **Why:** FORCE carried into spiking networks — the spiking reservoir route for pattern
  learning and prediction.
- **Where:** `model.py`; G-F.

### Bellec et al., 2020 — "A solution to the learning dilemma for recurrent networks of spiking neurons"
- Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass. *Nature Communications* 11:3625.
  e-prop.
- **Why:** online, local approximation to BPTT for recurrent SNNs; the option if genuine
  online weight updates are wanted instead of pretrain-then-freeze.
- **Where:** G-F; noted in `PLAN.md` out-of-scope (online re-learning) as the tool that path
  would need.

### Gilra & Gerstner, 2017 — "Predicting non-linear dynamics by stable local learning in a recurrent spiking neural network"
- *eLife* 6:e28295.
- **Why:** local-plasticity learning of arbitrary dynamical systems in an RSNN — directly
  relevant to representing a trajectory as network dynamics.
- **Where:** G-F background.

### Kim & Chow, 2018 — "Learning recurrent dynamics in spiking networks"
- *eLife* 7:e37124.
- **Why:** RLS-trained recurrent spiking networks that reproduce target dynamics stably;
  method reference for the reservoir readout.
- **Where:** `model.py`.

## Multiple timescales — fast detail, slow structure

Note: `materials/01-literature/multi-timescale-memory.md`. All entries there back the choice
between one mixed-τ recurrent layer and a fast→slow layer split for the memory core (Phase C).

### Yamashita & Tani, 2008 — "Emergence of Functional Hierarchy in a Multiple Timescale Neural Network Model: A Humanoid Robot Experiment"
- *PLoS Computational Biology* 4(11):e1000220.
- **Why:** the MTRNN — input → fast recurrent layer (τ = 5) → slow recurrent layer (τ = 70),
  trained by BPTT. Fast units learn movement primitives, slow units their sequence; equal
  time constants perform significantly worse. The reference for a fast→slow split.
- **Where:** memory-core architecture decision, Phase C.

### Perez-Nieves, Leung, Dragotti & Goodman, 2021 — "Neural heterogeneity promotes robust learning"
- *Nature Communications* 12:5791.
- **Why:** recurrent SNNs with gamma-distributed, learnable membrane time constants beat
  homogeneous ones on temporal tasks and are more robust. Justifies mixed τ within a layer.
- **Where:** memory-core initialisation, Phase C.

### Zheng et al., 2024 — "Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics"
- *Nature Communications* 15:277.
- **Why:** several learnable timing factors per neuron (dendritic branches) for
  multi-timescale tasks; the stronger form of within-layer heterogeneity.
- **Where:** memory-core alternative, Phase C.

### Bellec, Salaj, Subramoney, Legenstein & Maass, 2018 — "Long short-term memory and learning-to-learn in networks of spiking neurons"
- *NeurIPS* 31.
- **Why:** LSNN — an adaptive threshold decaying over seconds gives LIF networks long memory
  without long membrane constants. The slow layer's neuron model (`snn.AdaptiveLeaky`).
- **Where:** memory core, Phase C (`PLAN.md` §C design; `docs/snn-experiments-log.md` Run 4).

### Yin, Corradi & Bohté, 2021 — "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
- *Nature Machine Intelligence* 3:905–913.
- **Why:** LSNN with learnable membrane *and* adaptation time constants per neuron; the
  snnTorch-style recipe for multi-timescale recurrent SNNs.
- **Where:** memory-core alternative, Phase C.

### Hasson, Yang, Vallines, Heeger & Rubin, 2008 — "A hierarchy of temporal receptive windows in human cortex"
- *Journal of Neuroscience* 28(10):2539–2550.
- **Why:** cortex integrates over progressively longer windows from sensory to higher areas.
  Biological grounding for fast-feeds-slow.
- **Where:** discussion / architecture rationale.

### Murray et al., 2014 — "A hierarchy of intrinsic timescales across primate cortex"
- *Nature Neuroscience* 17:1661–1663.
- **Why:** single-neuron intrinsic timescales (autocorrelation decay) rise along the cortical
  hierarchy, ~50 ms sensory to ~300 ms prefrontal. The numbers behind the fast/slow ratio.
- **Where:** discussion / architecture rationale.

### Kiebel, Daunizeau & Friston, 2008 — "A hierarchy of time-scales and the brain"
- *PLoS Computational Biology* 4(11):e1000209.
- **Why:** the computational argument: slow levels set the parameters of fast levels'
  predictions. Our path head (slow) and horizon heads (fast) in one sentence.
- **Where:** discussion / architecture rationale.

### Barnes & Asselman, 1991 — "The mechanism of prediction in human smooth pursuit eye movements"
- *Journal of Physiology* 439:439–461.
- **Why:** humans tracking a periodic target lock on within a few cycles and then anticipate
  it from a stored velocity/timing memory. The behavioural version of our task.
- **Where:** introduction / motivation.

### Kettner et al., 1997 — "Prediction of complex two-dimensional trajectories by a cerebellar model of smooth pursuit eye movement"
- *Journal of Neurophysiology* 77(4):2115–2130.
- **Why:** cerebellar model with delay-line inputs and delayed-error learning predicts
  sum-of-sines and circular targets like monkeys do, and after a perturbation keeps following
  the learned path for ~80 ms — the prediction-error signature of a break.
- **Where:** architecture rationale; deviation-signal framing.

### Cerminara, Apps & Marple-Horvat, 2009 — "An internal model of a moving visual target in the lateral cerebellum"
- *Journal of Physiology* 587(2):429–442.
- **Why:** Purkinje cells encode the target's motion as an internal model, not the visual
  input as such. Direct evidence for a neural trajectory memory.
- **Where:** introduction / motivation.

## Deviation / novelty detection

### Schulz et al., 2021 — "The generation of cortical novelty responses through inhibitory plasticity"
- Schulz, Miehl, Berry, Gjorgjieva. *eLife* 10:e65309.
- **Why:** biological mechanism for "familiar → suppressed, novel → salient" via inhibitory
  plasticity — grounding for the deviation signal and for the out-of-scope attention work.
- **Where:** Phase D framing; paper discussion.

### "Vacuum Spiker: A Spiking Neural Network-Based Model for Efficient Anomaly Detection in Time Series", 2025
- arXiv:2510.06910.
- **Why:** recent, concrete SNN time-series anomaly detection — comparison point for the
  deviation-detection method.
- **Where:** Phase D; paper related-work.

## Attention — future-work context

### Itti & Koch, 2001 — "Computational modelling of visual attention"
- *Nature Reviews Neuroscience* 2(3):194–203.
- **Why:** canonical saliency + winner-take-all + inhibition-of-return model; "habituate to
  repetitive motion" is a temporal generalisation of IOR.
- **Where:** paper future-work section.

### "Look twice: A generalist computational model predicts return fixations across tasks and species", 2022
- *PLOS Computational Biology*.
- **Why:** model of gaze return/switching; benchmark target if the eye-tracking comparison is
  ever run.
- **Where:** paper future-work section.

## Non-neuromorphic baselines and surveys

### Ijspeert, Nakanishi & Schaal, 2002 — rhythmic dynamic movement primitives
- "Learning rhythmic movements by demonstration using nonlinear oscillators." *IROS 2002*.
- **Why:** the classical way to learn and reproduce a repetitive trajectory from demonstration
  — a baseline to compare against.
- **Where:** `baseline.py`; results table.

### Saveriano et al., 2023 — "Dynamic movement primitives in robotics: A tutorial survey"
- Saveriano, Abu-Dakka, Kramberger, Peternel. *International Journal of Robotics Research*.
- **Why:** survey grounding for the DMP baseline and the periodic-DMP formulation.
- **Where:** `baseline.py`; paper related-work.

### "A Survey on Deep Learning Models for Anomaly Trajectory Detection", 2025
- **Why:** situates the deviation-detection contribution against the (mostly GPS-scale,
  non-real-time, non-event) trajectory-anomaly literature.
- **Where:** paper related-work.

## Event-camera simulation

### Rebecq, Gehrig & Scaramuzza, 2018 — "ESIM: an Open Event Camera Simulator"
- *CoRL 2018*.
- **Why:** renders a 3-D scene to events with configurable intrinsics/distortion/threshold —
  fallback simulator if a textured 3-D scene is needed.
- **Where:** `simulate.py`.

### Hu, Liu & Delbruck, 2021 — "v2e: From Video Frames to Realistic DVS Events"
- *CVPR Workshops 2021*.
- **Why:** video → events with realistic non-idealities (threshold jitter, refractory period,
  shot noise, photoreceptor bandwidth), transfer function fittable to a real sensor. Primary
  simulator.
- **Where:** `simulate.py`.

### "Event Camera Simulator Improvements via Characterized Parameters", 2021
- *Frontiers in Neuroscience*.
- **Why:** procedure for matching simulator parameters to a measured real camera — the recipe
  for configuring v2e to the DVXplorer.
- **Where:** `simulate.py`; Phase A sim-vs-real check.

## Event datasets — external generalisation check only

### Mitrokhin et al., 2019 — "EV-IMO: Motion Segmentation Dataset and Learning Pipeline for Event Cameras"
- *IROS 2019*.
- **Why:** indoor, multiple fast-moving objects, Vicon trajectory ground truth — candidate
  external test for generalisation beyond the rig.
- **Where:** Phase F external check.

### "Event Stream-based Visual Object Tracking: A High-Resolution Benchmark" (EventVOT), 2024
- *CVPR 2024*. Authors unverified (Xiao Wang et al.).
- **Why:** 1280×720 single-object tracking benchmark — resolution closest to the DVXplorer;
  external generalisation test.
- **Where:** Phase F external check.

### "Object Tracking by Jointly Exploiting Frame and Event Domain" (FE108), 2021
- *ICCV 2021*. Authors unverified.
- **Why:** single-object event tracking benchmark (DAVIS346); noted as a lower-resolution
  alternative external test.
- **Where:** Phase F external check (secondary).
