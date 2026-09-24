# Bibliography

Every paper referenced anywhere in this project — to justify a design decision, a code change,
or a section of the paper — gets an entry here, with a note on *why* and *where*. Update this
file in the same change that leans on the paper.

Author lists and venues come from literature scans, not from the papers themselves; confirm
each before citing in the manuscript. Entries whose author list could not be checked say so.

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

## Repetitive motion in vision — the "how many cycles" literature

Note: `materials/01-literature/repetitive-motion-and-metrics.md` — the five fields that each
own one piece of this task, the metrics each reports, and the measured floor. This community
reads a whole clip and returns an integer count. Their two metrics, MAE of the count and OBO
(fraction within ±1), do not apply to a running prediction; the entries are here so related
work can say why, and so the framing is not mistaken for ours.

### Dwibedi et al., 2020 — "Counting Out Time: Class Agnostic Video Repetition Counting in the Wild"
- RepNet. *CVPR 2020*. Authors unverified (Dwibedi, Aytar, Tompson, Sermanet, Zisserman).
- **Why:** the reference method — temporal self-similarity matrix over frame embeddings,
  with heads for period and periodicity — and the Countix benchmark. What a reviewer
  means by "repetitive motion recognition".
- **Where:** paper related-work; the contrast that our task is prediction, not counting.

### Hu et al., 2022 — "TransRAC: Encoding Multi-Scale Temporal Correlation with Transformers for Repetitive Action Counting"
- *CVPR 2022*. RepCount dataset. Authors unverified.
- **Why:** the only repetition benchmark that annotates the start and end of each
  individual cycle, so it is the nearest thing to a per-cycle ground truth in that field.
- **Where:** paper related-work.

### "A Short Note on Evaluating RepNet for Temporal Repetition Counting in Videos", 2024
- arXiv:2411.08878.
- **Why:** the field's numbers do not reproduce across implementations and evaluation
  protocols; a reason not to quote counting accuracies as a comparison point.
- **Where:** paper related-work caveat.

### Cutler & Davis, 2000 — "Robust Real-Time Periodic Motion Detection, Analysis, and Applications"
- *IEEE TPAMI* 22(8):781–796.
- **Why:** the classical period estimator — track the object, build the self-similarity
  matrix over time, read the period from the peak of the average power spectral density.
  A learning-free comparator for the period alone, which is the half of our memory that
  can be scored without any model.
- **Where:** `baseline.py` (candidate); period-accuracy discussion.

### Perevalov et al., 2022 — "Frequency Cam: Imaging Periodic Signals in Real-Time"
- arXiv:2211.00198. Authors unverified.
- **Why:** period estimation straight from the event stream, per pixel, in real time —
  the event-native version of Cutler–Davis, and evidence that periodicity from events is
  cheap and solved. Sharpens what is new in ours: the *path*, not the period.
- **Where:** paper related-work.

## Prediction metrics — ADE / FDE and the naive floor

### Alahi et al., 2016 — "Social LSTM: Human Trajectory Prediction in Crowded Spaces"
- *CVPR 2016*.
- **Why:** where ADE (mean displacement error over the predicted horizon) and FDE (error
  at its end) became the standard pair. The naming `metrics.ade_fde` follows.
- **Where:** `metrics.py`; the results tables.

### Schöller, Aravantinos, Lay & Knoll, 2020 — "What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction"
- *IEEE Robotics and Automation Letters* 5(2):1696–1703.
- **Why:** the constant-velocity extrapolator beats most learned trajectory predictors on
  the standard benchmarks; the reason `baseline.Extrapolator` is reported in every table
  rather than assumed to be bad.
- **Where:** `baseline.py`; results tables.

### "Residual Kalman Dynamics for Event-Based UAV Forecasting", 2026
- arXiv:2609.00839. Authors unverified.
- **Why:** current event-camera forecasting work whose own baselines are the
  constant-velocity and constant-acceleration filters, with the constant-acceleration one
  slightly ahead — the same floor, on our kind of input.
- **Where:** `baseline.py` (`order=2`); paper related-work.

### Monaci et al., 2023 — "Fast Trajectory End-Point Prediction with Event Cameras for Reactive Robot Control"
- *CVPR Workshops 2023*, arXiv:2302.13796. Authors unverified.
- **Why:** event-native prediction of where a trajectory ends, on a real robot; an FDE-only
  point of comparison and a second data point on what event-based prediction reports.
- **Where:** paper related-work.

## Phase — the gait convention

### Kang, Molinaro, Choi, Camargo & Young, 2022 — "Continuous Locomotion Mode Recognition and Gait Phase Estimation"
- Accurate real-time phase estimation for normal and asymmetric gait. *IEEE ICORR 2022* /
  *IEEE TNSRE*. Authors unverified.
- **Why:** the convention for scoring a learned cycle's phase — RMSE as a percentage of
  the cycle, with published systems at 2.5–5 %. The units our clock's phase error should
  be reported in, and the one number that makes a half-period lock visible.
- **Where:** `metrics.py` (candidate phase metric); results tables.

## Deviation detection — how the field scores a detector

### Ahmad, Lavin, Purdy & Agha, 2017 — "Unsupervised real-time anomaly detection for streaming data"
- *Neurocomputing* 262:134–147. The Numenta Anomaly Benchmark (NAB); arXiv:1607.02480.
- **Why:** the published scoring rule closest to ours — an anomaly window per true event,
  only the first detection inside it counts, scored higher the earlier it lands, and false
  positives penalised by distance from a window. Our AUC + latency + false-alarms-per-minute
  triple is a hand-rolled version; NAB is what to cite for it. Its detector, HTM, is also
  our nearest cousin in kind: an online sequence memory that flags what it did not predict.
- **Where:** `metrics.py` (`deviation_roc`, `ratchet_threshold`); paper §IV.

### Tatbul, Lee, Zdonik, Alam & Gottschlich, 2018 — "Precision and Recall for Time Series"
- *NeurIPS 2018*. Range-based precision/recall.
- **Why:** precision and recall extended from points to intervals, with explicit terms for
  existence, size, position and cardinality of an overlap. The accepted answer to the fact
  that point-wise F1 is meaningless on segment anomalies.
- **Where:** `metrics.py` (candidate second detection metric).

### Kim, Choi, Yoon, Cho & Yoon, 2022 — "Towards a Rigorous Evaluation of Time-series Anomaly Detection"
- *AAAI 2022*.
- **Why:** shows the widely used point-adjust F1 is inflated to the point that random
  scores beat published detectors. The reason we report AUC and latency rather than an
  adjusted F1, stated in one line rather than assumed.
- **Where:** paper §IV methodology note.

### Paparrizos et al., 2022 — "Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection"
- *PVLDB* 15(11):2774–2787. VUS-ROC / VUS-PR.
- **Why:** threshold-free detection scoring with a tolerance buffer around each labelled
  event — robust to the boundary fuzziness our hand-labelled break times have.
- **Where:** paper §IV methodology note.

### Liu, Luo, Lian & Gao, 2018 — "Future Frame Prediction for Anomaly Detection — A New Baseline"
- *CVPR 2018*.
- **Why:** the video-anomaly field's statement of our deviation principle — anomaly is
  prediction error — with frame-level AUC on UCSD Ped2 / CUHK Avenue / ShanghaiTech
  (92.9 / 90.6 / 74.7 %) as the reference numbers. Ours is the same principle over a
  learned path rather than over pixels.
- **Where:** paper related-work; §IV framing.

### "Benchmark AUC Is Not Deployable Reliability: A Cross-Dataset Audit", 2026
- arXiv:2606.29506.
- **Why:** same-dataset frame AUC 0.704 falls to 0.499 across datasets; the argument for
  scoring held-out clips once and for reporting false alarms per minute alongside AUC.
- **Where:** paper §IV methodology note; the held-out protocol's justification.

### Mendes, Zhang, Peyrard & Berrada, 2021 — "Using Visual Anomaly Detection for Task Execution Monitoring"
- *IROS 2021*, arXiv:2107.14206. Authors unverified.
- **Why:** the application-side cousin — model the nominal motion of a repeated task, flag
  the deviation from it, from vision. Closest published problem statement to ours outside
  neuromorphic work, on frames and with a robot's own kinematics available.
- **Where:** paper related-work.

### Neto et al., 2024 — "Warped Time Series Anomaly Detection"
- arXiv:2404.12134. Authors unverified.
- **Why:** abnormal *cycles* in a repetitive task, with time warping between cycles — our
  deviation problem on proprioceptive rather than visual data.
- **Where:** paper related-work.

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
