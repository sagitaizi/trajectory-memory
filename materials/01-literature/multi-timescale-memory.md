# Multiple timescales: fast detail, slow structure

The question: the memory has two jobs — the next 25–200 ms (detail) and the cycle's period and
shape (structure). Should one recurrent layer with mixed time constants do both, or should
the network be split into a fast layer feeding a slow layer? What the brain and the
literature say.

## In the brain

### A hierarchy of timescales across cortex
- **Hasson et al. 2008** (*J Neurosci*): human cortex shows a gradient of "temporal receptive
  windows" — early sensory areas respond to the last few hundred ms, higher areas integrate
  over seconds to minutes. Measured by scrambling movies at different time scales.
- **Murray et al. 2014** (*Nat Neurosci*): the same gradient in single neurons of the macaque,
  measured from the decay of spike-count autocorrelation at rest — ~50–100 ms in sensory
  areas rising to ~300+ ms in prefrontal. The timescale is *intrinsic* (it is there without a
  task), and it follows the anatomical hierarchy: fast areas feed slow areas.
- **Kiebel, Daunizeau & Friston 2008** (*PLoS Comput Biol*): the computational reading — the
  world has structure at nested timescales, so a brain that models it needs a hierarchy where
  each level's slow states set the parameters of the level below. Prediction at a fast level
  is conditioned on the state of the slow level.

The pattern: **sensory in at the fast end, structure at the slow end, fast feeds slow, slow
modulates fast.** Not a single pool with a mix of leaks.

### Predicting a periodic target: smooth pursuit and the cerebellum
- **Barnes & Asselman 1991** (*J Physiol*): humans tracking a periodic target lock on within a
  few cycles and then lead it — the eye is driven by a stored copy of the target's velocity
  and timing, released in anticipation. Pursuit of predictable motion runs on an internal
  model, not on retinal error.
- **Kettner et al. 1997** (*J Neurophysiol*): a cerebellar model of predictive pursuit —
  an array of inputs at a range of delays (a tapped delay line onto the granule layer) and a
  Purkinje-cell learning rule driven by retinal-slip error arriving ~100 ms late. Tracks
  sum-of-sines targets with the same phase behaviour as monkeys, and after a perturbation
  *keeps following the learned path for ~80 ms* before correcting — exactly the "prediction
  error against the learned path" signature we use for deviation.
- **Cerminara, Apps & Marple-Horvat 2009** (*J Physiol*): direct evidence — Purkinje cells in
  the lateral cerebellum encode the moving target's motion, and their activity reflects an
  internal model of the target rather than the visual input as such.

For us: the biology of "learn a periodic path, predict ahead, notice a break" is the
cerebellum with a delay-line input and a slow-error learning rule — with the higher-level
structure (which pattern, what period) sitting upstream in cortex.

## In artificial and spiking networks

### Separate fast and slow layers
- **Yamashita & Tani 2008** (*PLoS Comput Biol*) — the **MTRNN**. Three layers: input/output,
  fast context (τ = 5 steps), slow context (τ = 70 steps). The input connects to the fast layer
  only; the slow layer sees the world through the fast one; both are recurrent and fully
  connected to each other. Trained by BPTT on robot movement sequences. Result: a functional
  hierarchy self-organised — fast units learned reusable movement primitives, slow units
  learned the *sequence* of primitives. Control: with equal time constants (τ ratio 1),
  performance was significantly worse, especially on recombining primitives. This is the
  closest published match to the split you propose, and it is the standard reference for it.
- **Koutník et al. 2014** (Clockwork RNN), **Chung, Ahn & Bengio 2016** (HM-RNN): the same
  idea in non-spiking RNNs — modules updated at different rates, slow modules capture long
  structure. Confirms the effect is general, not specific to Tani's setup.

### Mixed timescales inside one layer
- **Perez-Nieves et al. 2021** (*Nat Commun*): recurrent SNNs whose membrane time constants
  are drawn from a broad (gamma) distribution and made learnable beat homogeneous networks on
  every task with temporal structure, and are more robust to hyperparameters. Training pushes
  the distribution toward the one found in real neurons. Cheap to do (one vector of τ).
- **Zheng et al. 2024** (*Nat Commun*): DH-SNN — each neuron carries several dendritic
  branches with their own learnable timing factors, so one neuron integrates at several
  timescales; better on multi-timescale tasks than heterogeneity across neurons alone.
- **Bellec et al. 2018** (NeurIPS), **Yin, Corradi & Bohté 2021** (*Nat Mach Intell*): a
  *second* slow variable per neuron — an adaptive threshold that decays over seconds — gives
  a spiking network long memory (LSNN / adaptive SRNN), matching LSTM on sequence tasks.
  The slow timescale lives in adaptation, not the membrane. Yin makes both time constants
  learnable per neuron.

## What this says for our design

Both routes work; the literature does not force one. The trade-off:

| | One layer, mixed τ | Fast layer → slow layer |
|---|---|---|
| Support | Perez-Nieves 2021; simplest | MTRNN; cortical hierarchy |
| What training can find | any mixing; no structure imposed | fast = local motion, slow = cycle/shape, by construction |
| Readouts | both heads from the same pool | horizon heads from fast, path head from slow (natural fit) |
| Cost | one τ vector | two layers, two recurrent matrices, an extra hyper-parameter (layer sizes, τ ranges) |
| Risk | heads compete for the same neurons | slow layer starves if the fast layer's output is poor |

The split matches the two jobs and the two readouts one-to-one, and it makes the paper's
story cleaner ("the slow layer holds the cycle"). The single mixed layer is the ablation.
MTRNN's ratio of ~14 between fast and slow τ is the reference starting point.
