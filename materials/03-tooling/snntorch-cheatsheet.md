# snnTorch cheatsheet

Surrogate-gradient SNNs on top of PyTorch. Candidate for the Stage-1/2 localiser and,
possibly, the memory core. Verify names against the installed version (`import snntorch;
snntorch.__version__`).

Ref: Eshraghian et al., "Training Spiking Neural Networks Using Lessons from Deep Learning",
Proc. IEEE 2023.

## Core objects
```python
import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate

beta = 0.9                                  # membrane decay per step
spike_grad = surrogate.fast_sigmoid(slope=25)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(n_in, n_hid)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2  = nn.Linear(n_hid, n_out)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x_seq):               # x_seq: [T, B, n_in]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        rec_spk, rec_mem = [], []
        for t in range(x_seq.shape[0]):
            cur1 = self.fc1(x_seq[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            rec_spk.append(spk2); rec_mem.append(mem2)
        return torch.stack(rec_spk), torch.stack(rec_mem)
```

## Neuron models
`snn.Leaky` (LIF), `snn.Synaptic` (current-based), `snn.Lapicque`, `snn.RLeaky` /
`snn.RSynaptic` (recurrent — for the memory core), `snn.Alpha`. Recurrent variants take
`V` / recurrent weight args.

## Training
- Input is a **time-major tensor** `[T, B, ...]` — bin your events first (voxel grid / spike
  tensor, see `02-methods/event-representations.md`).
- Loss: `snntorch.functional` has `ce_rate_loss`, `mse_count_loss`, `mse_membrane_loss`. For
  **regression** (position, trajectory), read the membrane of a non-spiking final layer
  (`output=True`) and use plain `nn.MSELoss` on it.
- BPTT via autograd over the unrolled loop. `torch.optim.Adam`. Detach hidden states between
  sequences.
- Convolutional: wrap `nn.Conv2d` + `snn.Leaky` the same way for the frame localiser.

## Limitation to remember
BPTT is an **offline** paradigm. snnTorch has no built-in online continual learning — if
Stage 1 needs live weight updates, that's e-prop / a custom local rule, not snnTorch out of
the box. Pretrain-then-freeze is the intended fit.
