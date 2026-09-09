# Python environment

Shared conda env **`thesis`**, Python 3.14. `conda activate thesis`.

## Present (from the main thesis repo)
`dv_processing`, `dynamixel_sdk`, `opencv-python`, `numpy` (2.4.6), `pyyaml`, `nengo`,
`nengo-gui`, `pytest`, `tqdm`.

## Add for this project (`requirements.txt`)
See `requirements.txt` for pinned versions. `pip install -r requirements.txt` while online.

## Install outcome — 2026-09-10 (last online night)
- **Installed clean** (cp314 wheels, numpy stayed 2.4.6): `scipy` 1.18.1, `matplotlib` 3.11.1,
  `torch` 2.14.0 **(+cpu — no CUDA wheel for cp314)**, `torchvision` 0.29.0, `snntorch` 1.0.0.
- **`scipy` and `matplotlib` were NOT in the env** despite the line above once claiming so.
- **`v2e` is not on PyPI.** Source clone at `D:\Projects\v2e` (`SensorsINI/v2e` v1.5.1), used
  via `sys.path` like the Thesis repo — never `pip install` its `requirements.txt` (pins
  `numpy<2`, pulls Jupyter/Gooey/wxPython). Lean runtime deps installed instead: `h5py`,
  `numba` 0.67.0 + `llvmlite` 0.49.0, `engineering-notation`, `colored`, `pandas`,
  `screeninfo`, `easygui`.
- **v2e works via the Python API**, not the CLI: `from v2ecore.emulator import EventEmulator`,
  construct with `device="cpu"` (default is `cuda`, which the cpu torch rejects),
  `emu.generate_events(frame_f32, t_seconds)` → `(N, 4)` array of `[t, x, y, polarity]`.
  Smoke-tested 2026-09-10.
- **CPU-only torch.** SNN training (Stage-1 localiser, Stage-2) runs on CPU — fine at small
  scale, slow for anything large. A CUDA build would need re-installing while online.

## Not available — do not plan around these
| Package | Why |
|---|---|
| `nengo-dl` | build fails on Python 3.14 (needs old TF; declared for Nengo ≤3.2) |
| `nengo-loihi` | last release 2022; untested on Nengo 4.x |
| `lava-nc` | Intel archived the org 2026-05-13; also needs Python <3.11, numpy <2 |
| `tonic` | pins numpy<2 — conflicts with this env; replicate its transforms yourself |

Consequence: no trained conv-SNN inside Nengo, no Loihi/Lava deployment. Trained SNNs →
`snntorch` (co-installs fine, native 3.14 torch wheel). Energy/latency claims → **cited**.

## GPU
The installed `torch` 2.14.0 is **CPU-only** (`torch.cuda.is_available()` → False; no cp314
CUDA wheel at install time). snnTorch/torch training runs on CPU; the reservoir/LMU baselines
are CPU-fine anyway.

## Sanity check
```
python -c "import numpy, scipy, cv2, yaml, nengo, snntorch, torch, matplotlib; print('ok, cuda', torch.cuda.is_available())"
python -c "import sys; sys.path.insert(0, r'D:\Projects\v2e'); from v2ecore.emulator import EventEmulator; print('v2e ok')"
python -m pytest -q
```
