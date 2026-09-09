# Python environment

Shared conda env **`thesis`**, Python 3.14. `conda activate thesis`.

## Present (from the main thesis repo)
`dv_processing`, `dynamixel_sdk`, `opencv-python`, `numpy`, `scipy`, `pyyaml`, `nengo`,
`nengo-gui`, `pytest`.

## Add for this project (`requirements.txt`)
`snntorch`, `v2e`. (`pip install -r requirements.txt` while online, before travelling.)

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
CUDA available on this machine (per the main repo's notes). snnTorch/torch training can use
it; the reservoir/LMU baselines are CPU-fine.

## Sanity check before travel
```
conda activate thesis
python -c "import numpy, scipy, cv2, yaml, nengo, snntorch, torch; print('ok', torch.cuda.is_available())"
python -c "import v2e; print('v2e ok')"       # or: v2e --help
python -m pytest -q                            # once tests exist
```
