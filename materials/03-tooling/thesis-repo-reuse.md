# Reusing the main thesis repo

Imported, not copied. `_thesis_path.py` prepends `D:\Projects\Thesis` (override with
`THESIS_REPO`) to `sys.path`; `conftest.py` imports it so pytest picks it up. In a script:
`import _thesis_path` before any `recording` / `camera` / `pipeline` import.

The main repo's `config` resolves its calibration file and `params.yaml` relative to its own
location, so imports work with only `sys.path` set — no install.

## Modules and what they give you
Signatures below are from memory (2026-09-09) — check with `help()` / the source when working.

| Module | Use |
|---|---|
| `recording.player` | Deterministic `.aedat4` replay. Yields events (structured arrays: `x, y, t, polarity`) and, if present, IMU + encoder streams. The drop-in replacement for live capture. |
| `camera.calibration` | `CameraGeometry`; `Intrinsics`; `undistort_normalized` / `distort_normalized`; `px_per_rad`. Load with `dv.camera.CalibrationSet.LoadFromFile(...)` — never parse XML. |
| `pipeline.accumulator` | Events → accumulated frame (B.1). Stage-1 frontend. |
| `pipeline.time_surface` | Events → time surface (B.2). Alternative Stage-1 frontend. |
| `config.hardware` | `PROJECT_ROOT`, camera/motor constants (`640×480`, tick ranges, etc.). |
| `config.runtime` | `RuntimeConfig` dataclass + `load_config(path=...)` — takes an explicit path, so you can point it at the Thesis `params.yaml`. |
| `camera.tuning` | DVXplorer contrast-threshold constants — feed these to v2e. |

## Calibration file (for the camera model in v2e and any px↔rad work)
`calibration/3 - 22_08/calibration_camera_DVXplorer_DXA00126-refit-k1k2-2026_08_23.json`
— radial-tangential, `k1 ≈ −0.37`. **Not** the `.xml` beside it, **not** folder `4 - 22_08`.

## Gotcha carried over
`CameraGeometry.backProjectSequence` / `projectSequence` **silently ignore distortion** —
they're a bare pinhole pair. Any back-project/project must go through `camera.calibration`
(or `pipeline.iwe`'s `_backproject_lut` / `project_rays`) and apply distortion on **both**
sides or neither.
