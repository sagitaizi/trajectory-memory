import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import csv

import numpy as np
import yaml

from scripts.make_sim_dataset import main, random_sim_cfg, random_spec
from trajmem.data import load_clip
from trajmem.trajectories import sample

PATH_CFG = {
    "shapes": ["circle", "ellipse", "sweep", "figure8", "lissajous"],
    "period_s": [0.8, 4.0],
    "semi_axis": [0.05, 0.35],
    "margin": 0.03,
    "deviation_fraction": 0.5,
    "wobble": {
        "perfect_fraction": 0.3,
        "amp_depth": [0.0, 0.08], "amp_period_s": [3.0, 12.0],
        "drift": [0.0, 0.02], "drift_period_s": [4.0, 15.0],
        "phase_depth": [0.0, 0.15], "phase_period_s": [3.0, 10.0],
        "growth": [-0.02, 0.03],
    },
}
SIM_CFG = {
    "resolution": [64, 48], "fps": 200, "duration_s": 0.2,
    "blob": {"radius_px": 4, "bg_intensity": 30, "fg_intensity": 170},
    "calibration": None,
    "v2e": {"pos_thres": 0.2, "neg_thres": 0.2, "sigma_thres": 0.03, "cutoff_hz": 0,
            "leak_rate_hz": 0.0, "shot_noise_rate_hz": 0.0, "refractory_period_s": 0.0},
    "randomise": {
        "pos_thres": [0.15, 0.30], "sigma_thres": [0.01, 0.06],
        "shot_noise_rate_hz": [0.0, 0.1], "fg_intensity": [120, 220], "radius_px": [2, 5],
        "aspect": [1.0, 4.0],
        "string_fraction": 0.5, "string_intensity": [60, 150], "string_thickness_px": [1, 3],
        "texture_depth": [0.0, 0.8], "texture_scale_px": [2.0, 8.0],
        "lag_fraction": 0.5, "cutoff_hz": [60, 200],
        "path": PATH_CFG,
    },
}


def test_random_spec_is_deterministic_under_a_seed():
    a = random_spec(np.random.default_rng(3), PATH_CFG, duration_s=10.0)
    b = random_spec(np.random.default_rng(3), PATH_CFG, duration_s=10.0)
    assert a == b


def test_random_spec_stays_inside_the_frame_for_the_whole_clip():
    t = np.linspace(0, 10.0, 2000)
    for i in range(40):
        spec = random_spec(np.random.default_rng(i), PATH_CFG, duration_s=10.0)
        pos = sample(spec, t)
        assert pos.min() >= PATH_CFG["margin"] - 1e-9 and pos.max() <= 1 - PATH_CFG["margin"] + 1e-9


def test_random_spec_covers_every_shape_and_puts_deviations_in_the_middle_third():
    shapes, kinds = set(), set()
    for i in range(60):
        spec = random_spec(np.random.default_rng(i), PATH_CFG, duration_s=12.0)
        shapes.add("sweep" if spec.size[1] == 0 else spec.shape)
        assert PATH_CFG["period_s"][0] <= spec.period_s <= PATH_CFG["period_s"][1]
        for d in spec.deviations:
            kinds.add(d.kind)
            assert 4.0 <= d.at_t <= 8.0
        assert len(spec.deviations) <= 1
    assert shapes == set(PATH_CFG["shapes"])
    assert kinds == {"shrink", "speed_change", "drift", "switch_shape"}


def test_random_spec_leaves_some_paths_perfect_and_wobbles_the_rest_with_variety():
    specs = [random_spec(np.random.default_rng(i), PATH_CFG, duration_s=12.0) for i in range(100)]
    perfect = [s for s in specs if s.wobble is None]
    assert 15 <= len(perfect) <= 45
    wobbly = [s.wobble for s in specs if s.wobble is not None]
    w = PATH_CFG["wobble"]
    for wb in wobbly:
        assert w["amp_depth"][0] <= wb.amp_depth <= w["amp_depth"][1]
        assert w["growth"][0] <= wb.growth <= w["growth"][1]
        assert 0 <= wb.drift[0] <= w["drift"][1] and 0 <= wb.drift[1] <= w["drift"][1]
    assert np.std([wb.amp_depth for wb in wobbly]) > 0.01           # spread, not one value
    assert min(wb.amp_depth + wb.phase_depth + abs(wb.growth) for wb in wobbly) < 0.05   # some near-perfect


def test_random_sim_cfg_draws_shape_and_string():
    cfgs = [random_sim_cfg(np.random.default_rng(i), SIM_CFG) for i in range(60)]
    aspects = [c["blob"]["aspect"] for c in cfgs]
    assert min(aspects) >= 1.0 and max(aspects) <= 4.0 and np.std(aspects) > 0.3
    assert all(0 <= c["blob"]["angle"] < np.pi for c in cfgs)
    strung = [c for c in cfgs if c["blob"]["string"]]
    assert 15 <= len(strung) <= 45
    for c in strung:
        st = c["blob"]["string"]
        assert st["pivot"][1] < 0.0                                  # above the frame
        assert 60 <= st["intensity"] <= 150 and 1 <= st["thickness_px"] <= 3


def test_random_sim_cfg_draws_texture_and_an_optional_lag():
    cfgs = [random_sim_cfg(np.random.default_rng(i), SIM_CFG) for i in range(60)]
    depths = [c["blob"]["texture"]["depth"] for c in cfgs]
    assert min(depths) >= 0 and max(depths) <= 0.8 and np.std(depths) > 0.1
    assert len({c["blob"]["texture"]["seed"] for c in cfgs}) > 30      # not one pattern
    lagged = [c["v2e"]["cutoff_hz"] for c in cfgs if c["v2e"]["cutoff_hz"] > 0]
    assert 15 <= len(lagged) <= 45
    assert all(60 <= f <= 200 for f in lagged)


def test_random_sim_cfg_draws_inside_the_ranges_and_keeps_thresholds_symmetric():
    cfg = random_sim_cfg(np.random.default_rng(0), SIM_CFG)
    r = SIM_CFG["randomise"]
    assert r["pos_thres"][0] <= cfg["v2e"]["pos_thres"] <= r["pos_thres"][1]
    assert cfg["v2e"]["neg_thres"] == cfg["v2e"]["pos_thres"]
    assert r["radius_px"][0] <= cfg["blob"]["radius_px"] <= r["radius_px"][1]
    assert r["fg_intensity"][0] <= cfg["blob"]["fg_intensity"] <= r["fg_intensity"][1]
    assert "randomise" not in cfg
    assert SIM_CFG["v2e"]["pos_thres"] == 0.2                 # the template is untouched


def test_main_writes_clips_and_a_manifest_and_resumes(tmp_path):
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"sim": SIM_CFG}))
    out = tmp_path / "sim"
    args = ["--out", str(out), "--params", str(params), "--n", "2", "--seed", "1",
            "--no-distortion"]

    main(args)
    files = sorted(out.glob("*.npz"))
    assert [f.name for f in files] == ["sim_000.npz", "sim_001.npz"]
    with open(out / "manifest.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["name"] for r in rows] == ["sim_000", "sim_001"]
    assert {"shape", "period_s", "deviation", "deviation_t", "pos_thres", "radius_px",
            "aspect", "string", "wobble", "texture_depth", "cutoff_hz"} <= rows[0].keys()

    clip = load_clip(files[1])
    assert clip.gt is not None and len(clip.events) > 0
    assert clip.meta["resolution"] == (64, 48)
    assert clip.meta["sim_cfg"]["v2e"]["neg_thres"] == clip.meta["sim_cfg"]["v2e"]["pos_thres"]

    before = {f: f.stat().st_mtime_ns for f in files}
    main(args + ["--n", "3"])                       # one more clip; the first two stay put
    assert {f: f.stat().st_mtime_ns for f in files} == before
    assert (out / "sim_002.npz").exists()
    with open(out / "manifest.csv", newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 3
