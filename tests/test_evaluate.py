import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np

from scripts import evaluate
from tests.test_experiment import a_clip_of_events, a_spec
from trajmem.trajectories import Deviation


def test_evaluate_writes_the_paper_tables(tmp_path, monkeypatch):
    steady = a_clip_of_events(a_spec(), duration_s=12.0)
    broken = a_clip_of_events(a_spec([Deviation(at_t=8.0, kind="shrink", params={"factor": 0.4})]), duration_s=12.0)
    broken.deviation_times.append(8.0)
    monkeypatch.setattr(evaluate, "load_set", lambda name: [{"name": "steady", "clip": "steady"},
                                                            {"name": "broken", "clip": "broken"}])
    monkeypatch.setattr(evaluate, "open_one", lambda e: {"steady": steady, "broken": broken}[e["clip"]])
    monkeypatch.chdir(tmp_path)

    evaluate.main(["--set", "development", "--memories", "kalman", "harmonic",
                   "--horizons", "0.05", "0.1", "--warmup", "3", "--settle", "4", "--out", "ev"])

    per_clip = (tmp_path / "ev" / "development" / "per_clip.csv").read_text(encoding="utf-8").splitlines()
    assert len(per_clip) == 1 + 2 * 2 * 2                      # header + clips x memories x horizons
    pooled = (tmp_path / "ev" / "development" / "pooled.csv").read_text(encoding="utf-8").splitlines()
    assert len(pooled) == 1 + 2 * 2                            # header + memories x horizons
    md = (tmp_path / "ev" / "development" / "tables.md").read_text(encoding="utf-8")
    assert "pred 50 ms" in md and "pred 100 ms" in md and "deviation detection" in md
    assert "kalman" in md and "harmonic" in md
    tex = (tmp_path / "ev" / "development" / "tables.tex").read_text(encoding="utf-8")
    assert tex.count(r"\begin{tabular}") == 2 and r"\bottomrule" in tex


def test_pool_takes_medians_per_memory_input_and_horizon():
    rows = [{"set": "s", "memory": "kalman", "input": "centroid", "horizon_s": 0.1, "clip": c,
             "err_median_px": v, "auc": np.nan if c == "a" else 0.9,
             **{f: 1.0 for f in evaluate.FIELDS if f not in
                ("set", "memory", "input", "horizon_s", "clip", "err_median_px", "auc")}}
            for c, v in (("a", 10.0), ("b", 20.0), ("c", 30.0))]
    pooled = evaluate.pool(rows)
    assert len(pooled) == 1
    assert pooled[0]["err_median_px"] == 20.0 and pooled[0]["n_clips"] == 3
    assert pooled[0]["n_break_clips"] == 2 and pooled[0]["auc"] == 0.9
