import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)

from scripts import bench
from tests.test_experiment import a_clip_of_events, a_spec
from trajmem.snn_localise import SpikingLocaliser


def test_bench_renders_three_labelled_panels_and_an_index(tmp_path, monkeypatch):
    loc = SpikingLocaliser(n_cells=4, ch=(2, 3), hidden=8)
    loc.val_clips = ["sim_001"]
    loc.save(tmp_path / "loc.pt")
    clip = a_clip_of_events(a_spec(), duration_s=1.0)
    monkeypatch.setattr(bench, "open_clip", lambda path: (clip, "sim_001"))
    (tmp_path / "bench.yaml").write_text("clips:\n  - {clip: x, pool: sim blob}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    bench.main(["--bench", "bench.yaml", "--tag", "t", "--localiser-checkpoint", str(tmp_path / "loc.pt"),
                "--fps", "10", "--warmup", "0.5"])

    video = tmp_path / "runs" / "bench" / "t" / "sim_001.mp4"
    assert video.exists() and video.stat().st_size > 0
    import cv2

    cap = cv2.VideoCapture(str(video))
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 3 * 640 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 480 + bench._HEADER_PX
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 10
    index = (tmp_path / "runs" / "bench" / "t" / "index.md").read_text(encoding="utf-8")
    assert "sim_001" in index and "not trained on" in index
    for label, _, _ in bench.PIPELINES:
        assert label in index


def test_sim_status_falls_back_to_the_shared_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "corpus" / "sim" / "frames_8x_5000us"
    d.mkdir(parents=True)
    for i in range(20):
        (d / f"sim_{i:03d}.npz").write_bytes(b"")
    from trajmem.localise import split_frame_sets

    val = split_frame_sets(sorted(d.glob("sim_*.npz")))[1][0].stem
    assert bench.sim_status(val, object()) == "not trained on"
    assert bench.sim_status("sim_999", object()) == "trained on"


def test_the_bench_alarm_sees_every_memory_step_not_every_video_frame():
    clip = a_clip_of_events(a_spec(), duration_s=1.0)
    overlays = bench.make_overlays(clip, None, 5000, 0.1, 0.5, pipelines=bench.PIPELINES[:2])
    panels = bench.Panels(clip, overlays, "x", 0.02, 40, 1.0, 1.0)
    for t in [i / 10 for i in range(10)]:
        panels.frame(t)
    for alarm in panels.alarms:
        assert len(alarm.window) > 150                       # ~190 steps of 5 ms, not ~10 frames


def test_a_bench_panel_uses_the_alarm_constant_of_its_localiser_checkpoint(tmp_path):
    loc = SpikingLocaliser(n_cells=4, ch=(2, 3), hidden=8)
    loc.save(tmp_path / "pend_ft.pt")
    loc = SpikingLocaliser.load(tmp_path / "pend_ft.pt")
    clip = a_clip_of_events(a_spec(), duration_s=0.2)
    overlays = bench.make_overlays(clip, loc, 5000, 0.1, 0.5)
    panels = bench.Panels(clip, overlays, "x", 0.02, 40, 1.0, 1.0)
    assert [a.k for a in panels.alarms] == [25.0, 25.0, 30.0]
