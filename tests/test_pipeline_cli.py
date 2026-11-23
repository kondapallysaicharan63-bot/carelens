import sys
from datetime import datetime

import pytest

from carelens import pipeline as P
from carelens.config import load_config
from carelens.store import Store


def test_data_label_and_names():
    assert [P.default_data_label(k) for k in ("usb", "rtsp", "csi", "file", "cloud_clip")] == \
        ["live", "live", "live", "test", "test"]
    t = datetime(2026, 3, 10, 8, 5, 9)
    assert P.default_run_name(None, "rtsp", "porch", t) == "porch_20260310-080509"
    assert P.default_run_name(None, "usb", "webcam", t) == "webcam_20260310-080509"
    assert P.default_run_name(None, "file", "x", t, file_stem="clip1") == "clip1"
    assert P.default_run_name("mine", "rtsp", "porch", t) == "mine"
    assert P.infer_source_kind("0") == "usb" and P.infer_source_kind("RTSPS://h/s") == "rtsp"
    assert P.infer_source_kind("csi://0") == "csi" and P.infer_source_kind("a.mp4") == "file"


def test_resolve_storage(tmp_path):
    st, out = P.resolve_storage(tmp_path / "missing.yaml", None)
    assert st.retention_days == 7 and out == "output"
    (tmp_path / "c.yaml").write_text("storage: {out_dir: data, retention_days: 3, "
                                     "rollup_retention_days: 30, db_path: x.db}\n")
    st, out = P.resolve_storage(tmp_path / "c.yaml", None)
    assert (st.retention_days, st.rollup_retention_days, st.db_path) == (3, 30, "x.db")
    assert out == str(tmp_path / "data")
    assert P.resolve_storage(tmp_path / "c.yaml", "elsewhere")[1] == "elsewhere"


class FakePose:
    closed = False

    def __init__(self, *a, **k):
        pass

    def close(self):
        FakePose.closed = True


def _patch(monkeypatch, tmp_path):
    import carelens.pose
    monkeypatch.setattr(carelens.pose, "MediaPipePose", FakePose)
    FakePose.closed = False
    (tmp_path / "out").mkdir()


def test_main_bad_database_is_clean_error(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    (tmp_path / "out" / "carelens.db").mkdir()  # a directory: sqlite cannot open it
    with pytest.raises(SystemExit) as e:
        P.main(["--source", "a.mp4", "--config", str(tmp_path / "none.yaml"),
                "--out", str(tmp_path / "out")])
    assert str(e.value).startswith("error:") and FakePose.closed


def test_main_purge_failure_does_not_hide_results(monkeypatch, tmp_path, capsys):
    _patch(monkeypatch, tmp_path)
    res = {"events": [], "heart_rate": {"summary": {}}, "breathing_rate": {"summary": {}},
           "gait_segments": []}
    monkeypatch.setattr(P, "run", lambda *a, **k: res)

    def boom(self, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(Store, "purge", boom)
    P.main(["--source", "a.mp4", "--config", str(tmp_path / "none.yaml"),
            "--out", str(tmp_path / "out")])
    cap = capsys.readouterr()
    assert "retention purge failed" in cap.err and '"events"' in cap.out
