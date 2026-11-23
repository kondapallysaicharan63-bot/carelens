import argparse
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from carelens.config import (AppConfig, load_config, parse_env_file, redact_url, runtime_env,
                             resolve_source)
from carelens.pipeline import resolve_args, rotate_frame


def cam(**kw):
    d = dict(id="c1", room="r", type="file", source="a.mp4")
    d.update(kw)
    return d


def app(*cams, **kw):
    return AppConfig.model_validate({"cameras": list(cams), **kw})


def test_valid_load(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"cameras": [cam()], "storage": {"retention_days": 3}}))
    cfg = load_config(p)
    assert cfg.camera("c1").room == "r" and cfg.storage.retention_days == 3
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
    p.write_text("cameras: [{id: BAD ID}]")
    with pytest.raises(ValueError):
        load_config(p)


def test_rejections():
    with pytest.raises(ValueError, match="duplicate"):
        app(cam(), cam())
    with pytest.raises(ValueError):
        app(cam(romm="x"))
    with pytest.raises(ValueError, match="username_env"):
        app(cam(type="rtsp", source="rtsp://u:p@h/s"))
    with pytest.raises(ValueError, match="clip-export"):
        app(cam(type="cloud_clip", source="rtsp://h/s"))
    for t, s in [("usb", "abc"), ("usb", "-1"), ("csi", "csi://x"), ("csi", "0"),
                 ("rtsp", "http://h/s"), ("file", "http://h/a.mp4")]:
        with pytest.raises(ValueError):
            app(cam(type=t, source=s))
    for h in (0.4, 2.6):
        with pytest.raises(ValueError):
            app(cam(person_height_m=h))
    with pytest.raises(ValueError):
        app(cam(orientation=45))
    app(cam(type="usb", source="0"), cam(id="c2", type="csi", source="csi://1"))


def test_resolve_source_creds():
    c = app(cam(type="rtsp", source="rtsp://h:554/s", username_env="U", password_env="P")).cameras[0]
    url = resolve_source(c, {"U": "bob", "P": "p@s:s/w"})
    assert url == "rtsp://bob:p%40s%3As%2Fw@h:554/s"
    with pytest.raises(ValueError) as e:
        resolve_source(c, {"U": "bob-secret-value"})
    assert "P" in str(e.value) and "bob-secret-value" not in str(e.value)
    assert resolve_source(app(cam(type="usb", source="2")).cameras[0]) == 2
    assert resolve_source(app(cam()).cameras[0]) == str(Path("a.mp4").resolve())


def test_redact_url():
    assert redact_url("rtsp://bob:pw@h:554/s") == "rtsp://***:***@h:554/s"
    assert redact_url("rtsp://h/s") == "rtsp://h/s"
    assert redact_url("clip.mp4") == "clip.mp4"


def test_env_file(tmp_path, monkeypatch):
    e = tmp_path / ".env"
    e.write_text('# c\n\nCT_A=1\nB="two words"\nC=\'x\'\nCT_ENV_X=file\n')
    assert parse_env_file(e) == {"CT_A": "1", "B": "two words", "C": "x", "CT_ENV_X": "file"}
    monkeypatch.setenv("CT_ENV_X", "real")
    monkeypatch.delenv("CT_A", raising=False)
    env = runtime_env(e)
    assert env["CT_ENV_X"] == "real" and env["CT_A"] == "1"
    assert "CT_A" not in os.environ  # .env must not leak into the process environment


def test_thresholds_and_lookup():
    cfg = app(cam(thresholds={"inactivity_min": 5}), cam(id="c2"),
              thresholds={"inactivity_min": 40})
    assert cfg.thresholds_for(cfg.camera("c1")).inactivity_min == 5
    assert cfg.thresholds_for(cfg.camera("c2")).inactivity_min == 40
    with pytest.raises(KeyError, match="c1"):
        cfg.camera("zzz")


def test_repo_config_loads():
    cfg = load_config(Path(__file__).parent.parent / "config.yaml")
    assert cfg.camera("demo").enabled


def test_rotate_frame():
    a = np.arange(6).reshape(2, 3)
    assert rotate_frame(a, 0) is a
    assert np.array_equal(rotate_frame(a, 90), np.array([[3, 0], [4, 1], [5, 2]]))
    assert np.array_equal(rotate_frame(a, 180), a[::-1, ::-1])
    assert np.array_equal(rotate_frame(a, 270), np.array([[2, 5], [1, 4], [0, 3]]))


def test_resolve_args():
    cfg = app(cam(person_height_m=1.8, thresholds={"inactivity_min": 5, "fall_still_s": 9}),
              cam(id="off", enabled=False), storage={"out_dir": "o2"})
    ns = argparse.Namespace(camera="c1", inactivity_min=None, fall_still_s=None, height_m=None,
                            out=None, name=None)
    _, src, pc, out, name = resolve_args(ns, cfg, base_dir="/cfgdir")
    assert (src, pc.inactivity_seconds, pc.fall_still_seconds, pc.person_height_m, out, name) == \
        (str(Path("/cfgdir/a.mp4").resolve()), 300, 9, 1.8, str(Path("/cfgdir/o2")), "c1")
    ns.inactivity_min, ns.height_m, ns.out = 1, 1.5, "x"
    _, _, pc, out, _ = resolve_args(ns, cfg)
    assert (pc.inactivity_seconds, pc.person_height_m, out) == (60, 1.5, "x")
    ns.camera = "off"
    with pytest.raises(ValueError, match="disabled"):
        resolve_args(ns, cfg)


def test_partial_camera_thresholds_inherit_global():
    cfg = app(cam(thresholds={"inactivity_min": 5}), thresholds={"fall_still_s": 10})
    th = cfg.thresholds_for(cfg.camera("c1"))
    assert (th.inactivity_min, th.fall_still_s) == (5, 10)


def test_file_source_is_absolute_and_never_a_webcam_index(tmp_path):
    from carelens.config import CameraConfig
    c = CameraConfig.model_validate(cam(source="20240101"))
    out = resolve_source(c, base_dir=tmp_path)
    assert out == str(tmp_path / "20240101") and not out.isdigit()


def test_env_inline_comment_and_empty_real_env(tmp_path, monkeypatch):
    e = tmp_path / ".env"
    e.write_text('CT_P=hunter2 # main cam\nCT_Q="a # b"\n')
    monkeypatch.setenv("CT_P", "")  # empty export must not hide the .env value
    env = runtime_env(e)
    assert env["CT_P"] == "hunter2" and env["CT_Q"] == "a # b"


def test_oriented_source_rotates_before_resize():
    from carelens.pipeline import OrientedSource
    from carelens.sources import Frame

    class Fake:
        fps = 10.0
        def __iter__(self):
            yield Frame(image=np.zeros((1080, 1920, 3), np.uint8), t=0.0, index=0)

    fr = next(iter(OrientedSource(Fake(), 90, "x")))
    assert fr.image.shape[:2] == (1137, 640)


def test_rtsp_transport_validation():
    from carelens.config import CameraConfig
    base = dict(id="c1", room="r", type="rtsp", source="rtsp://h/s")
    assert CameraConfig(**base).rtsp_transport == "tcp"
    assert CameraConfig(**base, rtsp_transport="udp").open_timeout_s == 10
    with pytest.raises(Exception):
        CameraConfig(**base, rtsp_transport="http")
    with pytest.raises(Exception):
        CameraConfig(**base, open_timeout_s=0)
