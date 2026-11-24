import json
import os

import cv2
import pytest
import yaml

from carelens import cliplock as L
from carelens import pipeline as P
from carelens.config import StorageConfig
from carelens.store import Store
from carelens.synth import Scenario, Segment, render
from dashboard.app import create_app
from tests.test_store import make_results, ts_of


@pytest.fixture
def key():
    return L.load_key(L.generate_key())


def _enc(tmp_path, key, data=b"video-bytes", name="c_fall_3s.mp4"):
    p = tmp_path / name
    p.write_bytes(data)
    return L.encrypt_file(p, key), p


def test_roundtrip_and_plaintext_removed(tmp_path, key):
    enc, plain = _enc(tmp_path, key)
    assert enc.name == "c_fall_3s.mp4.enc" and not plain.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert enc.read_bytes().startswith(L.MAGIC) and b"video-bytes" not in enc.read_bytes()
    assert L.decrypt_to_bytes(enc, key) == b"video-bytes"


def test_wrong_key_tamper_rename_magic(tmp_path, key):
    enc, _ = _enc(tmp_path, key)
    with pytest.raises(L.ClipLockError, match="different key"):
        L.decrypt_to_bytes(enc, L.load_key(L.generate_key()))
    blob = bytearray(enc.read_bytes())
    blob[-20] ^= 1
    enc.write_bytes(bytes(blob))   # tampered, under its ORIGINAL name
    with pytest.raises(L.ClipLockError, match="authentication"):
        L.decrypt_to_bytes(enc, key)
    renamed = tmp_path / "other.mp4.enc"
    renamed.write_bytes(enc.read_bytes())
    with pytest.raises(L.ClipLockError, match="renamed"):
        L.decrypt_to_bytes(renamed, key)
    junk = tmp_path / "j.enc"
    junk.write_bytes(b"NOTCLENC" + os.urandom(80))
    with pytest.raises(L.ClipLockError, match="magic"):
        L.decrypt_to_bytes(junk, key)


def test_load_key_rejects_without_echo():
    for v in ("", "short", "!!!not-base64!!!secretvalue", L.generate_key()[:-8]):
        with pytest.raises(L.ClipLockError) as e:
            L.load_key(v)
        assert v not in str(e.value) or v == ""


def test_genkey_cli(capsys):
    L.main(["genkey"])
    cap = capsys.readouterr()
    assert len(L.load_key(cap.out.strip())) == 32 and "CARELENS_CLIP_KEY" in cap.err


def test_size_cap(tmp_path, key, monkeypatch):
    monkeypatch.setattr(L, "MAX_BYTES", 5)
    p = tmp_path / "x.mp4"
    p.write_bytes(b"123456")
    with pytest.raises(L.ClipLockError):
        L.encrypt_file(p, key)
    assert p.exists()


def _render(tmp_path):
    scn = Scenario([Segment("walk", 3, x0=0.2, x1=0.6),
                    Segment("transition", 0.5, posture="stand", to="lie", x0=0.6),
                    Segment("still", 8, posture="lie", x0=0.6)])
    clip, truth = tmp_path / "clip.mp4", tmp_path / "clip.truth.json"
    render(scn, clip)
    truth.write_text(json.dumps(scn.truth))
    return scn, clip, truth


def _cfg(tmp_path, **storage):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "cameras": [{"id": "hall", "room": "hallway", "type": "file", "source": "clip.mp4"}],
        "thresholds": {"fall_still_s": 3, "inactivity_min": 10},
        "storage": {"out_dir": "out", **storage}}))
    return cfg


def test_pipeline_encrypts_clips_and_dashboard_serves(tmp_path, monkeypatch):
    _, clip, truth = _render(tmp_path)
    keystr = L.generate_key()
    (tmp_path / ".env").write_text(f"CARELENS_CLIP_KEY={keystr}\n")
    monkeypatch.delenv("CARELENS_CLIP_KEY", raising=False)
    cfg = _cfg(tmp_path, encrypt_clips=True, clip_tmp_dir=str(tmp_path / "tmpfs"))
    P.main(["--camera", "hall", "--config", str(cfg), "--truth-pose", str(truth)])
    clips = tmp_path / "out" / "clips"
    assert not list(clips.glob("*.mp4")) and not list((tmp_path / "tmpfs").glob("*"))
    c = create_app(tmp_path / "out", env_dir=tmp_path).test_client()
    run = c.get(f"/api/run/{c.get('/api/runs').get_json()[0]['id']}").get_json()
    name = run["events"][0]["clip"]
    assert name.endswith(".mp4.enc") and (clips / name).is_file()
    r = c.get(f"/clips/{name}")
    assert r.status_code == 200 and r.mimetype == "video/mp4"
    dec = tmp_path / "dec.mp4"
    dec.write_bytes(r.data)
    cap = cv2.VideoCapture(str(dec))
    assert cap.read()[0]
    cap.release()
    # no key -> 403 JSON; traversal rejected; bad file -> 409
    nokey = create_app(tmp_path / "out", env_dir=tmp_path / "nowhere").test_client()
    r = nokey.get(f"/clips/{name}")
    assert r.status_code == 403 and "error" in r.get_json()
    assert c.get("/clips/../clip.mp4.enc").status_code == 404
    assert c.get("/clips/..%2f..%2fx.enc").status_code == 404
    (clips / "junk.mp4.enc").write_bytes(b"nope" * 20)
    assert c.get("/clips/junk.mp4.enc").status_code == 409
    # purge deletes the .enc clip
    with Store(tmp_path / "out" / "carelens.db", clock=lambda: ts_of("2099-01-01T00:00:00")) as st:
        assert st.purge(retention_days=1, clips_dir=clips)["files"] >= 1
    assert not (clips / name).exists()


def test_main_without_key_errors_before_camera(tmp_path, monkeypatch):
    _, clip, truth = _render(tmp_path)
    monkeypatch.delenv("CARELENS_CLIP_KEY", raising=False)
    cfg = _cfg(tmp_path, encrypt_clips=True)

    def boom(*a, **k):
        raise AssertionError("camera opened")
    monkeypatch.setattr(P, "VideoSource", boom)
    with pytest.raises(SystemExit) as e:
        P.main(["--camera", "hall", "--config", str(cfg), "--truth-pose", str(truth)])
    assert str(e.value).startswith("error:") and "CARELENS_CLIP_KEY" in str(e.value)


def test_storage_defaults():
    s = StorageConfig()
    assert s.encrypt_clips is False and s.clip_key_env == "CARELENS_CLIP_KEY" and s.clip_tmp_dir is None


def test_bad_tmp_dir_refused_at_startup(tmp_path, monkeypatch):
    _, clip, truth = _render(tmp_path)
    (tmp_path / ".env").write_text(f"CARELENS_CLIP_KEY={L.generate_key()}\n")
    monkeypatch.delenv("CARELENS_CLIP_KEY", raising=False)
    (tmp_path / "afile").write_text("x")
    cfg = _cfg(tmp_path, encrypt_clips=True, clip_tmp_dir="afile/sub")
    with pytest.raises(SystemExit) as e:
        P.main(["--camera", "hall", "--config", str(cfg), "--truth-pose", str(truth)])
    assert str(e.value).startswith("error:") and "clip_tmp_dir" in str(e.value)


def test_relative_tmp_dir_is_config_relative(tmp_path, monkeypatch):
    _, clip, truth = _render(tmp_path)
    (tmp_path / ".env").write_text(f"CARELENS_CLIP_KEY={L.generate_key()}\n")
    monkeypatch.delenv("CARELENS_CLIP_KEY", raising=False)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    cfg = _cfg(tmp_path, encrypt_clips=True, clip_tmp_dir="rel_tmp")
    P.main(["--camera", "hall", "--config", str(cfg), "--truth-pose", str(truth)])
    assert (tmp_path / "rel_tmp").is_dir() and not (cwd / "rel_tmp").exists()


def test_run_survives_unwritable_clip_dir(tmp_path):
    _, clip, truth = _render(tmp_path)
    (tmp_path / "blocker").write_text("x")
    from carelens.sources import VideoSource
    res = P.run(VideoSource(str(clip)), Scenario.from_truth(str(truth)).provider(), tmp_path / "o",
                P.Config(fall_still_seconds=3, inactivity_seconds=600), clip_key=L.load_key(L.generate_key()),
                clip_tmp_dir=tmp_path / "blocker" / "x")
    assert res["events"] and all(e["clip"] is None for e in res["events"])
    assert not list((tmp_path / "o" / "clips").glob("*"))


def test_sweep_orphans(tmp_path):
    out = tmp_path / "out"
    clips, tmpd = out / "clips", tmp_path / "tmpfs"
    clips.mkdir(parents=True)
    tmpd.mkdir()
    r = make_results("2026-03-10T12:00:00")
    r["events"][0]["clip"] = "kept.mp4.enc"
    for f in (clips / "kept.mp4.enc", clips / "orph.mp4", clips / "x.mp4.enc.tmp",
              clips / "old.mp4.enc", clips / "new.mp4.enc", tmpd / "t.mp4", tmpd / "keep.txt"):
        f.write_bytes(b"x")
    now = ts_of("2026-03-30T00:00:00")
    os.utime(clips / "old.mp4.enc", (now - 30 * 86400,) * 2)
    os.utime(clips / "new.mp4.enc", (now - 86400,) * 2)
    os.utime(clips / "kept.mp4.enc", (now - 30 * 86400,) * 2)
    with Store(tmp_path / "s.db", clock=lambda: now) as st:
        st.save_run(r, name="r")
        n = st.sweep_orphan_clips(clips, tmpd, retention_days=7)
    assert n == {"plaintext": 2, "partial": 1, "encrypted": 1}
    assert sorted(p.name for p in clips.iterdir()) == ["kept.mp4.enc", "new.mp4.enc"]
    assert [p.name for p in tmpd.iterdir()] == ["keep.txt"]


def _app(tmp_path, token=None):
    out = tmp_path / "out"
    (out / "clips").mkdir(parents=True)
    (out / "clips" / "a.mp4").write_bytes(b"vid")
    k = L.generate_key()
    (tmp_path / ".env").write_text(f"CARELENS_CLIP_KEY={k}\n" + (f"CARELENS_DASH_TOKEN={token}\n" if token else ""))
    p = out / "clips" / "b.mp4"
    p.write_bytes(b"secretvideo")
    L.encrypt_file(p, L.load_key(k))
    return create_app(out, env_dir=tmp_path)


def test_clips_gated_like_writes(tmp_path, monkeypatch):
    monkeypatch.delenv("CARELENS_CLIP_KEY", raising=False)
    monkeypatch.delenv("CARELENS_DASH_TOKEN", raising=False)
    c = _app(tmp_path).test_client()
    lan = {"environ_overrides": {"REMOTE_ADDR": "192.168.1.9"}}
    for n in ("a.mp4", "b.mp4.enc"):
        assert c.get(f"/clips/{n}", **lan).status_code == 403
        assert c.get(f"/clips/{n}").status_code == 200   # loopback
    t = tmp_path / "t"
    t.mkdir()
    c = _app(t, token="tok").test_client()
    for n in ("a.mp4", "b.mp4.enc"):
        assert c.get(f"/clips/{n}", **lan).status_code == 403
        assert c.get(f"/clips/{n}").status_code == 403   # token set: header always required
        assert c.get(f"/clips/{n}?token=tok", **lan).status_code == 403
        assert c.get(f"/clips/{n}", headers={"X-CareLens-Token": "tok"}, **lan).status_code == 200


def test_decrypt_cached_for_range_requests(tmp_path, monkeypatch):
    import dashboard.app as D
    monkeypatch.delenv("CARELENS_CLIP_KEY", raising=False)
    monkeypatch.delenv("CARELENS_DASH_TOKEN", raising=False)
    c = _app(tmp_path).test_client()
    calls = []
    real = D.decrypt_to_bytes
    monkeypatch.setattr(D, "decrypt_to_bytes", lambda *a: calls.append(1) or real(*a))
    assert c.get("/clips/b.mp4.enc").data == b"secretvideo"
    r = c.get("/clips/b.mp4.enc", headers={"Range": "bytes=0-3"})
    assert r.status_code == 206 and r.data == b"secr" and len(calls) == 1


def test_dashboard_config_fallback_for_key_env(tmp_path, monkeypatch):
    import dashboard.app as D
    monkeypatch.chdir(tmp_path)
    assert D.clip_key_env_for() == "CARELENS_CLIP_KEY"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "cameras": [], "storage": {"clip_key_env": "MY_KEY"}}))
    assert D.clip_key_env_for() == "MY_KEY"
