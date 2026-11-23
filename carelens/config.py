"""Declarative camera registry (config.yaml) validated with pydantic v2.

Cameras are declared, not hard-coded. Credentials are never stored in the file: a camera
names the environment variables (``username_env`` / ``password_env``) that hold them, and
they are read from the process environment or a local, git-ignored ``.env`` file.
"""
from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path
from typing import Literal, Mapping, Optional
from urllib.parse import quote

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class CameraType(str, Enum):
    file = "file"
    usb = "usb"
    rtsp = "rtsp"
    csi = "csi"
    cloud_clip = "cloud_clip"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Thresholds(_Strict):
    inactivity_min: float = Field(30.0, gt=0)
    fall_still_s: float = Field(20.0, gt=0)


_INLINE_CREDS = re.compile(r"^[a-z][a-z0-9+.-]*://[^/?#]*@", re.I)


class CameraConfig(_Strict):
    """One camera. ``cloud_clip`` is for clips exported from cloud-app cameras
    (Roku/Blink/Nest): those are clip-export only; live video requires RTSP/ONVIF/USB/CSI."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")
    room: str
    type: CameraType
    source: str
    username_env: Optional[str] = None
    password_env: Optional[str] = None
    orientation: Literal[0, 90, 180, 270] = 0
    person_height_m: float = Field(1.65, ge=0.5, le=2.5)
    enabled: bool = True
    thresholds: Optional[Thresholds] = None

    @model_validator(mode="after")
    def _check_source(self):
        s, t = self.source.strip(), self.type
        if t is CameraType.rtsp:
            if not s.lower().startswith(("rtsp://", "rtsps://")):
                raise ValueError("rtsp source must start with rtsp:// or rtsps://")
            if _INLINE_CREDS.match(s):
                raise ValueError("inline credentials (user:pass@) are not allowed in source; "
                                 "use username_env/password_env instead")
        elif t is CameraType.usb:
            if not s.isdigit():
                raise ValueError("usb source must be a non-negative device index, e.g. '0'")
        elif t is CameraType.csi:
            if not re.fullmatch(r"csi://\d+", s):
                raise ValueError("csi source must look like csi://0")
        else:  # file / cloud_clip
            if "://" in s:
                hint = (" Cloud-app cameras (Roku/Blink/Nest) are clip-export only: export a clip "
                        "from the vendor app; live video requires RTSP/ONVIF/USB/CSI."
                        if t is CameraType.cloud_clip else "")
                raise ValueError(f"{t.value} source must be a local file path, not a URL.{hint}")
        self.source = s
        return self


class StorageConfig(_Strict):
    out_dir: str = "output"
    retention_days: int = Field(7, ge=1)


class DashboardConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = 5000


class AppConfig(_Strict):
    cameras: list[CameraConfig] = []
    thresholds: Thresholds = Thresholds()
    storage: StorageConfig = StorageConfig()
    dashboard: DashboardConfig = DashboardConfig()

    @field_validator("cameras")
    @classmethod
    def _unique_ids(cls, cams):
        seen = set()
        for c in cams:
            if c.id in seen:
                raise ValueError(f"duplicate camera id: {c.id!r}")
            seen.add(c.id)
        return cams

    def camera(self, cam_id: str) -> CameraConfig:
        for c in self.cameras:
            if c.id == cam_id:
                return c
        raise KeyError(f"unknown camera {cam_id!r}; known ids: {[c.id for c in self.cameras]}")

    def thresholds_for(self, cam: CameraConfig) -> Thresholds:
        if cam.thresholds is None:
            return self.thresholds
        # Per-camera keys override only the fields they set; the rest come from the global block.
        return self.thresholds.model_copy(update=cam.thresholds.model_dump(exclude_unset=True))


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Tiny .env parser: KEY=VALUE, blanks/comments ignored, optional quotes stripped."""
    out: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        else:
            v = re.split(r"\s+#", v, maxsplit=1)[0]  # strip inline comment on unquoted values
        out[k.strip()] = v
    return out


def runtime_env(env_file: str | Path | None = ".env") -> dict[str, str]:
    """Process environment overlaid on ``env_file`` values (real env wins). Does not mutate os.environ."""
    real = {k: v for k, v in os.environ.items() if v}  # an empty export must not hide a .env value
    return {**(parse_env_file(env_file) if env_file else {}), **real}


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate the camera registry. Credentials are not read here (see runtime_env)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p} (see config.yaml in the repo root)")
    try:
        data = yaml.safe_load(p.read_text()) or {}
        return AppConfig.model_validate(data)
    except yaml.YAMLError as e:
        raise ValueError(f"{p}: invalid YAML: {e}") from None
    except ValidationError as e:
        raise ValueError(f"{p}: invalid configuration:\n{e}") from None


def redact_url(url: str) -> str:
    """Replace any userinfo in a URL with ***:***@ so it is safe to log."""
    return re.sub(r"^([a-z][a-z0-9+.-]*://)[^/?#]*@", r"\1***:***@", str(url), flags=re.I)


def resolve_source(cam: CameraConfig, env: Mapping[str, str] | None = None,
                   base_dir: str | Path | None = None) -> str | int:
    """Return what VideoSource accepts: an int for USB, else a string (creds injected for RTSP).

    File/cloud_clip paths are resolved against ``base_dir`` (the config file's directory) and
    returned absolute, so a digit-only filename is never mistaken for a webcam index.
    """
    env = os.environ if env is None else env
    if cam.type is CameraType.usb:
        return int(cam.source)
    if cam.type in (CameraType.file, CameraType.cloud_clip):
        return str((Path(base_dir or ".") / cam.source).resolve())
    if cam.type is not CameraType.rtsp or not (cam.username_env or cam.password_env):
        return cam.source
    vals = {}
    for name in (cam.username_env, cam.password_env):
        if name:
            if not env.get(name):
                raise ValueError(f"camera {cam.id!r}: environment variable {name} is not set "
                                 "(define it in .env or the environment)")
            vals[name] = env[name]
    user = quote(vals[cam.username_env], safe="") if cam.username_env else ""
    pw = quote(vals[cam.password_env], safe="") if cam.password_env else ""
    scheme, rest = cam.source.split("://", 1)
    userinfo = user + (f":{pw}" if cam.password_env else "")
    return f"{scheme}://{userinfo}@{rest}"
