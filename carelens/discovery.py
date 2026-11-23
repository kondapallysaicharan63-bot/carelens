"""ONVIF WS-Discovery and stream-URI lookup (stdlib only).

Discovery is LAN-only UDP multicast: it finds standards-based (ONVIF) IP cameras on the
local network. Cloud-app cameras (Roku/Blink/Nest) do not answer WS-Discovery and will not
appear; they are clip-export only.

Everything received from the network is untrusted: packets are size-capped, payloads
containing DOCTYPE/ENTITY declarations are dropped, and malformed XML is ignored silently.
Credentials are never printed; URIs returned by cameras have any userinfo stripped.

    python -m carelens.discovery [--timeout 3] [--stream-uri]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import os
import re
import socket
import sys
import time
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from xml.sax.saxutils import escape as _xml_escape
from urllib.parse import unquote, urlsplit, urlunsplit

import yaml

from .config import redact_url, runtime_env

WS_ADDRESS = ("239.255.255.250", 3702)
MAX_PACKET = 65535
_NS_SOAP = "http://www.w3.org/2003/05/soap-envelope"
_NS_ONVIF_DEV = "http://www.onvif.org/ver10/device/wsdl"
_NS_ONVIF_MEDIA = "http://www.onvif.org/ver10/media/wsdl"
_NS_SCHEMA = "http://www.onvif.org/ver10/schema"


@dataclass
class DiscoveredDevice:
    epr: str
    xaddrs: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    name: str = ""
    hardware: str = ""
    location: str = ""

    @property
    def host(self) -> str:
        for x in self.xaddrs:
            h = urlsplit(x).hostname
            if h:
                return h
        return ""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


_CTRL = re.compile("[\x00-\x1f\x7f-\x9f\u2028\u2029]")


def _clean(s: str) -> str:
    """Strip control characters (C0/C1, DEL, line separators) from network-derived text."""
    return _CTRL.sub("", s or "")


def _safe_text(data: bytes) -> str | None:
    """Decode untrusted XML as strict UTF-8 (no UTF-16/32 tricks); None if unsafe."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\x00" in text:
        return None
    low = text.lower()
    if "<!doctype" in low or "<!entity" in low:
        return None
    return text


def _unsafe(data: bytes) -> bool:
    return _safe_text(data) is None


def host_allowed(url: str, allow_hosts=()) -> bool:
    """Only http(s) URLs to private/loopback/link-local IPs or .local names (or opted-in hosts)."""
    try:
        sp = urlsplit(url)
        host = sp.hostname
    except ValueError:
        return False
    if sp.scheme.lower() not in ("http", "https") or not host:
        return False
    if host.lower() in {h.lower() for h in allow_hosts}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host.lower().endswith(".local")
    return ip.is_private or ip.is_loopback or ip.is_link_local


def _scope_value(scopes: list[str], key: str) -> str:
    prefix = f"onvif://www.onvif.org/{key}/"
    for s in scopes:
        if s.lower().startswith(prefix):
            return _clean(unquote(s[len(prefix):]))
    return ""


def parse_probe_matches(xml_bytes: bytes) -> list[DiscoveredDevice]:
    """Parse a WS-Discovery ProbeMatches (or ResolveMatches) message; [] if malformed."""
    if not xml_bytes or len(xml_bytes) > MAX_PACKET or _unsafe(xml_bytes):
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except (ET.ParseError, ValueError):
        return []
    out: list[DiscoveredDevice] = []
    for m in root.iter():
        if _local(m.tag) not in ("ProbeMatch", "ResolveMatch"):
            continue
        epr, xaddrs, scopes = "", [], []
        for el in m.iter():
            n, text = _local(el.tag), _clean((el.text or "").strip())
            if n == "Address" and not epr:
                epr = text
            elif n == "XAddrs":
                xaddrs += [_clean(x) for x in text.split()]
            elif n == "Scopes":
                scopes += [_clean(x) for x in text.split()]
        if not epr and not xaddrs:
            continue
        out.append(DiscoveredDevice(
            epr=epr or xaddrs[0], xaddrs=xaddrs, scopes=scopes,
            name=_scope_value(scopes, "name"), hardware=_scope_value(scopes, "hardware"),
            location=_scope_value(scopes, "location")))
    return out


def _probe_message(message_id: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<e:Envelope xmlns:e="{_NS_SOAP}" xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
        'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
        'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        f'<e:Header><w:MessageID>uuid:{message_id}</w:MessageID>'
        '<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
        '<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>'
        '<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>'
        '</e:Envelope>').encode()


def probe(timeout: float = 3.0, address: tuple[str, int] = WS_ADDRESS,
          iface_ip: str | None = None) -> list[DiscoveredDevice]:
    """Send a WS-Discovery Probe and collect ProbeMatches for ``timeout`` seconds.

    Results are deduped by endpoint address. ``address`` may be unicast (for tests).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    found: dict[str, DiscoveredDevice] = {}
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.bind((iface_ip or "", 0))
        if iface_ip:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(iface_ip))
        sock.sendto(_probe_message(str(uuid.uuid4())), address)
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            sock.settimeout(left)
            try:
                data, _ = sock.recvfrom(MAX_PACKET)
            except (socket.timeout, BlockingIOError):
                break
            except OSError:
                break
            for d in parse_probe_matches(data):
                found.setdefault(d.epr, d)
    finally:
        sock.close()
    return list(found.values())


# --- ONVIF media (SOAP over HTTP) -------------------------------------------------------

def _security_header(username: str | None, password: str | None) -> str:
    if not username:
        return ""
    nonce = os.urandom(16)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(
        hashlib.sha1(nonce + created.encode() + (password or "").encode()).digest()).decode()
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        '<s:Header><Security xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">'
        f'<UsernameToken><Username>{esc(username)}</Username>'
        '<Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
        f'{digest}</Password><Nonce>{base64.b64encode(nonce).decode()}</Nonce>'
        f'<Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</Created>'
        '</UsernameToken></Security></s:Header>')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _soap(url: str, body: str, username, password, timeout: float) -> ET.Element:
    env = (f'<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="{_NS_SOAP}">'
           f'{_security_header(username, password)}<s:Body>{body}</s:Body></s:Envelope>').encode()
    req = urllib.request.Request(url, data=env, headers={
        "Content-Type": 'application/soap+xml; charset=utf-8'})
    with _OPENER.open(req, timeout=timeout) as r:
        data = r.read(1_000_000)
    if _unsafe(data):
        raise ValueError("unexpected DOCTYPE/ENTITY in SOAP response")
    return ET.fromstring(data)


def _strip_creds(uri: str) -> str:
    uri = _clean(uri)
    sp = urlsplit(uri)
    if "@" not in sp.netloc:
        return uri
    return urlunsplit(sp._replace(netloc=sp.netloc.rsplit("@", 1)[1]))


def _find_all(root: ET.Element, name: str):
    return [e for e in root.iter() if _local(e.tag) == name]


def get_stream_uri(xaddr: str, username: str | None = None, password: str | None = None,
                   timeout: float = 5, allow_hosts=()) -> list[tuple[str, str]]:
    """Return [(profile_token, rtsp_uri)] via ONVIF Media GetProfiles/GetStreamUri.

    Uses ``xaddr`` as the media endpoint, after trying GetCapabilities to find the real one.
    Credentials embedded in returned URIs are stripped.
    """
    if not host_allowed(xaddr, allow_hosts):
        raise ValueError(f"refusing ONVIF endpoint {redact_url(xaddr)}: only http(s) to "
                         "private/link-local/.local hosts (use --allow-host to opt in)")
    endpoint = xaddr
    try:
        caps = _soap(xaddr, f'<tds:GetCapabilities xmlns:tds="{_NS_ONVIF_DEV}">'
                            '<tds:Category>Media</tds:Category></tds:GetCapabilities>',
                     username, password, timeout)
        for media in _find_all(caps, "Media"):
            for x in _find_all(media, "XAddr"):
                cand = _clean((x.text or "").strip())
                if not cand:
                    continue
                if host_allowed(cand, allow_hosts):
                    endpoint = cand
                else:
                    print(f"warning: ignoring media endpoint {redact_url(cand)}", file=sys.stderr)
                break
    except Exception:
        pass  # many cameras serve media calls at the device XAddr
    profiles = _soap(endpoint, f'<trt:GetProfiles xmlns:trt="{_NS_ONVIF_MEDIA}"/>',
                     username, password, timeout)
    out = []
    for p in _find_all(profiles, "Profiles"):
        token = _clean(p.get("token") or "")
        if not token:
            continue
        body = (f'<trt:GetStreamUri xmlns:trt="{_NS_ONVIF_MEDIA}" xmlns:tt="{_NS_SCHEMA}">'
                '<trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream>'
                '<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup>'
                f'<trt:ProfileToken>{_xml_escape(token)}</trt:ProfileToken></trt:GetStreamUri>')
        try:
            resp = _soap(endpoint, body, username, password, timeout)
        except Exception:
            continue
        for u in _find_all(resp, "Uri"):
            if (u.text or "").strip():
                out.append((token, _strip_creds(u.text.strip())))
                break
    return out


def to_config_snippet(device: DiscoveredDevice, uri: str | None = None,
                      used_ids: set[str] | None = None) -> str:
    """YAML for config.yaml (credentials come from RTSP_USER / RTSP_PASS in .env).

    ``used_ids`` (mutated) makes ids unique across devices by suffixing -2, -3, ...
    """
    base = _clean(device.name) or device.host or "camera"
    slug = re.sub(r"[^a-z0-9_-]+", "-", base.lower()).strip("-_")[:28].strip("-_") or "camera"
    cid, n = slug, 1
    while used_ids is not None and cid in used_ids:
        n += 1
        cid = f"{slug}-{n}"
    if used_ids is not None:
        used_ids.add(cid)
    host = device.host
    if ":" in host:
        host = f"[{host}]"
    src = _strip_creds(uri) if uri else f"rtsp://{host or '<host>'}:554/stream1"
    entry = {"id": cid, "room": "unknown", "type": "rtsp", "source": src,
             "username_env": "RTSP_USER", "password_env": "RTSP_PASS"}
    body = yaml.safe_dump([entry], sort_keys=False, default_flow_style=False, allow_unicode=False)
    return "".join("  " + line + "\n" for line in body.splitlines())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find ONVIF cameras on the LAN (WS-Discovery)")
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument("--allow-host", action="append", default=[], metavar="HOST",
                    help="also trust ONVIF endpoints on this host (repeatable)")
    ap.add_argument("--stream-uri", action="store_true",
                    help="query RTSP URIs using RTSP_USER/RTSP_PASS from the environment/.env")
    a = ap.parse_args(argv)
    env = runtime_env() if a.stream_uri else {}
    devices = probe(timeout=a.timeout)
    if not devices:
        print("No ONVIF devices answered (LAN multicast only; cloud-app cameras will not appear).")
        return 0
    print("cameras:")
    used: set[str] = set()
    for d in devices:
        print(f"# {d.name or '?'} ({d.hardware or '?'}) at {d.host or '?'}  {redact_url(d.epr)}")
        uris: list[tuple[str, str]] = [("", "")]
        if a.stream_uri and d.xaddrs:
            try:
                uris = get_stream_uri(d.xaddrs[0], env.get("RTSP_USER"), env.get("RTSP_PASS"),
                                    allow_hosts=a.allow_host) or uris
            except Exception as e:
                print(f"#   stream-uri lookup failed: {type(e).__name__}", file=sys.stderr)
        print(to_config_snippet(d, uris[0][1] or None, used), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
