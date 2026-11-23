import base64
import hashlib
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import yaml

from carelens.config import AppConfig
from carelens.discovery import (DiscoveredDevice, get_stream_uri, parse_probe_matches, probe,
                                to_config_snippet)

SCOPES = ("onvif://www.onvif.org/name/Front%20Door onvif://www.onvif.org/hardware/IPC-1 "
          "onvif://www.onvif.org/location/living%20room onvif://www.onvif.org/Profile/S")


def matches(wsd, relates="uuid:x", epr="urn:uuid:cam-1"):
    return (f'<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
            f'xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing" xmlns:d="{wsd}">'
            f'<s:Header><a:RelatesTo>{relates}</a:RelatesTo></s:Header><s:Body><d:ProbeMatches>'
            f'<d:ProbeMatch><a:EndpointReference><a:Address>{epr}</a:Address></a:EndpointReference>'
            f'<d:Types>dn:NetworkVideoTransmitter</d:Types><d:Scopes>{SCOPES}</d:Scopes>'
            '<d:XAddrs>http://192.168.1.50:8080/onvif/device_service</d:XAddrs>'
            '</d:ProbeMatch></d:ProbeMatches></s:Body></s:Envelope>').encode()


def test_parse_both_namespaces_and_scopes():
    for ns in ("http://schemas.xmlsoap.org/ws/2005/04/discovery",
               "http://docs.oasis-open.org/ws-dd/ns/discovery/2009/01"):
        [d] = parse_probe_matches(matches(ns))
        assert d.epr == "urn:uuid:cam-1" and d.host == "192.168.1.50"
        assert (d.name, d.hardware, d.location) == ("Front Door", "IPC-1", "living room")
        assert len(d.scopes) == 4


def test_malformed_and_doctype_ignored():
    ns = "http://schemas.xmlsoap.org/ws/2005/04/discovery"
    good = matches(ns)
    bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]>' + good.split(b"?>", 1)[-1] \
        if good.startswith(b"<?xml") else b'<!DOCTYPE x [<!ENTITY a "aa">]>' + good
    assert parse_probe_matches(bomb) == []
    assert parse_probe_matches(b'<!doctype x>' + good) == []
    for junk in (b"", b"\x00\xff garbage", b"<a><b></a>", b"<a/>", b"x" * 70000):
        assert parse_probe_matches(junk) == []


def _responder(reply_twice=True):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(5)

    def run():
        try:
            data, addr = s.recvfrom(65535)
        except OSError:
            return
        import re
        mid = re.search(rb"MessageID>([^<]+)<", data).group(1).decode()
        pkt = matches("http://schemas.xmlsoap.org/ws/2005/04/discovery", relates=mid)
        for _ in range(2 if reply_twice else 1):
            s.sendto(pkt, addr)
        s.sendto(b"junk", addr)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return s, t


def test_probe_loopback_dedupes():
    s, t = _responder()
    try:
        devs = probe(timeout=0.6, address=("127.0.0.1", s.getsockname()[1]), iface_ip="127.0.0.1")
    finally:
        t.join(2)
        s.close()
    assert len(devs) == 1 and devs[0].epr == "urn:uuid:cam-1"


PROFILES = (b'<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body>'
            b'<trt:GetProfilesResponse xmlns:trt="http://www.onvif.org/ver10/media/wsdl">'
            b'<trt:Profiles token="prof1"/><trt:Profiles token="prof2"/>'
            b'</trt:GetProfilesResponse></s:Body></s:Envelope>')


def _stream(token):
    return (b'<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body>'
            b'<trt:GetStreamUriResponse xmlns:trt="http://www.onvif.org/ver10/media/wsdl">'
            b'<trt:MediaUri><tt:Uri xmlns:tt="http://www.onvif.org/ver10/schema">'
            b'rtsp://admin:hunter2@127.0.0.1:554/' + token + b'</tt:Uri></trt:MediaUri>'
            b'</trt:GetStreamUriResponse></s:Body></s:Envelope>')


def test_get_stream_uri_ws_security_and_cred_strip():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append(body.decode())
            b = body.decode()
            if "GetCapabilities" in b:
                out = b'<s:Envelope xmlns:s="x"><s:Body/></s:Envelope>'
            elif "GetProfiles" in b:
                out = PROFILES
            else:
                out = _stream(b"p2" if "prof2" in b else b"p1")
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        res = get_stream_uri(f"http://127.0.0.1:{srv.server_port}/onvif/device_service",
                             "admin", "s3cret-pw", timeout=3)
    finally:
        srv.shutdown()
        srv.server_close()
    assert res == [("prof1", "rtsp://127.0.0.1:554/p1"), ("prof2", "rtsp://127.0.0.1:554/p2")]
    assert seen and all("UsernameToken" in b and "PasswordDigest" in b for b in seen)
    assert not any("s3cret-pw" in b for b in seen)
    import re
    b = seen[1]
    nonce = base64.b64decode(re.search(r"<Nonce>([^<]+)</Nonce>", b).group(1))
    created = re.search(r"<Created[^>]*>([^<]+)</Created>", b).group(1)
    want = base64.b64encode(hashlib.sha1(nonce + created.encode() + b"s3cret-pw").digest()).decode()
    assert f">{want}</Password>" in b


def test_snippet_validates_against_config():
    d = DiscoveredDevice(epr="e", xaddrs=["http://10.0.0.9/onvif/device_service"], name="Front Door!")
    for uri in (None, "rtsp://u:p@10.0.0.9:554/s1"):
        snip = to_config_snippet(d, uri)
        cfg = AppConfig.model_validate({"cameras": yaml.safe_load("cameras:\n" + snip)["cameras"]})
        cam = cfg.cameras[0]
        assert cam.id == "front-door" and cam.type.value == "rtsp" and "@" not in cam.source


def test_snippet_injection_neutralised():
    import carelens.discovery as disc
    xml = matches("http://schemas.xmlsoap.org/ws/2005/04/discovery").replace(
        b"Front%20Door", b"Evil%0Aenabled:%20false%1b%0D")
    [d] = parse_probe_matches(xml)
    assert not any(c in d.name for c in "\n\r\x1b")
    uri = 'rtsp://10.0.0.9/x"\n    enabled: false'
    snip = to_config_snippet(d, uri)
    cams = yaml.safe_load("cameras:\n" + snip)["cameras"]
    assert len(cams) == 1 and set(cams[0]) == {"id", "room", "type", "source", "username_env",
                                               "password_env"}
    assert disc._clean("a\x85b\u2028c\x7f") == "abc"
    d2 = DiscoveredDevice(epr="e", name="x\ny: z")
    assert len(yaml.safe_load("cameras:\n" + to_config_snippet(d2, uri))["cameras"]) == 1


def test_untrusted_xaddr_refused_before_network(monkeypatch):
    import carelens.discovery as disc

    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(disc._OPENER, "open", boom)
    import pytest
    for bad in ("file:///etc/passwd", "ftp://10.0.0.1/x", "http://8.8.8.8/onvif",
                "http://example.com/onvif"):
        with pytest.raises(ValueError):
            get_stream_uri(bad, "u", "p")
    assert disc.host_allowed("http://192.168.1.5/x") and disc.host_allowed("https://cam.local/x")
    assert disc.host_allowed("http://[fe80::1]/x") and not disc.host_allowed("http://8.8.8.8/x")
    assert disc.host_allowed("http://example.com/x", ["example.com"])


def test_utf16_doctype_rejected():
    good = matches("http://schemas.xmlsoap.org/ws/2005/04/discovery").decode()
    bomb = '<?xml version="1.0" encoding="utf-16"?><!DOCTYPE x [<!ENTITY a "b">]>' + good
    assert parse_probe_matches(bomb.encode("utf-16")) == []
    assert parse_probe_matches(good.encode("utf-16-le")) == []
    assert parse_probe_matches(good.replace("<d:ProbeMatches>", "<!DocType x><d:ProbeMatches>").encode()) == []
    assert len(parse_probe_matches(good.encode())) == 1


def test_duplicate_ids_suffixed():
    used = set()
    d = DiscoveredDevice(epr="e", name="Cam")
    ids = [yaml.safe_load("c:\n" + to_config_snippet(d, None, used))["c"][0]["id"] for _ in range(3)]
    assert ids == ["cam", "cam-2", "cam-3"]


def test_ipv6_host_snippet():
    d = DiscoveredDevice(epr="e", xaddrs=["http://[fd00::5]:80/onvif/device_service"])
    cfg = AppConfig.model_validate({"cameras": yaml.safe_load("cameras:\n" + to_config_snippet(d))["cameras"]})
    assert "[fd00::5]" in cfg.cameras[0].source
