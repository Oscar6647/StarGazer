"""
skynet_client.py -- minimal, NON-DESTRUCTIVE client for the Skynet 2 API (https://api.skynetgo.org).

Docs: https://docs.skynetgo.org/developers/   spec: https://docs.skynetgo.org/openapi.json
Auth: personal API token (skynetgo.org/u/<you>/settings/api-tokens) in .env next to this file
      (SKYNETTOKEN=... or SKYNET_TOKEN=...) or in an environment variable of either name,
      sent as "Authorization: Bearer <token>". A token scoped to observations:read + telescopes:read
      is recommended: scoped personal tokens cannot create observations, so nothing can be billed.

Safety: only GET requests and the four stateless observation-specs endpoints are allowed.
        POST /v1/observation-specs:publish (creates a real, charged observation) and every other
        POST/PUT/PATCH/DELETE are refused here, in code.
"""
from __future__ import annotations

import os
import re

try:                                   # Windows cert store (same as emphimeris_analyzer)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import requests

API = "https://api.skynetgo.org/v1"
STATELESS_POSTS = {                      # documented as "transient, never-persisted"
    "/observation-specs:validate",
    "/observation-specs:plan",
    "/observation-specs:observability",
    "/observation-specs:compatible-instruments",
}
SPEC_KIND_CANDIDATES = [("observation_spec", 1), ("observation", 1), ("ObservationSpec", 1)]


ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
TOKEN_VARS = ("SKYNET_TOKEN", "SKYNETTOKEN")


def _token_from_env():
    """Environment variable first, then the .env file next to this module."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)       # real env vars win over the file
    except ImportError:
        pass
    return next((os.environ[v].strip() for v in TOKEN_VARS if os.environ.get(v, "").strip()), None)


class SkynetError(RuntimeError):
    pass


class SkynetClient:
    def __init__(self, token=None, base=API, timeout=60):
        token = token or _token_from_env()
        if not token:
            raise SkynetError("No Skynet token: put SKYNETTOKEN=... (or SKYNET_TOKEN) in the .env "
                              "file next to skynet_client.py, or set it as an environment variable "
                              "(create one at https://skynetgo.org -> settings -> API tokens).")
        self.base, self.timeout = base.rstrip("/"), timeout
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})
        self._kind = None

    # ------------------------------------------------------------- transport
    def get(self, path, **params):
        r = self.s.get(self.base + path, params=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise SkynetError(f"GET {path} -> HTTP {r.status_code}: {r.text[:400]}")
        return r.json()

    def post(self, path, body):
        """Stateless endpoints only. Returns (status_code, json_or_text); never raises on 4xx so the
        caller can report validation errors."""
        if path not in STATELESS_POSTS:
            raise PermissionError(f"refusing POST {path}: not a stateless preflight endpoint")
        r = self.s.post(self.base + path, json=body, timeout=self.timeout)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, r.text[:400]

    # ------------------------------------------------------------- discovery
    def spec_kind(self, override=None):
        """(schemaKind, schemaVersion) for ObservationSpec envelopes. Not documented publicly, so:
        config override -> export the envelope of one of your own observations -> probe :validate."""
        if override:
            return tuple(override)
        if self._kind:
            return self._kind
        try:
            items = self.get("/observations", size=1).get("items") or []
            if items:
                oid = items[0].get("id") or items[0].get("uid")
                env = self.get(f"/observations/{oid}/spec", include_target="false",
                               include_requests="false", include_configuration="false")
                self._kind = (env["schemaKind"], int(env["schemaVersion"]))
                return self._kind
        except (SkynetError, KeyError):
            pass
        for kind, ver in SPEC_KIND_CANDIDATES:
            st, _ = self.post("/observation-specs:validate",
                              {"schemaKind": kind, "schemaVersion": ver, "data": {"name": "probe"}})
            if st == 200:
                self._kind = (kind, ver)
                return self._kind
        raise SkynetError("Could not determine the ObservationSpec schemaKind/schemaVersion. "
                          "Set SKYNET_SPEC_KIND = (kind, version) in capture_planner.py "
                          "(export any observation's spec in the web app to see it).")

    def imager_facts(self, name, binning=1):
        """Telescope you have observing access to, matched by name/shortName/slug -> its optical
        imager: uid, filters on the wheel(s), mount min altitude, local horizon, readout time."""
        tels = self.get("/me/observing-access/telescopes")
        want = name.lower()
        match = [t for t in tels
                 if want in {str(t.get(k) or "").lower() for k in ("name", "shortName", "slug")}]
        if not match:
            names = sorted(str(t.get("name")) for t in tels)
            raise SkynetError(f"Telescope {name!r} is not in your observing access. Available: {names}")
        t = match[0]
        imagers = [i for i in t.get("instruments") or [] if i.get("instrumentType") == "optical_imager"]
        if not imagers:
            raise SkynetError(f"{t.get('name')}: no optical imager")
        inst = imagers[0]

        wheels = inst.get("filterWheels") or t.get("filterWheels") or []
        filters = sorted({f for w in wheels for p in w.get("positions") or []
                          for f in p.get("filterIds") or []})
        cam = inst.get("camera") or {}
        modes = cam.get("readoutModes") or (cam.get("model") or {}).get("readoutModes") or []
        same_bin = [m for m in modes if m.get("binning") == binning] or modes
        readout = min((m["readoutTimeSec"] for m in same_bin if "readoutTimeSec" in m), default=None)
        return dict(
            telescope=t.get("name"), telescope_uid=t.get("uid"),
            instrument=inst.get("name"), instrument_uid=inst.get("uid"),
            filters=filters,
            min_alt_deg=(t.get("mount") or {}).get("minAltitudeDeg"),
            local_horizon=t.get("localHorizon"),          # [[az, alt], ...] or None
            readout_s=readout,
            pixel_scale=inst.get("pixelScale"), fov=inst.get("fov"),
        )


def match_filter(wanted, available):
    """Config filter id -> id on the wheel ('h-alpha' also matches 'h-alpha-5nm')."""
    for a in available:
        if a == wanted or re.fullmatch(re.escape(wanted) + r"-\d+nm", a):
            return a
    return None
