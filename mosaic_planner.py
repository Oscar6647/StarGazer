"""
mosaic_planner.py -- Skynet request planner for GZ02 (MDRS-WF) and GZ03 (HSC)

  1. Target extraction : per site, the four planets' ICRF (J2000) RA/Dec, az/el, Sun elevation over
                         the selected window (default: the conjunction window test.py found).
  2. Capture strategy  : one field per planet (planets share a field only when they have common usable
                         time AND stay inside one FOV for all of it); filter + exposure per planet from
                         instrument_analyzer.exposure_model.
  3. Request generation: Skynet 2 ObservationSpec per field (fixed ICRS target, window, elevation/Sun
                         limits, filter/exposure requests, imager pinned) -> {site}_mosaic_requests.json
  4. Preflight         : local checks always; with --remote also Skynet's stateless endpoints
                         observation-specs :validate, :compatible-instruments, :observability, :plan.
                         Nothing is created: skynet_client refuses :publish and every other write.
  5. Feasibility report: per field and per telescope, incl. the contiguous-mosaic tile count it avoids
                         -> mosaic_feasibility.csv

Usage (from TEST_BASE):
    python mosaic_planner.py                     # offline, reuses GZ02/GZ03 _conj_raw.csv
    python mosaic_planner.py --remote            # + Skynet discovery & preflight (needs SKYNET_TOKEN)
    python mosaic_planner.py --refresh           # rerun test.py's conjunction search first
    python mosaic_planner.py --start "2026-11-15 13:00" --stop "2026-11-15 14:30"   # UTC window
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import capture_planner as cfg
import test as mp
from emphimeris_analyzer import FMT, JPL_request, PLANETS, load_clean, load_sun
from instrument_analyzer import (FOUR, configuration_extent, diffraction_sampling,
                                 disk_arcsec, exposure_model, load_planets)
from skynet_client import SkynetClient, SkynetError, match_filter

STEP = timedelta(minutes=1)          # sample spacing of the refine files (cfg.REFINE_STEP)


def iso(t):
    return pd.Timestamp(t).tz_localize(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------- site limits (config or Skynet)
def site_limits(site, facts=None):
    sky = cfg.SKYNET[site]
    min_alt = max(cfg.MIN_ELEV, sky["horizon_deg"] or cfg.MIN_ELEV)
    src = "config"
    if facts and facts.get("min_alt_deg") is not None:
        min_alt, src = max(cfg.MIN_ELEV, facts["min_alt_deg"]), "Skynet mount"
    readout = facts.get("readout_s") if facts else None
    fov = (facts or {}).get("fov")
    return dict(min_alt=min_alt, min_alt_src=src,
                max_sun=cfg.SKYNET_MAX_SUN_DEG,
                fov_rad=tuple(math.radians(v) for v in fov) if fov else None,
                fov_src="Skynet" if fov else "config optics",
                horizon=(facts or {}).get("local_horizon") or None,
                filters=(facts or {}).get("filters") or sky["filters"],
                filters_src="Skynet wheel" if facts and facts.get("filters") else "config",
                overhead_s=readout if readout is not None else sky["overhead_s"],
                overhead_src="Skynet readout" if readout is not None else "config (ASSUMED)")


def horizon_at(az, limits):
    alt = np.full(np.shape(az), float(limits["min_alt"]))
    prof = limits["horizon"]
    if prof:
        p = np.array(sorted(prof), float)
        alt = np.maximum(alt, np.interp(np.asarray(az, float) % 360, p[:, 0], p[:, 1], period=360))
    return alt


def throughput(fid):
    if fid in cfg.FILTER_THROUGHPUT:
        return cfg.FILTER_THROUGHPUT[fid]
    m = re.fullmatch(r"(.+)-(\d+)nm", fid)
    if m:
        return int(m.group(2)) / 88
    raise KeyError(f"no throughput for filter {fid!r}: add it to capture_planner.FILTER_THROUGHPUT")


# ----------------------------------------------------------------- 1. target extraction
def site_coord(site):
    """GZ sites from cfg.SITES, Skynet-only telescopes from cfg.SKYNET_SITES."""
    return cfg.SITES.get(site) or cfg.SKYNET_SITES[site]


def window_file(site, start=None, stop=None, refresh=False):
    """Raw Horizons CSV for the selected window."""
    if start is not None:
        raw = f"{site}_mosaic_raw.csv"
        JPL_request(start, stop, cfg.REFINE_STEP, raw, SITE_COORD=mp.coord_str(site_coord(site)))
        return raw
    if site not in cfg.SITES:
        raise SystemExit(f"[{site}] is a Skynet-only site: give a window with --start/--stop")
    raw = f"{site}_conj_raw.csv"
    if refresh or not os.path.exists(raw):
        scan = mp.fetch(site, cfg.START, cfg.STOP, cfg.SCAN_STEP, "scan")
        if mp.find_conjunction(site, scan) is None:
            raise RuntimeError(f"[{site}] no visible conjunction window in {cfg.START}..{cfg.STOP}")
    return raw


def extract_targets(raw, limits):
    """Long table: one row per (planet, time) with ICRF coords and the usable flag."""
    df = load_clean(raw)
    sun = load_sun(raw)[["date", "elev"]].rename(columns={"elev": "sun_elev"})
    df = df.merge(sun, on="date", validate="many_to_one")
    df["name"] = df["planet"].map(PLANETS)
    df["dt"] = pd.to_datetime(df["date"], format=FMT)
    df["ok"] = ((df["elev"] >= horizon_at(df["azi"].values, limits))
                & (df["sun_elev"] <= np.minimum(df["name"].map(cfg.SUN_LIMIT), limits["max_sun"])))
    return df[["planet", "name", "dt", "ra_icrf", "dec_icrf", "azi", "elev", "sun_elev",
               "apmag", "delta", "ok"]].sort_values(["planet", "dt"]).reset_index(drop=True)


# ----------------------------------------------------------------- 2. capture strategy
def site_instrument(site, filt=None):
    """capture_planner instrument with Skynet binning and (optionally) filter throughput applied.
    CMOS binning approximated as: px/b, full well * b^2, read noise * b, dark * b^2."""
    b = cfg.SKYNET[site]["binning"]
    inst = mp.make_instrument(cfg.SITE_INSTRUMENT.get(site) or cfg.SKYNET[site]["instrument"])
    inst = replace(inst, h_px=inst.h_px // b, v_px=inst.v_px // b, fwc=inst.fwc * b * b,
                   read_noise=inst.read_noise * b, dark_current=inst.dark_current * b * b)
    if filt:
        inst = replace(inst, name=f"{inst.name} {filt}", t_opt=inst.t_opt * throughput(filt))
    return inst


def choose_exposures(site, raw, targets, limits):
    """Per planet: first available filter in FILTER_PREFERENCE with max(t_snr, min_exp) <= t_max."""
    sky = cfg.SKYNET[site]
    t = load_planets(raw)
    vis = (targets.pivot(index="dt", columns="planet", values="ok")
           .reindex(index=t["elev"].index, columns=FOUR).fillna(False).astype(bool))
    blur = diffraction_sampling(site_instrument(site))["theta_eff_arcsec"]
    order = [(w, match_filter(w, limits["filters"])) for w in cfg.FILTER_PREFERENCE[site]]
    missing = [w for w, f in order if f is None]
    if missing:
        print(f"    filters not on the {limits['filters_src']}: {missing} (skipped)")

    out, first = {}, {}
    for _, filt in order:
        if filt is None:
            continue
        expo, _ = exposure_model(site_instrument(site, filt), t, vis, blur, cfg.SKY_MAG)
        for name, r in expo.iterrows():
            first.setdefault(name, (filt, r))
            if name in out or r["visible_samples"] == 0:
                continue
            lo, hi = max(r["t_snr_s"], sky["min_exp_s"]), r["t_max_s"]
            if lo <= hi:
                exp = float(np.clip(r["t_suggest_s"], lo, hi))
                out[name] = dict(filter=filt, exp_s=float(f"{exp:.3g}"), ok=True,
                                 t_snr_s=r["t_snr_s"], t_sat_s=r["t_sat_s"], why="")
    for name, (filt, r) in first.items():          # nothing worked: report why, first filter
        if name in out:
            continue
        if r["visible_samples"] == 0:
            why = "never usable in window"
        elif r["t_sat_s"] < sky["min_exp_s"]:
            why = f"saturates before min exposure in every filter (t_sat {r['t_sat_s']:.2g}s)"
        else:
            why = "target SNR not reachable before saturation/trailing"
        out[name] = dict(filter=filt, exp_s=sky["min_exp_s"], ok=False,
                         t_snr_s=r["t_snr_s"], t_sat_s=r["t_sat_s"], why=why)
    return out


def _extent_rad(points):
    return configuration_extent(points["ra_icrf"].values, points["dec_icrf"].values)


def _centre(points):
    a, d = np.radians(points["ra_icrf"].values), np.radians(points["dec_icrf"].values)
    v = np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)]).sum(axis=1)
    v /= np.linalg.norm(v)
    return math.degrees(math.atan2(v[1], v[0])) % 360, math.degrees(math.asin(v[2]))


def _fov(inst, fov_rad):
    return fov_rad if fov_rad else (inst.fov_h, inst.fov_v)


def field_geometry(targets, members, inst, fov_rad=None):
    """Common usable time of the members, their pointing centre, and whether all of them
    (incl. motion over that time and disk size) stay inside the FOV minus margin."""
    usable = targets[targets["ok"] & targets["name"].isin(members)]
    common = set.intersection(*(set(g["dt"]) for _, g in usable.groupby("name"))) \
        if usable["name"].nunique() == len(members) else set()
    if not common:
        return None
    pts = usable[usable["dt"].isin(common)]
    ext_h, ext_v = _extent_rad(pts) if len(pts) > 1 else (0.0, 0.0)
    disk = math.radians(disk_arcsec(pts["delta"].values[:, None].repeat(4, 1)).max() / 3600)
    fov_h, fov_v = _fov(inst, fov_rad)
    lim_h, lim_v = fov_h * (1 - inst.margin), fov_v * (1 - inst.margin)
    ra, dec = _centre(pts)
    return dict(members=list(members), start=min(common), stop=max(common),
                n_samples=len(common), ra_deg=ra, dec_deg=dec,
                mags=pts.groupby("name")["apmag"].mean().to_dict(),
                extent_arcmin=(math.degrees(ext_h) * 60, math.degrees(ext_v) * 60),
                fov_arcmin=(math.degrees(fov_h) * 60, math.degrees(fov_v) * 60),
                fits=(ext_h + disk <= lim_h) and (ext_v + disk <= lim_v))


def plan_fields(targets, inst, fov_rad=None):
    """Greedy: one field per planet; merge a planet into an earlier field only if the merged
    field still fits for their whole common usable time."""
    fields = []
    for name in [PLANETS[p] for p in FOUR]:
        own = field_geometry(targets, [name], inst, fov_rad)
        if own is None:
            fields.append(dict(members=[name], unusable=True))
            continue
        for i, f in enumerate(fields):
            if f.get("unusable"):
                continue
            merged = field_geometry(targets, f["members"] + [name], inst, fov_rad)
            if merged is not None and merged["fits"]:
                fields[i] = merged
                break
        else:
            fields.append(own)
    return fields


def contiguous_mosaic(targets, inst, fov_rad=None):
    """Tiles a single continuous mosaic would need at the window midpoint (gnomonic extent of the
    four planets; overestimates somewhat for very wide fields)."""
    times = sorted(targets["dt"].unique())
    mid = targets[targets["dt"] == times[len(times) // 2]]
    ext_h, ext_v = _extent_rad(mid)
    fov_h, fov_v = _fov(inst, fov_rad)
    step_h, step_v = fov_h * (1 - cfg.MOSAIC_OVERLAP), fov_v * (1 - cfg.MOSAIC_OVERLAP)
    nx, ny = math.ceil(ext_h / step_h), math.ceil(ext_v / step_v)
    return dict(extent_deg=(math.degrees(ext_h), math.degrees(ext_v)), nx=nx, ny=ny, tiles=nx * ny)


# ----------------------------------------------------------------- 3. request generation
def skynet_spec(site, field, exposures, limits, facts):
    """Skynet 2 ObservationSpec 'data' (camelCase, see openapi ObservationSpec)."""
    sky = cfg.SKYNET[site]
    names = field["members"]
    mags = field["mags"]
    stop = field["stop"] + STEP
    spec = {
        "name": f"StarGazer {site} {'+'.join(names)}",
        "startAfter": iso(field["start"]),
        "cancelOn": iso(stop),
        "target": {
            "name": "+".join(names),
            "position": {"positionType": "fixed",
                         "coordinates": {"coordinateType": "equatorial",
                                         "raDeg": round(field["ra_deg"], 6),
                                         "decDeg": round(field["dec_deg"], 6)}},
        },
        "opticalImagingConfiguration": {
            "trackingMode": "sidereal",
            "ditherStrategy": "none", "temporalOffsetSec": 0.0,
            "maxTiles": 1, "tileOverlap": 0.0,
            "minElevationDeg": float(limits["min_alt"]),
            "maxSunElevationDeg": float(min(min(cfg.SUN_LIMIT[n] for n in names), limits["max_sun"])),
            "brightnessModel": {"modelType": "source", "isPointSource": True,
                                "magnitudeMag": round(max(mags.values()), 2), "filterId": "V"},
            "saturationModel": {"modelType": "source", "isPointSource": True,
                                "magnitudeMag": round(min(mags.values()), 2), "filterId": "V"},
        },
        "requests": [{"requestType": "optical_imaging", "order": i,
                      "filterSpecifierIds": [exposures[n]["filter"]],
                      "exposureTimeSec": exposures[n]["exp_s"],
                      "sampleCount": sky["count"],
                      "allowBinning": sky["binning"] > 1}
                     for i, n in enumerate(names)],
    }
    if facts and facts.get("instrument_uid"):
        spec["instrumentMode"] = "include"
        spec["instrumentUids"] = [facts["instrument_uid"]]
    return spec


def build_request(site, field, exposures, limits, facts):
    names = field["members"]
    return {
        "site": site,
        "telescope": (facts or {}).get("telescope") or cfg.SKYNET[site]["skynet_name"],
        "instrument_uid": (facts or {}).get("instrument_uid"),
        "target": "+".join(names),
        "ra_deg": round(field["ra_deg"], 5),
        "dec_deg": round(field["dec_deg"], 5),
        "window_utc": [iso(field["start"]), iso(field["stop"] + STEP)],
        "exposures": [{"for": n, "filter": exposures[n]["filter"], "exp_s": exposures[n]["exp_s"],
                       "count": cfg.SKYNET[site]["count"]} for n in names],
        "skynetSpec": skynet_spec(site, field, exposures, limits, facts),
    }


# ----------------------------------------------------------------- 4. preflight
def local_preflight(site, field, req, exposures, limits):
    sky = cfg.SKYNET[site]
    names = field["members"]
    window_s = (field["stop"] - field["start"] + STEP).total_seconds()
    need_s = sum(e["count"] * (e["exp_s"] + limits["overhead_s"]) for e in req["exposures"])
    return {
        "visible": (True, f"{field['n_samples']} usable min, "
                          f"{field['start']:%H:%M}-{field['stop'] + STEP:%H:%M} UTC"),
        "filters": (all(e["filter"] in limits["filters"] for e in req["exposures"]),
                    ", ".join(f"{e['for']}:{e['filter']}" for e in req["exposures"])),
        "exposure": (all(exposures[n]["ok"] for n in names),
                     "; ".join(f"{n} {exposures[n]['exp_s']}s"
                               + ("" if exposures[n]["ok"] else f" ({exposures[n]['why']})")
                               for n in names)),
        "time_budget": (need_s <= window_s, f"needs {need_s:.0f}s of {window_s:.0f}s "
                                            f"(overhead {limits['overhead_s']:.3g}s/frame)"),
        "in_fov": (field["fits"], "extent {:.1f}x{:.1f}' vs FOV {:.1f}x{:.1f}'".format(
            *field["extent_arcmin"], *field["fov_arcmin"])),
    }, need_s, window_s


def _detail(body):
    if isinstance(body, dict) and "detail" in body:
        return json.dumps(body["detail"])[:300]
    return str(body)[:300]


def remote_preflight(client, req, facts):
    """Skynet's stateless endpoints. Returns {check: (pass, detail)} plus raw responses."""
    kind, ver = client.spec_kind(cfg.SKYNET_SPEC_KIND)
    env = {"schemaKind": kind, "schemaVersion": ver, "data": req["skynetSpec"]}
    uid = facts["instrument_uid"]
    t0, t1 = req["window_utc"]
    span_days = max((pd.Timestamp(t1) - pd.Timestamp(t0)).total_seconds() / 86400, 1 / 24)
    checks, raw = {}, {}

    st, body = client.post("/observation-specs:validate", env)
    raw["validate"] = body
    checks["sky:validate"] = (st == 200, "ok" if st == 200 else f"HTTP {st} {_detail(body)}")

    st, body = client.post("/observation-specs:compatible-instruments",
                           {"spec": env, "instrumentUids": [uid]})
    raw["compatible"] = body
    res = (body.get("results") or [{}])[0] if st == 200 and isinstance(body, dict) else {}
    checks["sky:compatible"] = (bool(res.get("compatible")),
                                "; ".join(res.get("reasons") or []) or
                                ("compatible" if res.get("compatible") else f"HTTP {st} {_detail(body)}"))

    st, body = client.post("/observation-specs:observability",
                           {"spec": env, "instrumentUid": uid, "start": t0,
                            "spanDays": span_days, "resolutionSec": 60})
    raw["observability"] = body
    if st == 200 and isinstance(body, dict):
        wins = body.get("windows") or []
        good = [w for w in wins if w.get("observable")]
        mins = sum((pd.Timestamp(w["endsOn"]) - pd.Timestamp(w["startsOn"])).total_seconds()
                   for w in good) / 60
        reasons = sorted({w.get("reason", "") for w in wins if not w.get("observable")} - {""})
        checks["sky:observable"] = (bool(good), f"{mins:.0f} observable min"
                                    + (f"; blocked: {', '.join(reasons)}" if reasons else ""))
    else:
        checks["sky:observable"] = (False, f"HTTP {st} {_detail(body)}")

    st, body = client.post("/observation-specs:plan",
                           {"spec": env, "instrumentUid": uid, "startsOn": t0, "endsOn": t1})
    raw["plan"] = body
    if st == 200 and isinstance(body, dict):
        plan = body.get("opticalImagingConfigurationPlan") or {}
        series = plan.get("seriesPlans") or []
        capped = [s for s in series if s.get("exposureCapped")]
        changed = [s for s in series
                   if s.get("exposureSec") is not None
                   and abs(s["exposureSec"] - req["exposures"][s.get("requestOrder", 0)]["exp_s"]) > 1e-3]
        cons = body.get("constraints") or []
        det = (f"{len(plan.get('tiles') or [])} tile(s), {plan.get('totalExposures')} exp, "
               f"{plan.get('totalExposureTimeSec')} s exposing, FOV {60 * (plan.get('xFov') or 0):.1f}x"
               f"{60 * (plan.get('yFov') or 0):.1f}'")
        if not series:
            det += "; per-frame exposure NOT checked by Skynet (fixed-exposure mode)"
        if changed:
            det += "; Skynet exposure " + ", ".join(
                f"{req['exposures'][s.get('requestOrder', 0)]['for']}={s['exposureSec']}s" for s in changed)
        if cons:
            det += "; constraints: " + "; ".join(map(str, cons))
        checks["sky:plan"] = (bool(plan) and not capped and not changed, det)
    else:
        checks["sky:plan"] = (False, f"HTTP {st} {_detail(body)}")
    return checks, raw


# ----------------------------------------------------------------- 5. report
def plan_site(site, raw, client=None):
    sky = cfg.SKYNET[site]
    print(f"\n==================== {site} / {sky['skynet_name']} ====================")
    facts = None
    if client:
        try:
            facts = client.imager_facts(sky["skynet_name"], sky["binning"])
            print(f"[skynet] {facts['telescope']} / {facts['instrument']} uid={facts['instrument_uid']}\n"
                  f"         filters={facts['filters']}  min_alt={facts['min_alt_deg']}  "
                  f"readout={facts['readout_s']}s  pixel_scale={facts['pixel_scale']}  fov={facts['fov']}")
        except SkynetError as e:
            print(f"[skynet] discovery failed -> config values used: {e}")
    limits = site_limits(site, facts)

    targets = extract_targets(raw, limits)
    inst = site_instrument(site)
    print(f"[1] {raw}: {targets['dt'].min()} -> {targets['dt'].max()} UTC | min alt "
          f"{limits['min_alt']} deg ({limits['min_alt_src']}"
          f"{', + local horizon' if limits['horizon'] else ''}) | Sun <= {limits['max_sun']} | "
          f"FOV from {limits['fov_src']} | plate {inst.plate_x:.2f}\"/px "
          f"(bin {sky['binning']})")
    print(targets[targets["ok"]].groupby("name").agg(
        usable_min=("dt", "size"), first=("dt", "min"), last=("dt", "max"),
        elev_max=("elev", "max"), mag=("apmag", "mean")).round(2).to_string())

    exposures = choose_exposures(site, raw, targets, limits)
    fields = plan_fields(targets, inst, limits["fov_rad"])
    mosaic = contiguous_mosaic(targets, inst, limits["fov_rad"])
    print(f"\n[2] {sum(not f.get('unusable') for f in fields)} field(s) for 4 planets "
          f"vs contiguous mosaic {mosaic['extent_deg'][0]:.1f} x {mosaic['extent_deg'][1]:.1f} deg "
          f"= {mosaic['nx']} x {mosaic['ny']} = {mosaic['tiles']} tiles")

    requests_out, rows = [], []
    for f in fields:
        if f.get("unusable"):
            print(f"\n  {'+'.join(f['members'])}: never usable in this window -> no request")
            rows.append(dict(site=site, target="+".join(f["members"]), feasible=False,
                             reason="never usable in window"))
            continue
        req = build_request(site, f, exposures, limits, facts)
        checks, need_s, window_s = local_preflight(site, f, req, exposures, limits)
        if client and facts:
            try:
                remote, raw_resp = remote_preflight(client, req, facts)
                checks.update(remote)
                req["skynetResponses"] = raw_resp
            except SkynetError as e:
                checks["sky:preflight"] = (False, str(e))
        req["preflight"] = {k: {"pass": bool(p), "detail": d} for k, (p, d) in checks.items()}
        requests_out.append(req)
        ok = all(p for p, _ in checks.values())
        print(f"\n  {req['target']}: RA {req['ra_deg']:.4f}  Dec {req['dec_deg']:+.4f}  "
              f"-> {'FEASIBLE' if ok else 'NOT feasible'}")
        for k, (p, d) in checks.items():
            print(f"    [{'ok' if p else 'XX'}] {k:<15} {d}")
        rows.append(dict(site=site, telescope=req["telescope"], target=req["target"],
                         ra_deg=req["ra_deg"], dec_deg=req["dec_deg"],
                         window_start=req["window_utc"][0], window_stop=req["window_utc"][1],
                         filters=";".join(e["filter"] for e in req["exposures"]),
                         exp_s=";".join(str(e["exp_s"]) for e in req["exposures"]),
                         need_s=round(need_s), window_s=round(window_s),
                         remote_checked=bool(client and facts), feasible=ok,
                         reason="; ".join(k for k, (p, _) in checks.items() if not p)))

    out = f"{site}_mosaic_requests.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(requests_out, fh, indent=2, default=str)
    print(f"\n[3] {len(requests_out)} request spec(s) -> {out}")
    print("[4] remote Skynet preflight: " + ("done (stateless endpoints, nothing created)"
                                             if client and facts else
                                             "skipped (run with --remote and SKYNET_TOKEN)"))

    summary = dict(site=site, telescope=sky["skynet_name"],
                   fields=len(requests_out), feasible=sum(r["feasible"] for r in rows),
                   remote=bool(client and facts),
                   mosaic_tiles=mosaic["tiles"],
                   mosaic_min_time_s=round(mosaic["tiles"] * sky["count"] * limits["overhead_s"]),
                   fields_time_s=sum(r.get("need_s", 0) for r in rows))
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description="Skynet request planner (GZ02/GZ03)")
    ap.add_argument("--sites", nargs="+", default=cfg.MOSAIC_SITES)
    ap.add_argument("--start", help='UTC "YYYY-MM-DD HH:MM" (needs --stop; fetches a new window)')
    ap.add_argument("--stop")
    ap.add_argument("--refresh", action="store_true", help="rerun test.py's conjunction search")
    ap.add_argument("--remote", action="store_true",
                    help="query Skynet (discovery + stateless preflight); needs SKYNET_TOKEN")
    a = ap.parse_args()
    if bool(a.start) != bool(a.stop):
        ap.error("--start and --stop go together")
    parse = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M")
    start, stop = (parse(a.start), parse(a.stop)) if a.start else (None, None)
    client = SkynetClient() if a.remote else None

    all_rows, summaries = [], []
    for site in a.sites:
        if site not in cfg.SKYNET:
            print(f"Skipping {site}: no SKYNET entry in capture_planner.py")
            continue
        rows, summary = plan_site(site, window_file(site, start, stop, a.refresh), client)
        all_rows += rows
        summaries.append(summary)

    pd.DataFrame(all_rows).to_csv("mosaic_feasibility.csv", index=False)
    print("\n===================== [5] FEASIBILITY (both telescopes) =====================")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nmosaic_min_time_s = tiles x count x overhead only (lower bound, no exposure time).")
    print("Per-field details -> mosaic_feasibility.csv; full specs + Skynet responses -> *_mosaic_requests.json")
    if not client:
        print("Offline run: min exposure, horizon, overhead and filters come from capture_planner (ASSUMED).")


if __name__ == "__main__":
    main()
