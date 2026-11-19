"""
mission_planner.py  -- one pipeline, one set of rules (capture_planner.py)

  A. Conjunction search  : per site, tightest 4-planet grouping that is VISIBLE (all >= MIN_ELEV,
                           Sun <= MAX_SUN_ALL). Coarse scan -> 1-min refine.
  B. Capture plan        : shortest time span, across sites, containing one usable slot per planet
                           (per-planet Sun limits). Uses the same coarse scan as A.
  C. Optics / exposure   : for every chosen time, fetch a 1-min window around it and run
                           instrument_analyzer.optical_geometry on exactly that file.
  D. Heliocentric view   : Sun-centred JPL vectors (orbital_analyzer.py) -> where the planets are
                           around the Sun vs. where Earth sees them at the best moment.
"""
from __future__ import annotations

import inspect
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import capture_planner as cfg
from emphimeris_analyzer import (JPL_request, angular_separation_analysis, per_timestamp,
                                 load_clean, load_sun, PLANETS, FMT)
from instrument_analyzer import Instrument, optical_geometry

ALL_NAMES = [v for k, v in PLANETS.items() if k != "10"]
PAD = timedelta(minutes=cfg.REFINE_HALF_WINDOW_MIN)
FAILED = {}          # site -> reason, filled by load_all, printed with the SUMMARY


# ----------------------------------------------------------------- helpers
def coord_str(c):
    return ",".join(str(x) for x in c)


def make_instrument(key: str) -> Instrument:
    spec = cfg.INSTRUMENTS[key]
    params = inspect.signature(Instrument).parameters
    bad = sorted(set(spec) - set(params))
    if bad:
        raise KeyError(f"[{key}] not Instrument fields: {bad}")
    return Instrument(name=key, **spec)


def fetch(site, start, stop, step, tag):
    """Ephemeris file for a site; returns its path. Reuses the file on disk only if it was made
    for exactly this site position + window + step (recorded in <file>.key) and loads cleanly."""
    raw = f"{site}_{tag}_raw.csv"
    key_file = raw + ".key"
    key = f"{coord_str(cfg.SITES[site])} | {start:%Y-%m-%d %H:%M} | {stop:%Y-%m-%d %H:%M} | {step}"
    if not cfg.REFRESH and os.path.exists(raw) and os.path.exists(key_file):
        with open(key_file) as f:
            if f.read() == key:
                try:
                    load_clean(raw)
                    print(f"[{site}] cached {raw}")
                    return raw
                except ValueError:
                    pass                      # incomplete file -> download again
    JPL_request(start, stop, step, raw, SITE_COORD=coord_str(cfg.SITES[site]))
    with open(key_file, "w") as f:
        f.write(key)
    return raw


ORBIT_STEP = "6 h"
ORBIT_FILES = ("planetary_orbital_positions.csv", "planetary_orbital_elements.csv")


def orbits(when):
    """D. Heliocentric view: JPL Sun-centred vectors + osculating elements for the planner window
    (cached like fetch(), key = window + step), then the alignment snapshot at `when`."""
    import orbital_analyzer as orb            # imported here so a broken orbit module can't stop A-C
    key_file = ORBIT_FILES[0] + ".key"
    key = f"heliocentric | {cfg.START:%Y-%m-%d %H:%M} | {cfg.STOP:%Y-%m-%d %H:%M} | {ORBIT_STEP}"
    cached = (not cfg.REFRESH and all(os.path.exists(f) for f in ORBIT_FILES)
              and os.path.exists(key_file) and open(key_file).read() == key)
    if cached:
        print(f"[orbits] cached {', '.join(ORBIT_FILES)}")
    else:
        orb.JPL_Orbital_Position_request(cfg.START, cfg.STOP, ORBIT_STEP)
        orb.JPL_Orbital_Elements_request(cfg.START, cfg.STOP, ORBIT_STEP)
        with open(key_file, "w") as f:
            f.write(key)
    return orb.alignment(when, ORBIT_FILES[0])


def usable_mask(df):
    """Per-planet rule: elevation >= MIN_ELEV and Sun below that planet's limit."""
    return (df["elev"] >= cfg.MIN_ELEV) & (df["sun_elev"] <= df["name"].map(cfg.SUN_LIMIT))


def load_site(site, raw):
    pl = load_clean(raw)
    sun = load_sun(raw)[["date", "elev"]].rename(columns={"elev": "sun_elev"})
    pl = pl.merge(sun, on="date", validate="many_to_one")
    pl["site"] = site
    pl["instrument"] = cfg.SITE_INSTRUMENT.get(site)
    pl["name"] = pl["planet"].map(PLANETS)
    pl["dt"] = pd.to_datetime(pl["date"], format=FMT)
    pl["ok"] = usable_mask(pl)
    return pl[["site", "instrument", "name", "dt", "date", "azi", "elev", "sun_elev", "apmag", "ok"]]


def load_all():
    frames, scans = [], {}
    for site, coord in cfg.SITES.items():
        if coord is None:
            print(f"Skipping {site}: coordinates not set")
            continue
        try:
            raw = fetch(site, cfg.START, cfg.STOP, cfg.SCAN_STEP, "scan")
            frames.append(load_site(site, raw))
            scans[site] = raw
        except Exception as e:                 # one bad site must not stop the others
            FAILED[site] = f"{type(e).__name__}: {e}"
            print(f"[{site}] FAILED, continuing without it -> {FAILED[site]}")
    if not frames:
        raise RuntimeError(f"every site failed: {FAILED}")
    return pd.concat(frames, ignore_index=True), scans

def validate_config():
    errors = []
    for site, coord in cfg.SITES.items():
        if coord is None:
            continue
        lon, lat, alt = coord
        if not -180 <= lon <= 180:
            errors.append(f"{site}: lon={lon} outside -180..180")
        if not -90 <= lat <= 90:
            errors.append(f"{site}: lat={lat} outside -90..90 (pasted as lat,lon? SITES wants lon,lat)")
    if errors:
        raise ValueError("Config errors:\n  " + "\n  ".join(errors))

    print(f"Window: {cfg.START:%Y-%m-%d %H:%M} -> {cfg.STOP:%Y-%m-%d %H:%M} UTC, step {cfg.SCAN_STEP}")
    for site, coord in cfg.SITES.items():
        print(f"  {site:6s} {coord}  instrument={cfg.SITE_INSTRUMENT.get(site)}")


# ----------------------------------------------------------------- A. conjunction
def _best_visible(sep_csv):
    ts = per_timestamp(sep_csv)
    ok = ts[(ts["min_elev"] >= cfg.MIN_ELEV) & (ts["sun_elev"] <= cfg.MAX_SUN_ALL)]
    return None if ok.empty else ok.loc[ok["S"].idxmin()]


def find_conjunction(site, scan_raw):
    sep = f"{site}_scan_sep.csv"
    angular_separation_analysis(scan_raw, sep, verbose=False)
    row = _best_visible(sep)
    if row is None:
        return None
    t0 = row["dt"].to_pydatetime()

    raw = fetch(site, t0 - PAD, t0 + PAD, cfg.REFINE_STEP, "conj")       # also the optics input
    sep = f"{site}_conj_sep.csv"
    angular_separation_analysis(raw, sep, verbose=False)
    fine = _best_visible(sep)
    if fine is not None:
        row = fine
    return {"site": site, "dt": row["dt"].to_pydatetime(), "date": row["date"],
            "S_deg": round(float(row["S"]), 4), "min_elev": round(float(row["min_elev"]), 1),
            "sun_elev": round(float(row["sun_elev"]), 1), "raw": raw}


# ----------------------------------------------------------------- B. capture plan
def best_capture_plans(ok: pd.DataFrame, planets: list[str], n: int = 3):
    """Shortest windows (any site) containing one usable slot for every planet."""
    ok = (ok.sort_values(["dt", "elev"], ascending=[True, False])
            .drop_duplicates(["name", "dt"]))
    per_planet = {p: g.reset_index(drop=True) for p, g in ok.groupby("name")}
    if set(per_planet) != set(planets):
        print(f"No usable slots for: {sorted(set(planets) - set(per_planet))}")
        return []

    times = {p: pd.DatetimeIndex(g["dt"]) for p, g in per_planet.items()}
    plans = {}
    for t0 in pd.DatetimeIndex(ok["dt"].drop_duplicates()):
        rows = []
        for p in planets:
            i = times[p].searchsorted(t0)
            if i == len(times[p]):
                break
            rows.append(per_planet[p].iloc[i])
        else:
            f = pd.DataFrame(rows)
            plans[f["dt"].min()] = (f["dt"].max() - f["dt"].min(), f)
    return sorted(plans.items(), key=lambda kv: kv[1][0])[:n]


def plan_to_targets(f):
    """Chosen plan -> {site: (planets, first_time, last_time, mid_time)}: what C should check."""
    out = {}
    for site, g in f.groupby("site"):
        lo, hi = g["dt"].min(), g["dt"].max()
        out[site] = (list(g["name"]), lo, hi, (lo + (hi - lo) / 2).round("min"))
    return out


# ----------------------------------------------------------------- C. exposure
def exposure_check(site, planets, peak, raw):
    key = cfg.SITE_INSTRUMENT.get(site)
    if key is None:
        print(f"[{site}] no instrument defined -> optics check skipped")
        return None
    print(f"\n######## {site} / {key} @ {peak} UTC | {', '.join(planets)} ########")
    return optical_geometry(raw, make_instrument(key), peak=peak, min_elev=cfg.MIN_ELEV,
                            only=planets, sky_mag=cfg.SKY_MAG)


# ----------------------------------------------------------------- unified output (Phase 7.5)
def _to_json(o):
    """json.dump calls this for anything it can't write natively. One place = one convention:
    times -> ISO 8601 UTC with 'Z', durations -> seconds, tables -> list of rows, numpy -> float."""
    if isinstance(o, (datetime, pd.Timestamp)):
        return o.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(o, (timedelta, pd.Timedelta)):
        return o.total_seconds()
    if isinstance(o, pd.DataFrame):
        d = o.reset_index().astype(object)
        return d.where(d.notna(), None).to_dict(orient="records")     # NaN -> null (valid JSON)
    if isinstance(o, pd.Series):
        return o.to_dict()
    if hasattr(o, "item"):                     # numpy scalar
        return o.item()
    return str(o)


def optics_summary(r):
    """Keep the parts of optical_geometry's result that matter for the report."""
    if r is None:
        return None
    return {k: r[k] for k in ("reference", "fits_single_frame", "extent_deg", "required_deg",
                              "sampling", "exposure")}


def write_results(results):
    run_dir = Path(__file__).parent / "output" / f"run_{results['run']['started_utc']:%Y%m%dT%H%M%SZ}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_to_json)
    rows = [{"site": s, "status": "failed" if s in FAILED else ("ok" if r else "no window"),
             "best_time_utc": r["dt"] if r else None, "S_deg": r["S_deg"] if r else None,
             "min_elev_deg": r["min_elev"] if r else None, "sun_elev_deg": r["sun_elev"] if r else None,
             "reason": FAILED.get(s)}
            for s, r in {**results["A_conjunction"], **{k: None for k in FAILED}}.items()]
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    print(f"\nResults written to {run_dir}")


# ----------------------------------------------------------------- main
def main():
    results = {
        "run": {"started_utc": datetime.now(timezone.utc).replace(tzinfo=None),
                "window_utc": [cfg.START, cfg.STOP], "scan_step": cfg.SCAN_STEP,
                "units": "angles deg, distances km/AU, times UTC ISO 8601, exposures s",
                "frames": "azi/elev topocentric airless apparent; ra/dec_icrf astrometric J2000"},
        "config": {"sites": cfg.SITES, "site_instrument": cfg.SITE_INSTRUMENT,
                   "min_elev_deg": cfg.MIN_ELEV, "sun_limit_deg": cfg.SUN_LIMIT},
        "failed": FAILED, "A_conjunction": {}, "A_optics": {}, "B_plans": [], "C_optics": {},
        "observations": cfg.OBSERVATIONS,
    }
    validate_config()
    allp, scans = load_all()
    usable = allp[allp["ok"]]
    print("\nUsable samples (%s each):" % cfg.SCAN_STEP)
    print(usable.groupby(["site", "name"]).size().unstack(fill_value=0))

    print("\n===== A. Single-site conjunction (all four visible at once) =====")
    summary = {}
    for site, scan_raw in scans.items():
        c = find_conjunction(site, scan_raw)
        summary[site] = None if c is None else {k: v for k, v in c.items() if k != "raw"}
        if c is None:
            print(f"[{site}] no time with all four >= {cfg.MIN_ELEV} deg and Sun <= {cfg.MAX_SUN_ALL} deg")
            continue
        print(f"[{site}] {c['date']} UTC  S={c['S_deg']} deg  min_elev={c['min_elev']}  "
              f"sun={c['sun_elev']}")
        results["A_optics"][site] = optics_summary(exposure_check(site, ALL_NAMES, c["dt"], c["raw"]))

    print("\n===== B. Multi-site capture plans (shortest total span) =====")
    plans = best_capture_plans(usable, list(cfg.SUN_LIMIT))
    for start, (span, f) in plans:
        print(f"\nSpan {span} starting {start} UTC")
        print(f[["name", "site", "instrument", "date", "azi", "elev", "sun_elev"]]
              .round(1).to_string(index=False))
        results["B_plans"].append({"start_utc": start, "span_s": span,
                                   "slots": f[["name", "site", "instrument", "dt", "azi", "elev",
                                               "sun_elev"]]})

    if plans:
        print("\n===== C. Optics for the best capture plan =====")
        for site, (planets, lo, hi, mid) in plan_to_targets(plans[0][1][1]).items():
            raw = fetch(site, (lo - PAD).to_pydatetime(), (hi + PAD).to_pydatetime(),
                        cfg.REFINE_STEP, "capture")
            results["C_optics"][site] = optics_summary(
                exposure_check(site, planets, mid.to_pydatetime(), raw))

    print("\n===== SUMMARY (UTC) =====")
    for site, r in summary.items():
        print(site, r)
    for site, why in FAILED.items():
        print(site, "FAILED:", why)

    print("\n===== D. Heliocentric view (orbits + alignment) =====")
    found = [r["dt"] for s, r in summary.items() if r]
    when = (summary.get("GZ05") or {}).get("dt") or (min(found) if found else cfg.START)
    try:
        d = orbits(when)
        results["D_orbits"] = d
        print(d["bodies"][["name", "r_au", "helio_lon", "geo_lon", "elong"]].round(1).to_string(index=False))
        print(f"at {d['when_utc']:%Y-%m-%d %H:%M} UTC: seen from Earth the four planets span "
              f"{d['geo_spread_deg']} deg of ecliptic longitude; around the Sun they span "
              f"{d['helio_spread_deg']} deg")
    except Exception as e:                    # D is extra: report it, keep the A-C results
        results["D_orbits"] = {"error": f"{type(e).__name__}: {e}"}
        print("[orbits] FAILED:", results["D_orbits"]["error"])

    results["A_conjunction"] = summary
    write_results(results)


if __name__ == "__main__":
    main()