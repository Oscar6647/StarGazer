"""
skynet_site_survey.py -- which Skynet telescopes (that you have access to) can reach each planet, when?

For every optical telescope from GET /v1/me/observing-access/telescopes (real lat/lon/elevation,
mount minimum altitude, local horizon), fetch a Horizons ephemeris and count, per night, the minutes
each planet is above that telescope's limit with the Sun below Skynet's safety limit (-10 deg).

    python skynet_site_survey.py                      # cfg.START .. cfg.STOP, 10-min steps
    python skynet_site_survey.py --stop 2026-12-05    # extend the date range

Output: skynet_survey/<slug>_raw.csv (ephemerides) and skynet_site_survey.csv (minutes per night).
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

import capture_planner as cfg
from emphimeris_analyzer import FMT, JPL_request, PLANETS, load_clean, load_sun
from mosaic_planner import horizon_at
from skynet_client import SkynetClient

STEP_MIN = 10


def telescope_sites(client):
    out = []
    for t in client.get("/me/observing-access/telescopes"):
        if not any(i.get("instrumentType") == "optical_imager" for i in t.get("instruments") or []):
            continue
        out.append(dict(name=t["name"], slug=t["slug"], lat=t["latitudeDeg"], lon=t["longitudeDeg"],
                        elev_km=t["elevationM"] / 1000,
                        min_alt=max(cfg.MIN_ELEV, (t.get("mount") or {}).get("minAltitudeDeg") or 0),
                        horizon=t.get("localHorizon") or None))
    return out


def survey(site, start, stop, outdir):
    raw = os.path.join(outdir, f"{site['slug']}_raw.csv")
    if not os.path.exists(raw):
        JPL_request(start, stop, f"{STEP_MIN}m", raw,
                    SITE_COORD=f"{site['lon']},{site['lat']},{site['elev_km']}")
    df = load_clean(raw)
    sun = load_sun(raw)[["date", "elev"]].rename(columns={"elev": "sun_elev"})
    df = df.merge(sun, on="date")
    df["name"] = df["planet"].map(PLANETS)
    df["dt"] = pd.to_datetime(df["date"], format=FMT)
    limits = dict(min_alt=site["min_alt"], horizon=site["horizon"])
    df["ok"] = (df["elev"] >= horizon_at(df["azi"].values, limits)) & \
               (df["sun_elev"] <= cfg.SKYNET_MAX_SUN_DEG)
    # "night" = local date of the evening the night starts (shift by longitude, then -12 h)
    df["night"] = (df["dt"] + pd.to_timedelta(site["lon"] / 15, unit="h")
                   - pd.Timedelta(hours=12)).dt.date
    ok = df[df["ok"]]
    mins = ok.groupby(["night", "name"]).size().unstack(fill_value=0) * STEP_MIN
    peak = ok.groupby(["night", "name"])["elev"].max().unstack()
    res = mins.reindex(columns=[PLANETS[k] for k in PLANETS if k != "10"], fill_value=0)
    res = res.add_suffix("_min").join(peak.add_suffix("_maxel").round(1))
    res.insert(0, "telescope", site["name"])
    res.insert(1, "lat", round(site["lat"], 2))
    res.insert(2, "min_alt", site["min_alt"])
    return res.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=cfg.START.strftime("%Y-%m-%d"))
    ap.add_argument("--stop", default=cfg.STOP.strftime("%Y-%m-%d"))
    a = ap.parse_args()
    start, stop = (datetime.strptime(a.start, "%Y-%m-%d"), datetime.strptime(a.stop, "%Y-%m-%d"))
    outdir = f"skynet_survey_{a.start}_{a.stop}"
    os.makedirs(outdir, exist_ok=True)

    sites = telescope_sites(SkynetClient())
    print(f"{len(sites)} optical telescopes in your Skynet access")
    rows = []
    for s in sites:
        print(f"\n--- {s['name']} (lat {s['lat']:.1f}, min alt {s['min_alt']} deg)")
        try:
            rows.append(survey(s, start, stop, outdir))
        except Exception as e:                         # one bad site must not stop the survey
            print(f"    skipped: {e}")
    allr = pd.concat(rows, ignore_index=True)
    allr.to_csv("skynet_site_survey.csv", index=False)

    tot = allr.groupby(["telescope", "lat", "min_alt"])[
        ["Mercury_min", "Venus_min", "Mars_min", "Jupiter_min"]].sum()
    tot["nights_all4"] = allr.assign(all4=(allr[["Mercury_min", "Venus_min", "Mars_min",
                                                 "Jupiter_min"]] > 0).all(axis=1)) \
        .groupby(["telescope", "lat", "min_alt"])["all4"].sum()
    print(f"\n===== usable minutes {a.start} .. {a.stop} (Sun <= {cfg.SKYNET_MAX_SUN_DEG}, "
          f"planet >= mount min / local horizon) =====")
    print(tot.sort_values(["Mercury_min", "Venus_min"], ascending=False).to_string())
    print("\nPer-night detail -> skynet_site_survey.csv")


if __name__ == "__main__":
    main()
