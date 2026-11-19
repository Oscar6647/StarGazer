#!/usr/bin/env python3
"""
IRF Kiruna all-sky image projector (reuses the emphimeris_analyzer Horizons CSV)
--------------------------------------------------------------------

Reads the raw Horizons CSV that emphimeris_analyzer.JPL_request() already saved
(columns = emphimeris_analyzer.cols) and projects the planets' az/elev onto an all-sky
image with an equidistant, zenith-centred fisheye model. No new JPL request.

Image convention: N=top, E=left, S=bottom, W=right. Azimuth clockwise from N.

Model:
    z   = 90 deg - elevation                      (zenith distance)
    rho = R * z / 90 deg                          (pixel radius from centre)

Distortion of the equidistant model (what this script reports):
    radial scale      k_r = R / 90                  px/deg  (constant)
    tangential scale  k_t = (R / 90) * z / sin z    px/deg  (z in radians)
    anisotropy        k_t / k_r = z / sin z         1.00 at zenith, 1.571 at horizon
So a circle on the sky becomes an ellipse stretched along the almucantar,
by up to 57 % at the horizon. Pairwise: the true great-circle separation is
compared with (pixel distance) / k_r, which is what you would "read" off the
image if you assumed a constant plate scale.

Usage:
    python irf_fisheye_projector.py image.jpg ephem.csv --date "2026-Nov-13 18:00"
    python irf_fisheye_projector.py image.jpg ephem.csv --best   # tightest grouping S(t)

IMPORTANT: the ephemeris must have been fetched for the CAMERA's site.
For IRF Kiruna pass SITE_COORD as an override to JPL_request, e.g.
    JPL_request(t0, t1, "10m", "kiruna.csv", SITE_COORD=KIRUNA_SITE)
or simply reuse GZ04_conj_raw.csv / GZ04_scan_raw.csv written by the mission planner.
Same projection as instrument_analyzer.fisheye_project for IRF_Kiruna_ASC.
"""

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import capture_planner as cfg
from emphimeris_analyzer import (FMT, PLANETS, add_sun_elev, angular_separation_analysis,
                       angular_separation_haversine, load_clean, load_sun,
                       per_timestamp)

# IRF Kiruna = site GZ04 in capture_planner: E-lon, lat, altitude km (Horizons SITE_COORD).
KIRUNA_SITE = ",".join(str(v) for v in cfg.SITES["GZ04"])


# --------------------------------------------------------------------------- geometry

def detect_sky_circle(image: np.ndarray,
                      min_radius_fraction: float = 0.40,
                      max_radius_fraction: float = 0.51):
    """Detect the dominant circular sky boundary with HoughCircles."""
    gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 9)
    max_dim = min(gray.shape)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max_dim * 0.5,
        param1=100, param2=50,
        minRadius=int(max_dim * min_radius_fraction),
        maxRadius=int(max_dim * max_radius_fraction),
    )
    if circles is None:
        raise RuntimeError("Could not detect the sky circle. "
                           "Supply --cx, --cy and --radius manually.")
    x, y, r = max(circles[0], key=lambda c: c[2])
    return float(x), float(y), float(r)


def project_altaz(az_deg, elev_deg, cx, cy, radius):
    """Equidistant zenith-centred fisheye. N top, E left."""
    if not -90.0 <= elev_deg <= 90.0:
        raise ValueError(f"Invalid elevation: {elev_deg}")
    rho = radius * (90.0 - elev_deg) / 90.0
    az = math.radians(az_deg % 360.0)
    return cx - rho * math.sin(az), cy - rho * math.cos(az), rho


def local_distortion(elev_deg, radius):
    """Radial / tangential plate scale (px/deg) and their ratio at this elevation."""
    k_r = radius / 90.0
    z = math.radians(90.0 - elev_deg)
    ratio = 1.0 if z < 1e-9 else z / math.sin(z)
    return k_r, k_r * ratio, ratio


# --------------------------------------------------------------------------- ephemeris

def load_ephemeris(csv_base):
    """Planets (all 4 present per timestamp) + sun_elev column, via emphimeris_analyzer."""
    return add_sun_elev(load_clean(csv_base), load_sun(csv_base))


def pick_date(df, csv_base, date=None, min_elev=cfg.MIN_ELEV, max_sun=cfg.MAX_SUN_ALL):
    """Exact/nearest Horizons timestamp, or the tightest-grouping one (min S(t))."""
    if date:
        wanted = pd.to_datetime(date, format=FMT)
        stamps = pd.to_datetime(pd.Series(df["date"].unique()), format=FMT)
        best = stamps.iloc[(stamps - wanted).abs().argmin()]
        if best != wanted:
            print(f"Note: {date} not in ephemeris; using nearest {best.strftime(FMT)}")
        return best.strftime(FMT)

    pairs_csv = Path(csv_base).with_name(Path(csv_base).stem + "_pairs.csv")
    angular_separation_analysis(csv_base, pairs_csv, verbose=False)
    ts = per_timestamp(pairs_csv)
    ok = ts[(ts["min_elev"] >= min_elev) & (ts["sun_elev"] <= max_sun)]
    if ok.empty:
        raise ValueError(f"No timestamp with all planets >= {min_elev} deg "
                         f"and Sun <= {max_sun} deg")
    row = ok.loc[ok["S"].idxmin()]
    print(f"Best grouping: {row['date']}  S(t)={row['S']:.3f} deg")
    return row["date"]


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("image", type=Path)
    ap.add_argument("ephemeris_csv", type=Path, help="raw CSV written by emphimeris_analyzer.JPL_request")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help=f'UT timestamp in Horizons format, e.g. "2026-Nov-13 18:00"')
    g.add_argument("--best", action="store_true", help="use the tightest-grouping timestamp")
    ap.add_argument("--min-elev", type=float, default=cfg.MIN_ELEV)
    ap.add_argument("--max-sun", type=float, default=cfg.MAX_SUN_ALL)
    ap.add_argument("--output", type=Path, default=Path("irf_planet_overlay.jpg"))
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--radius", type=float)
    ap.add_argument("--margin-px", type=float, default=5.0)
    args = ap.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    manual = (args.cx, args.cy, args.radius)
    if any(v is not None for v in manual):
        if not all(v is not None for v in manual):
            ap.error("Specify all three overrides: --cx, --cy and --radius.")
        cx, cy, radius = manual
        source = "manual"
    else:
        cx, cy, radius = detect_sky_circle(image)
        source = "Hough"

    df = load_ephemeris(args.ephemeris_csv)
    date = pick_date(df, args.ephemeris_csv, args.date, args.min_elev, args.max_sun)
    day = df[df["date"] == date].set_index("planet")
    sun_elev = float(day["sun_elev"].iloc[0])

    overlay = image.copy()
    cv2.circle(overlay, (round(cx), round(cy)), round(radius), (0, 220, 255), 2)
    cv2.drawMarker(overlay, (round(cx), round(cy)), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)

    # --- per-planet projection + local distortion
    report = {}
    for pid, p in day.iterrows():
        name = PLANETS.get(pid, pid)
        az, el = float(p["azi"]), float(p["elev"])
        x, y, rho = project_altaz(az, el, cx, cy, radius)
        k_r, k_t, ratio = local_distortion(el, radius)
        fits = el >= 0.0 and rho + args.margin_px <= radius
        report[pid] = dict(name=name, az=az, el=el, x=x, y=y, rho=rho,
                           k_r=k_r, k_t=k_t, ratio=ratio, fits=fits)

        color = (40, 230, 40) if fits else (40, 40, 255)
        cv2.drawMarker(overlay, (round(x), round(y)), color, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(overlay, name, (round(x) + 8, round(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    cv2.putText(overlay, f"UT {date}  Sun {sun_elev:+.1f} deg", (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(args.output), overlay):
        raise RuntimeError(f"Could not write {args.output}")

    # --- pairwise: true separation vs. what the image shows at constant scale
    k_r = radius / 90.0
    pairs = []
    for a, b in combinations(day.index, 2):
        pa, pb, ra, rb = day.loc[a], day.loc[b], report[a], report[b]
        true_sep = angular_separation_haversine(pa["azi"], pa["elev"], pb["azi"], pb["elev"])
        radec_sep = angular_separation_haversine(pa["ra_app"], pa["dec_app"],
                                                 pb["ra_app"], pb["dec_app"])
        px = math.hypot(ra["x"] - rb["x"], ra["y"] - rb["y"])
        img_sep = px / k_r
        pairs.append((f"{ra['name']}-{rb['name']}", radec_sep, true_sep, px, img_sep,
                      img_sep - true_sep,
                      100 * (img_sep - true_sep) / true_sep if true_sep else float("nan")))

    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]} px)")
    print(f"Sky circle ({source}): centre=({cx:.1f}, {cy:.1f}) R={radius:.1f} px "
          f"-> radial scale {k_r:.3f} px/deg")
    print(f"Timestamp: {date} UT   Sun elev {sun_elev:+.2f} deg")
    print(f"Overlay saved: {args.output}\n")

    print(f"{'Planet':<9}{'Az':>8}{'El':>7}{'X px':>9}{'Y px':>9}"
          f"{'k_r':>8}{'k_t':>8}{'k_t/k_r':>9}  Inside")
    for r in report.values():
        print(f"{r['name']:<9}{r['az']:>8.2f}{r['el']:>7.2f}{r['x']:>9.1f}{r['y']:>9.1f}"
              f"{r['k_r']:>8.3f}{r['k_t']:>8.3f}{r['ratio']:>9.3f}  {r['fits']}")

    print(f"\n{'Pair':<17}{'RA/Dec sep':>11}{'AltAz sep':>10}{'px':>8}"
          f"{'img sep':>9}{'err deg':>9}{'err %':>8}")
    for name, rd, t, px, im, err, pct in pairs:
        print(f"{name:<17}{rd:>11.3f}{t:>10.3f}{px:>8.1f}{im:>9.3f}{err:>+9.3f}{pct:>+8.1f}")

    print("\nk_r/k_t: radial/tangential px per degree. 'img sep' = pixel distance / k_r.")
    print("AltAz vs RA/Dec sep differ slightly because az/el include refraction.")
    print("First-order model only; calibrate against catalogue stars for precision.")


if __name__ == "__main__":
    main()
