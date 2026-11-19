try:                                   # Windows corporate/venv cert store; optional elsewhere
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import requests
import pandas as pd
import math
from datetime import datetime, timedelta
from itertools import combinations

"""*******************************************************************************************************************************************************
 Date__(UT)__HR:MN, , , R.A._(ICRF), DEC_(ICRF), R.A._(a-app), DEC_(a-app), Azi_(a-app), Elev_(a-app),    APmag,  S-brt,             delta,     deldot,
*******************************************************************************************************************************************************"""

cols = ["planet", "date", "flag1", "flag2",
        "ra_icrf", "dec_icrf", "ra_app", "dec_app",
        "azi", "elev", "apmag", "sbrt", "delta", "deldot", "extra"]
URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
JST_OFFSET = timedelta(hours=9)
TIME_FMT = "%Y-%m-%d %H:%M"
PLANETS = {"10": "Sun", "199": "Mercury", "299": "Venus", "499": "Mars", "599": "Jupiter"}
N_TARGETS = len(PLANETS) - 1          # everything except the Sun

BASE_PARAMS = {
    "format": "json",
    "OBJ_DATA": "YES",
    "MAKE_EPHEM": "YES",
    "EPHEM_TYPE": "OBSERVER",
    "CENTER": "coord@399",
    "COORD_TYPE": "GEODETIC",
    "SITE_COORD": "130.87521,33.88342,0",
    "QUANTITIES": "1,2,4,9,20",
    "CSV_FORMAT": "YES",
    "ANG_FORMAT": "DEG",          # ra_app/dec_app are DEGREES (instrument_analyzer.RA_IN_HOURS=False)
}
FMT = "%Y-%b-%d %H:%M"


def local_to_utc_gz1(local_dt):
    """JST datetime -> UTC datetime (what Horizons expects)."""
    return local_dt - JST_OFFSET


def build_params(body, start_utc, stop_utc, step, **overrides):
    """Extra keywords override or add keys."""
    params = dict(BASE_PARAMS)
    params.update({
        "COMMAND": body,
        "START_TIME": start_utc.strftime(TIME_FMT),
        "STOP_TIME": stop_utc.strftime(TIME_FMT),
        "STEP_SIZE": step,
    })
    params.update(overrides)
    return params


def fetch_rows(params):
    """One Horizons request -> list of raw CSV lines between $$SOE and $$EOE."""
    response = requests.get(URL, params=params, timeout=30)
    response.raise_for_status()
    result = response.json()["result"]
    if "$$SOE" not in result:
        raise ValueError(f"No ephemeris in response:\n{result[:500]}")
    start = result.index("$$SOE") + len("$$SOE")
    end = result.index("$$EOE")
    return result[start:end].strip().splitlines()


def JPL_request(start_utc, stop_utc, step, outfile, bodies=PLANETS, **overrides):
    lines = []
    for body in bodies:
        try:
            rows = fetch_rows(build_params(body, start_utc, stop_utc, step, **overrides))
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"Skipping {body}: {e}")
            continue
        lines += [f"{body},{row}" for row in rows]
        print(f"{body}: {len(rows)} rows")
    if not lines:
        raise RuntimeError(f"Horizons returned nothing for {outfile}")

    csv_data = "\n".join(lines) + "\n"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(csv_data)
    return csv_data


def angular_separation_haversine(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    a = (math.sin((dec2 - dec1) / 2) ** 2 +
         math.cos(dec1) * math.cos(dec2) * math.sin((ra2 - ra1) / 2) ** 2)
    return math.degrees(2 * math.asin(math.sqrt(a)))


def load_clean(csv_base, sun_id="10"):
    """Planets only (no Sun). Timestamps missing any of the four planets are dropped (with a note),
    matching what instrument_analyzer.load_planets does, so every stage sees the same rows."""
    df = pd.read_csv(csv_base, header=None, names=cols,
                     skipinitialspace=True, dtype={"planet": str})
    df["planet"] = df["planet"].str.strip()
    df["date"] = df["date"].str.strip()
    df = df.dropna(subset=["ra_app", "elev"])
    df = df[df["planet"] != sun_id].drop_duplicates(["planet", "date"]).reset_index(drop=True)
    incomplete = df.groupby("date")["planet"].transform("size") != N_TARGETS
    if incomplete.any():
        print(f"[load_clean] dropping {df.loc[incomplete, 'date'].nunique()} incomplete timestamp(s)")
        df = df[~incomplete].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{csv_base}: no timestamp has all {N_TARGETS} planets")
    return df


def add_sun_elev(df, sun_df):
    assert sun_df["date"].is_unique, "Sun has duplicate dates"
    sun = sun_df[["date", "elev"]].rename(columns={"elev": "sun_elev"})
    out = df.merge(sun, on="date", how="left", validate="many_to_one")
    assert out["sun_elev"].notna().all(), "Some dates have no Sun elevation"
    return out


def load_sun(csv_sun, sun_id="10"):
    sun = pd.read_csv(csv_sun, header=None, names=cols, skipinitialspace=True, dtype={"planet": str})
    sun["planet"] = sun["planet"].str.strip()
    sun["date"] = sun["date"].str.strip()
    sun = sun.drop_duplicates(["planet", "date"]).reset_index(drop=True)
    sun_df = sun[sun["planet"] == sun_id].reset_index(drop=True)
    assert sun_df["date"].is_unique, "Sun has duplicate dates"
    return sun_df


def angular_separation_analysis(csv_base, csv_output, verbose=True):
    df = load_clean(csv_base)
    sun_df = load_sun(csv_base)
    df = add_sun_elev(df, sun_df)
    results = []
    for date in df["date"].unique():
        day = df[df["date"] == date].set_index("planet")
        for p1, p2 in combinations(day.index, 2):
            a, b = day.loc[p1], day.loc[p2]
            sep = angular_separation_haversine(a["ra_app"], a["dec_app"],
                                               b["ra_app"], b["dec_app"])
            results.append({"date": date, "pair": f"{p1}-{p2}", "sep_deg": sep,
                            "azi1": a["azi"], "azi2": b["azi"],
                            "elev1": a["elev"], "elev2": b["elev"],
                            "apmag1": a["apmag"], "apmag2": b["apmag"]})

    res = pd.DataFrame(results)
    sun = sun_df[["date", "elev"]].rename(columns={"elev": "sun_elev"})
    res = res.merge(sun, on="date")
    res.to_csv(csv_output, index=False)
    if verbose:
        best = res.loc[res.groupby("pair")["sep_deg"].idxmin()]
        print("Closest_Approach between each pair\n", best.to_string())
        spread = res.groupby("date")["sep_deg"].max()
        print("Smallest S(t) timestamps:\n", spread.nsmallest(10).round(4))
        print("Tightest grouping:", spread.idxmin(), round(spread.min(), 4), "deg")
    return res


def per_timestamp(csv_output):
    res = pd.read_csv(csv_output)
    g = res.groupby("date")
    ts = pd.DataFrame({
        "S":      g["sep_deg"].max(),    # S(t), worst pair
        "S_mean": g["sep_deg"].mean(),   # average of the six
        "S_min":  g["sep_deg"].min(),    # closest pair
        "min_elev": g[["elev1", "elev2"]].min().min(axis=1),
        "sun_elev": g["sun_elev"].first(),
    })
    ts = ts.join(res.pivot(index="date", columns="pair", values="sep_deg"))  # e.g. "199-299"
    ts = ts.reset_index()
    ts["dt"] = pd.to_datetime(ts["date"], format=FMT)
    ts["night"] = (ts["dt"] - pd.Timedelta(hours=12)).dt.date
    return ts.sort_values("dt").reset_index(drop=True)


def best_nights(ts, metric="S", min_elev=0, max_sun=-6, n=5):
    ok = ts[(ts["min_elev"] >= min_elev) & (ts["sun_elev"] <= max_sun)]
    return ok.groupby("night")[metric].agg(best="min", mean="mean", n="count").nsmallest(n, "best")


def best_time_window(ts, night, metric, tol, min_elev, max_sun, max_gap_h, mode="relative"):
    d = ts[(ts["night"] == night) &
           (ts["min_elev"] >= min_elev) & (ts["sun_elev"] <= max_sun)].copy()
    if d.empty:
        return d, pd.DataFrame()

    # relative: within tol deg of the night's best; absolute: metric <= tol
    limit = d[metric].min() + tol if mode == "relative" else tol
    d["good"] = d[metric] <= limit

    gap = d["dt"].diff() > pd.Timedelta(hours=max_gap_h)       # don't join rows across daylight gaps
    d["block"] = ((d["good"] != d["good"].shift()) | gap).cumsum()

    blocks = (d[d["good"]].groupby("block")
              .agg(start=("dt", "min"), end=("dt", "max"),
                   best=(metric, "min"), n=(metric, "size")))
    blocks.insert(0, "metric", metric)
    return d, blocks