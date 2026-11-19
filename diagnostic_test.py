from datetime import datetime
import pandas as pd
from emphimeris_analyzer import *
import capture_planner as cfg

NAME  = "GZ04"
COORD = ",".join(str(v) for v in cfg.SITES[NAME])     # from capture_planner
MIN_ELEV, MAX_SUN = cfg.MIN_ELEV, cfg.MAX_SUN_ALL

raw, out = f"{NAME}_diag_raw.csv", f"{NAME}_diag_sep.csv"
JPL_request(cfg.START, cfg.STOP, "1h", raw, SITE_COORD=COORD)
angular_separation_analysis(raw, out)

# One row per timestamp: S, min_elev, sun_elev + each planet's own elevation
ts = per_timestamp(out)
df = load_clean(raw)
elev = df.pivot(index="date", columns="planet", values="elev").rename(columns=PLANETS)
ts = ts.merge(elev, left_on="date", right_index=True)
names = list(elev.columns)
ts["lowest"] = ts[names].idxmin(axis=1)          # which planet limits min_elev

up   = ts["min_elev"] >= MIN_ELEV                # all four above the horizon margin
dark = ts["sun_elev"] <= MAX_SUN                 # sky dark enough

# 1) How many timestamps pass each condition
print(f"\n[{NAME}] {len(ts)} hourly timestamps")
print(f"all 4 planets >= {MIN_ELEV} deg : {up.sum()}")
print(f"Sun <= {MAX_SUN} deg            : {dark.sum()}")
print(f"both                            : {(up & dark).sum()}")

# 2) Best S under each condition set
cols = ["date", "S", "min_elev", "lowest", "sun_elev"] + names
for label, mask in [("No filter (pure geometry)", slice(None)),
                    ("Elevation only", up),
                    ("Darkness only", dark),
                    ("Both", up & dark)]:
    sub = ts[mask]
    print(f"\n--- {label} ---")
    if sub.empty:
        print("no timestamps")
    else:
        print(sub.loc[[sub["S"].idxmin()], cols].round(2).to_string(index=False))

# 3) During dark hours only: how often is each planet above MIN_ELEV?
d = ts[dark]
if d.empty:
    print("\nSun never reaches the darkness threshold in this range.")
else:
    print(f"\nDuring dark hours ({len(d)} timestamps), share with each planet >= {MIN_ELEV} deg:")
    print((d[names] >= MIN_ELEV).mean().round(2).to_string())
    print("\nPlanet most often the lowest during dark hours:")
    print(d["lowest"].value_counts().to_string())