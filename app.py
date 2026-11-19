"""
StarGazer · Mission Control  --  web front end for the test.py pipeline.

    streamlit run app.py

Everything shown here comes from the pipeline: the "Run pipeline" button calls test.main(),
which writes output/run_<UTC>/results.json; every tab reads that file (or the ephemeris CSVs
the pipeline cached). No Skynet token is needed or used here.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                      # the pipeline reads/writes its CSVs in this folder
sys.path.insert(0, str(HERE))

import capture_planner as cfg       # noqa: E402
import SPS as pipeline             # noqa: E402
from emphimeris_analyzer import FMT, PLANETS, load_clean, load_sun   # noqa: E402

st.set_page_config(page_title="StarGazer · Mission Control", page_icon=str(HERE / "assets" / "Logo.png"), layout="wide")
st.logo(image=str(HERE / "assets" / "AppLogo.png"),)


# ---- visual tokens (dark surface; categorical slots 1-3 validated all-pairs on #1a1a19) ----
SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8a8984", "#383835"
SERIES = ["#3987e5", "#d95926", "#199e70"]
STATUS_ICON = {"delivered": "✅", "pending": "⏳", "failed": "❌"}
SITE_LABEL = {"GZ01": "Kitakyushu, JP", "GZ02": "MDRS, Utah (Skynet)", "GZ03": "Hampden-Sydney, VA (Skynet)",
              "GZ04": "Kiruna all-sky, SE", "GZ05": "Guadalupe, Monterrey MX", "GZ05B": "Monterrey MX (EOS R)",
              "GZ05C": "Torreón MX", "GZ06": "Cerro Tololo, CL (Skynet)"}



def style(chart):
    return (chart.configure(background=SURFACE)
            .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor=INK2,
                            titleColor=INK2, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelColor=INK2, titleColor=INK2, orient="top")
            .configure_view(strokeWidth=0))


def show(chart):
    st.altair_chart(style(chart), use_container_width=True, theme=None)


# ------------------------------------------------------------------------- data access
def run_files():
    return sorted((HERE / "output").glob("run_*/results.json"), reverse=True)


@st.cache_data
def load_run(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_pipeline() -> str:
    """Run test.main() exactly as the terminal would, capturing what it prints."""
    pipeline.FAILED.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pipeline.main()
    return buf.getvalue()


def utc(s):
    return None if not s else datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


@st.cache_resource
def _tf():
    from timezonefinder import TimezoneFinder
    return TimezoneFinder()


def zone_at(lat, lon):
    """IANA time zone for a spot (offline lookup), or None at sea. Handles daylight saving."""
    try:
        return _tf().timezone_at(lat=lat, lng=lon)
    except Exception:
        return None


def to_local(t, lat, lon):
    """UTC (naive) -> (local datetime, 'UTC-6 · America/Monterrey'), or (None, '') if unknown."""
    from zoneinfo import ZoneInfo
    z = zone_at(lat, lon)
    if t is None or z is None:
        return None, ""
    lt = t.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(z))
    h = lt.utcoffset().total_seconds() / 3600
    return lt, f"UTC{h:+g} · {z}"


def local_str(site, t):
    c = cfg.SITES.get(site)
    lt, zone = to_local(t, c[1], c[0]) if c else (None, "")
    return f"{lt:%H:%M} local ({zone.split(' · ')[0]})" if lt else ""


@st.cache_data
def sky_table(site: str):
    raw = HERE / f"{site}_conj_raw.csv"
    if not raw.exists():
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        pl = load_clean(str(raw))
        sun = load_sun(str(raw))[["date", "elev"]].rename(columns={"elev": "sun_elev"})
    pl = pl.merge(sun, on="date")
    pl["name"] = pl["planet"].map(PLANETS)
    pl["dt"] = pd.to_datetime(pl["date"], format=FMT)
    for c in ("azi", "elev", "apmag", "sun_elev"):
        pl[c] = pd.to_numeric(pl[c], errors="coerce")
    return pl[["dt", "name", "azi", "elev", "apmag", "sun_elev"]]


@st.cache_data(show_spinner="Fetching dawn ephemerides from JPL Horizons…")
def dawn_mercury(sites: tuple, day: str):
    """Mercury altitude vs Sun altitude during one dawn, per site (5-min steps, cached by the pipeline)."""
    start = datetime.strptime(day, "%Y-%m-%d")
    out = []
    for site in sites:
        with contextlib.redirect_stdout(io.StringIO()):
            raw = pipeline.fetch(site, start, start + timedelta(days=1), "5m", "dawn")
            pl = load_clean(raw)
            sun = load_sun(raw)[["date", "elev"]].rename(columns={"elev": "sun_elev"})
        m = pl[pl["planet"] == "199"].merge(sun, on="date")
        m["dt"] = pd.to_datetime(m["date"], format=FMT)
        m = m.sort_values("dt")
        rising = m["sun_elev"].diff() > 0
        m = m[rising & m["sun_elev"].between(-18, 0)]
        m["site"] = f"{site} · {SITE_LABEL.get(site, site)}"
        out.append(m[["site", "dt", "sun_elev", "elev"]])
    return pd.concat(out, ignore_index=True)


@st.cache_data(show_spinner="Asking JPL Horizons for your sky (≈20–40 s the first time)…", max_entries=200)
def your_conjunction(lat: float, lon: float, alt_km: float):
    """Pipeline step A for a visitor's coordinates: registered as a temporary site, files kept in
    user_sites/ (cached by coordinates like every other site). Returns (site id, conjunction or None);
    the site id is the file prefix, so sky_table / optics read <id>_conj_raw.csv like a mission site."""
    (HERE / "user_sites").mkdir(exist_ok=True)
    site = "user_sites/u" + f"{lat:+08.3f}_{lon:+09.3f}".replace("+", "p").replace("-", "m")
    cfg.SITES[site] = (lon, lat, alt_km)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            raw = pipeline.fetch(site, cfg.START, cfg.STOP, cfg.SCAN_STEP, "scan")
            c = pipeline.find_conjunction(site, raw)
    finally:
        cfg.SITES.pop(site, None)
    return site, (None if c is None else {k: v for k, v in c.items() if k != "raw"})


def make_loc(site_id, label, short, lat, lon, alt_km, conj, mission):
    """One 'active location' record that every tab reads."""
    dt = None if not conj else (utc(conj["dt"]) if isinstance(conj["dt"], str) else conj["dt"])
    return dict(id=site_id, label=label, short=short, lat=lat, lon=lon, alt_km=alt_km, dt=dt,
                conj=conj, mission=mission, has_sky=bool(dt) and (HERE / f"{site_id}_conj_raw.csv").exists())


# ------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🔭 StarGazer")
    st.caption("Four-planet grouping · November 2026\nMercury · Venus · Mars · Jupiter")
    if st.button("▶ Run pipeline", type="primary", use_container_width=True,
                 help="Runs test.py: validate → ephemerides → A conjunction → B plans → C optics → results.json"):
        with st.spinner("Running the pipeline (cached ephemerides ≈ 20 s, fresh download ≈ 2 min)…"):
            try:
                st.session_state["log"] = run_pipeline()
                st.session_state["log_ok"] = True
            except Exception as e:      # show the error in the UI instead of a blank page
                st.session_state["log"] = f"{type(e).__name__}: {e}"
                st.session_state["log_ok"] = False
        st.cache_data.clear()
        st.rerun()

    files = run_files()
    if not files:
        st.warning("No pipeline run yet. Press **Run pipeline**.")
        st.stop()
    choice = st.selectbox("Pipeline run", files, format_func=lambda p: p.parent.name.replace("run_", "run · "))
    if st.session_state.get("log_ok") is False:
        st.error("Last run failed — see the Pipeline tab.")
    R = load_run(str(choice))
    A = R["A_conjunction"]

    # ---- the one control that drives every tab
    st.divider()
    st.markdown("**📍 Observing location**")
    OWN = "✏️ My own place…"
    loc_opts = [s for s, c in cfg.SITES.items() if c] + [OWN]
    pick = st.selectbox("Observing location", loc_opts, key="loc_pick", label_visibility="collapsed",
                        index=loc_opts.index("GZ05") if "GZ05" in loc_opts else 0,
                        format_func=lambda s: s if s == OWN else f"{s} · {SITE_LABEL.get(s, s)}")
    LOC = None
    if pick == OWN:
        with st.form("own_place"):
            o_place = st.text_input("Place", "My backyard")
            o_coords = st.text_input("Latitude, longitude", "25.6866, -100.3161",
                                     help="Paste straight from Google Maps: right-click a spot → click the numbers.")
            o_elev = st.number_input("Elevation (m)", -400, 6000, 500, step=50)
            if st.form_submit_button("Use this place", type="primary", use_container_width=True):
                import re
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?", o_coords)
                if len(nums) >= 2 and -90 <= float(nums[0]) <= 90 and -180 <= float(nums[1]) <= 180:
                    st.session_state["own"] = (o_place, float(nums[0]), float(nums[1]), o_elev / 1000)
                else:
                    st.error("Use “latitude, longitude” in degrees, e.g. 25.6866, -100.3161.")
        if "own" in st.session_state:
            o_place, o_lat, o_lon, o_alt = st.session_state["own"]
            try:
                sid, conj = your_conjunction(o_lat, o_lon, o_alt)
                LOC = make_loc(sid, o_place, o_place, o_lat, o_lon, o_alt, conj, False)
            except Exception as e:
                st.error(f"JPL Horizons didn't answer ({type(e).__name__}). Try again in a minute.")
        else:
            st.info("Type a place and press **Use this place**. Until then the dashboard shows Monterrey.")
        st.caption("Your coordinates are sent to JPL Horizons only.")
        if LOC is None:
            pick = "GZ05" if "GZ05" in cfg.SITES else loc_opts[0]
    if LOC is None:
        lon_s, lat_s, alt_s = cfg.SITES[pick]
        LOC = make_loc(pick, f"{pick} · {SITE_LABEL.get(pick, pick)}", SITE_LABEL.get(pick, pick).split(" (")[0],
                       lat_s, lon_s, alt_s, A.get(pick), True)
    st.divider()
    st.caption("Ephemerides: JPL Horizons · Telescopes: Skynet · Planner: SPS.py")


def loc_local(t):
    """Local time at the active location -> (datetime, 'UTC-6 · America/Monterrey')."""
    lt, zone = to_local(t, LOC["lat"], LOC["lon"])
    if lt:
        return lt, zone
    h = round(LOC["lon"] / 15)                          # at sea: no zone polygon
    return t + timedelta(hours=h), f"UTC{h:+d} · estimated"


NO_WINDOW = ("no moment between {a:%b %d} and {b:%b %d} when all four planets are ≥ {e}° up while the sky is dark "
             "enough (Sun ≤ {s}°). Mercury stays too low in the bright dawn — this month that happens across the "
             "Southern Hemisphere (the dawn ecliptic is shallow there) and at high northern latitudes.")
#obs = cfg.OBSERVATIONS          # live table (also copied into results.json on each run)

# ------------------------------------------------------------------------- header + KPIs
c_logo, c_title = st.columns([1, 9])
c_logo.image(str(HERE / "assets" / "AppLogo.png"), width=80)
c_title.title("StarGazer · Mission Control")
st.caption(f"Run started {R['run']['started_utc']} · window {R['run']['window_utc'][0][:10]} → "
           f"{R['run']['window_utc'][1][:10]} UTC · scan step {R['run']['scan_step']}")

#captured = sorted({o["planet"] for o in obs if o["status"] == "delivered"})
with_window = [s for s, r in A.items() if r]
best_site = min(with_window, key=lambda s: A[s]["S_deg"]) if with_window else None
k1, k2, k3, k4 = st.columns(4)
k1.metric("Sites planned", len(R["config"]["sites"]))
k2.metric("Sites with all four visible", f"{len(with_window)} / {len(A) + len(R['failed'])}")
#k3.metric("Planets imaged", f"{len(captured)} / 4", help=", ".join(captured))
if LOC["dt"]:
    lt, zone = loc_local(LOC["dt"])
    k4.metric(f"Best moment · {LOC['short']}", f"{lt:%b %d, %H:%M}", f"local time ({zone.split(' · ')[0]})",
              delta_color="off")
else:
    k4.metric(f"Best moment · {LOC['short']}", "No window", "Mercury too low at dawn", delta_color="off")

tab_mission, tab_sky, tab_sol, tab_optics, = st.tabs(
    ["🌍 Mission", "🌅 Sky view", "🪐 Solar system",  "📷 Optics"])

# ------------------------------------------------------------------------- Mission
with tab_mission:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Best four-planet moment per site")
        rows = []
        for site, r in A.items():
            t = utc(r["dt"]) if r else None
            rows.append({"Site": ("▶ " if site == LOC["id"] else "") + site, "Where": SITE_LABEL.get(site, ""),
                         "Status": "✅ window" if r else "— no window",
                         "Best time (UTC)": f"{t:%b %d %H:%M}" if t else "",
                         "Local": local_str(site, t),
                         "Spread S (deg)": r["S_deg"] if r else None,
                         "Lowest planet (deg)": r["min_elev"] if r else None,
                         "Sun (deg)": r["sun_elev"] if r else None})
        for site, why in R["failed"].items():
            rows.append({"Site": site, "Where": SITE_LABEL.get(site, ""), "Status": "❌ failed", "Local": why})
        if not LOC["mission"]:
            c = LOC["conj"]
            rows.append({"Site": "▶ you", "Where": LOC["label"], "Status": "✅ window" if c else "— no window",
                         "Best time (UTC)": f"{LOC['dt']:%b %d %H:%M}" if c else "",
                         "Local": f"{loc_local(LOC['dt'])[0]:%H:%M} local" if c else "",
                         "Spread S (deg)": c["S_deg"] if c else None,
                         "Lowest planet (deg)": c["min_elev"] if c else None,
                         "Sun (deg)": c["sun_elev"] if c else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("▶ = the location selected in the sidebar. Rule for “visible”: every planet ≥ "
                   f"{R['config']['min_elev_deg']}° above the horizon and the Sun ≤ −6°.")
    with right:
        st.subheader("Where we observe from")
        pts = [{"site": s, "lon": c[0], "lat": c[1], "color": SERIES[1] if s == LOC["id"] else SERIES[0],
                "size": 140000 if s == LOC["id"] else 60000}
               for s, c in R["config"]["sites"].items() if c]
        if not LOC["mission"]:
            pts.append({"site": "you", "lon": LOC["lon"], "lat": LOC["lat"], "color": SERIES[1], "size": 140000})
        st.map(pd.DataFrame(pts), latitude="lat", longitude="lon", size="size", color="color")
        st.caption("Orange = selected location.")

    # ---- detail panel for the selected location
    st.divider()
    st.subheader(f"📍 {LOC['label']}")
    if not LOC["dt"]:
        st.warning(f"**{LOC['short']}**: " + NO_WINDOW.format(a=cfg.START, b=cfg.STOP, e=cfg.MIN_ELEV,
                                                               s=cfg.MAX_SUN_ALL))
    else:
        c = LOC["conj"]
        lt, zone = loc_local(LOC["dt"])
        ysky = sky_table(LOC["id"]) if LOC["has_sky"] else None
        if ysky is not None:
            ysky = ysky[ysky["dt"] == ysky.loc[(ysky["dt"] - pd.Timestamp(LOC["dt"])).abs().idxmin(), "dt"]]
        look = ""
        if ysky is not None and not ysky.empty:
            pts16 = ["north", "north-northeast", "northeast", "east-northeast", "east", "east-southeast",
                     "southeast", "south-southeast", "south", "south-southwest", "southwest", "west-southwest",
                     "west", "west-northwest", "northwest", "north-northwest"]
            look = f" Look **{pts16[round(float(ysky['azi'].mean()) / 22.5) % 16]}**, low, before sunrise."
        st.success(f"Best moment **{lt:%A %b %d, %H:%M}** local ({zone}) = {LOC['dt']:%b %d %H:%M} UTC.{look}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Grouping spread", f"{c['S_deg']:.1f}°")
        m2.metric("Lowest planet", f"{c['min_elev']:.1f}°",
                  ysky.loc[ysky["elev"].idxmin(), "name"] if ysky is not None and not ysky.empty else None,
                  delta_color="off")
        m3.metric("Sun", f"{c['sun_elev']:.1f}°", "below horizon", delta_color="off")
        if A.get("GZ05") and LOC["id"] != "GZ05":
            h = (LOC["dt"] - utc(A["GZ05"]["dt"])).total_seconds() / 3600
            m4.metric("vs. Monterrey (UTC)", f"{h:+.1f} h", "same local dawn, different clock", delta_color="off")
        st.caption("The grouping looks about the same everywhere, but the moment is the *local* dawn, so the clock "
                   "time moves with longitude and latitude. Change the location in the sidebar — every tab follows. "
                   "Sky view, Solar system and Optics all open at this moment.")

    with st.expander("Multi-site capture plans (Part B) — shortest span holding one slot per planet"):
        for i, p in enumerate(R["B_plans"], 1):
            st.markdown(f"**Plan {i}** · starts {p['start_utc']} · span {p['span_s'] / 60:.0f} min")
            st.dataframe(pd.DataFrame(p["slots"]).drop(columns=["index"], errors="ignore").round(1),
                         hide_index=True, use_container_width=True)
        st.info("Known limitation: Part B applies camera Sun limits (−4°/−6°) to every site. Robotic "
                "telescopes need the Sun ≤ −10°, so slots at Skynet sites can be infeasible. "
                "Future work: per-site Sun limits.")

# ------------------------------------------------------------------------- Sky view
with tab_sky:
    if not LOC["has_sky"]:
        st.info(f"**{LOC['short']}** has no four-planet window, so there is no sky to show. "
                "Pick another location in the sidebar.")
    else:
        site = LOC["id"]
        st.markdown(f"**{LOC['label']}** · ±{cfg.REFINE_HALF_WINDOW_MIN} min around its best moment")
        df = sky_table(site)
        times = sorted(pd.to_datetime(df["dt"].unique()))
        best = pd.Timestamp(LOC["dt"])
        t_sel = st.select_slider("Time (UTC)", options=times, value=best if best in times else times[0],
                                 key=f"sky_t_{site}",           # new location -> slider restarts at its best moment
                                 format_func=lambda t: f"{pd.Timestamp(t):%b %d %H:%M}")
        now = df[df["dt"] == t_sel].copy()
        now["size"] = (90 + 70 * (2 - now["apmag"])).clip(60, 700)
        sun = float(now["sun_elev"].iloc[0])

        m1, m2, m3 = st.columns(3)
        lt, zone = loc_local(pd.Timestamp(t_sel).to_pydatetime())
        m1.metric("Local time", f"{lt:%H:%M}", zone, delta_color="off")
        m2.metric("Sun altitude", f"{sun:.1f}°", "civil twilight" if sun > -6 else "darker than −6°",
                  delta_color="off")
        m3.metric("Lowest planet", f"{now['elev'].min():.1f}°", now.loc[now["elev"].idxmin(), "name"],
                  delta_color="off")

        x_dom = [df["azi"].min() - 8, df["azi"].max() + 8]
        y_dom = [0, max(df["elev"].max() + 8, 30)]
        base = alt.Chart(now).encode(
            x=alt.X("azi:Q", title="Azimuth (deg) → east is 90°, south is 180°", scale=alt.Scale(domain=x_dom)),
            y=alt.Y("elev:Q", title="Altitude above horizon (deg)", scale=alt.Scale(domain=y_dom)),
            tooltip=[alt.Tooltip("name:N", title="Planet"), alt.Tooltip("azi:Q", format=".1f", title="Azimuth"),
                     alt.Tooltip("elev:Q", format=".1f", title="Altitude"),
                     alt.Tooltip("apmag:Q", format=".2f", title="Magnitude")])
        dots = base.mark_circle(color=INK, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
            size=alt.Size("size:Q", legend=None, scale=None))
        labels = base.mark_text(align="left", dx=14, dy=-2, color=INK, fontSize=13).encode(text="name:N")
        limit = alt.Chart(pd.DataFrame({"y": [cfg.MIN_ELEV]})).mark_rule(
            color=MUTED, strokeDash=[4, 4], strokeWidth=1).encode(y="y:Q")
        limit_lbl = alt.Chart(pd.DataFrame({"y": [cfg.MIN_ELEV], "x": [x_dom[0]]})).mark_text(
            align="left", dx=4, dy=-7, color=MUTED, fontSize=11).encode(
            x="x:Q", y="y:Q", text=alt.value(f"planner minimum {cfg.MIN_ELEV}°"))
        ground = alt.Chart(pd.DataFrame({"y": [0], "y2": [y_dom[1] * 0.04]})).mark_rect(
            color="#2a2a28").encode(y="y:Q", y2="y2:Q")
        show(alt.layer(ground, limit, limit_lbl, dots, labels).properties(height=460))
        st.caption("Dot size = brightness (bigger is brighter). Positions are topocentric, airless, from JPL "
                   "Horizons for this site. Drag the slider to watch the planets rise.")

# ------------------------------------------------------------------------- Solar system (step D)
@st.cache_data
def orbit_lines(elements_csv: str):
    """One osculating ellipse per body (first epoch in the file), heliocentric ecliptic J2000, AU."""
    import orbital_plot
    el = pd.read_csv(elements_csv)
    out = []
    for body, g in el.groupby("Body"):
        pts, _, _ = orbital_plot.ellipse_points(g.iloc[0], N=360)
        out.append(pd.DataFrame({"name": PLANETS.get(str(body), "Earth" if body == 399 else str(body)),
                                 "x": pts[:, 0], "y": pts[:, 1], "i": range(len(pts))}))
    return pd.concat(out, ignore_index=True)


@st.cache_data
def orbit_snapshot(t: str):
    import orbital_analyzer as orb
    with contextlib.redirect_stdout(io.StringIO()):
        return orb.alignment(pd.Timestamp(t), pipeline.ORBIT_FILES[0])


@st.cache_data
def orbit_times():
    import orbital_analyzer as orb
    return sorted(pd.to_datetime(orb.load_positions(pipeline.ORBIT_FILES[0])["dt"].unique()))


with tab_sol:
    D = R.get("D_orbits")
    if not D or "error" in D or not all((HERE / f).exists() for f in pipeline.ORBIT_FILES):
        st.info(D["error"] if D and "error" in D else "No heliocentric data in this run — press **Run pipeline**.")
    else:
        st.subheader("Why they line up: the view from above the Solar System")
        c1, c2 = st.columns([1, 3])
        zoom = c1.radio("Scale", ["Inner planets", "Out to Jupiter"], horizontal=False)
        times = orbit_times()
        want = pd.Timestamp(LOC["dt"] or utc(D["when_utc"]))      # the selected location's best moment
        best = min(times, key=lambda t: abs(t - want))
        c2.caption(f"Opens at the best moment for **{LOC['short']}**" if LOC["dt"] else
                   f"{LOC['short']} has no window — showing the mission's best moment")
        t_sel = c2.select_slider("Date (UTC)", options=times, value=best, key=f"sol_t_{LOC['id']}",
                                 format_func=lambda t: f"{pd.Timestamp(t):%b %d %H:%M}")
        snap = orbit_snapshot(str(t_sel))
        b = snap["bodies"]
        pl = b[b["body"] != 399]
        m1, m2, m3 = st.columns(3)
        m1.metric("Spread seen from Earth", f"{snap['geo_spread_deg']:.1f}°",
                  "ecliptic longitude, all four", delta_color="off")
        m2.metric("Spread around the Sun", f"{snap['helio_spread_deg']:.1f}°", "heliocentric longitude",
                  delta_color="off")
        merc = pl[pl["name"] == "Mercury"]
        if not merc.empty:
            m3.metric("Mercury from the Sun", f"{float(merc['elong'].iloc[0]):.1f}°", "elongation seen from Earth",
                      delta_color="off")

        lim = 1.8 if zoom == "Inner planets" else 5.8
        sc = dict(scale=alt.Scale(domain=[-lim, lim], nice=False))
        orbits_df = orbit_lines(pipeline.ORBIT_FILES[1])
        earth = b[b["body"] == 399].iloc[0]
        sight = pd.concat([pd.DataFrame({"name": pl["name"], "x": earth["x"], "y": earth["y"], "o": 0}),
                           pd.DataFrame({"name": pl["name"], "x": pl["x"], "y": pl["y"], "o": 1})])
        sun_line = pd.DataFrame({"x": [earth["x"], 0], "y": [earth["y"], 0], "o": [0, 1]})

        orbit_l = alt.Chart(orbits_df).mark_line(color=GRID, strokeWidth=1.2, clip=True).encode(
            x=alt.X("x:Q", title="X (AU) · ecliptic J2000", **sc), y=alt.Y("y:Q", title="Y (AU)", **sc),
            detail="name:N", order="i:Q")
        sight_l = alt.Chart(sight).mark_line(color=SERIES[1], strokeWidth=1.5, strokeDash=[5, 4],
                                             opacity=0.9, clip=True).encode(
            x="x:Q", y="y:Q", detail="name:N", order="o:Q")
        sun_l = alt.Chart(sun_line).mark_line(color=MUTED, strokeWidth=1, strokeDash=[2, 3], clip=True).encode(
            x="x:Q", y="y:Q", order="o:Q")
        sun = alt.Chart(pd.DataFrame({"x": [0], "y": [0], "name": ["Sun"]})).mark_circle(
            color="#e8b84a", size=260, opacity=1).encode(x="x:Q", y="y:Q")
        tips = [alt.Tooltip("name:N", title="Body"), alt.Tooltip("r_au:Q", format=".3f", title="Distance from Sun (AU)"),
                alt.Tooltip("helio_lon:Q", format=".1f", title="Heliocentric longitude"),
                alt.Tooltip("geo_lon:Q", format=".1f", title="Longitude seen from Earth"),
                alt.Tooltip("elong:Q", format=".1f", title="Angle from Sun (Earth view)")]
        bod = alt.Chart(b).encode(x="x:Q", y="y:Q", tooltip=tips)
        dots = bod.mark_circle(size=120, opacity=1, stroke=SURFACE, strokeWidth=2, clip=True).encode(
            color=alt.condition(alt.datum.name == "Earth", alt.value(SERIES[0]), alt.value(INK)))
        lbls = bod.mark_text(align="left", dx=10, dy=-8, color=INK, fontSize=13, clip=True).encode(text="name:N")
        sun_lbl = alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(
            align="left", dx=12, dy=12, color=INK2, fontSize=12).encode(x="x:Q", y="y:Q", text=alt.value("Sun"))
        show(alt.layer(orbit_l, sun_l, sight_l, sun, dots, lbls, sun_lbl).properties(height=620))
        st.caption("Top-down, Sun-centred (JPL Horizons vectors, ecliptic J2000). Orange dashed = Earth's line of "
                   "sight to each planet; grey dotted = direction of the Sun. The grouping we photograph is the "
                   "fan of orange lines: all four point into the same patch of dawn sky, just west of the Sun. "
                   "Orbits are osculating ellipses from the JPL elements (orbital_analyzer.py → orbital_plot.py).")
        st.dataframe(pl[["name", "r_au", "dist_au", "helio_lon", "geo_lon", "geo_lat", "elong"]].rename(columns={
            "r_au": "from Sun (AU)", "dist_au": "from Earth (AU)", "helio_lon": "helio lon (deg)",
            "geo_lon": "lon seen from Earth (deg)", "geo_lat": "lat (deg)", "elong": "from Sun (deg)"}).round(2),
            hide_index=True, use_container_width=True)



# ------------------------------------------------------------------------- Optics
from instrument_analyzer import Instrument, optical_geometry   # noqa: E402

PRESETS = {   # starting points for "Try your own" (typical published specs; edit freely)
    "Phone · main camera": dict(focal_length=6.3, f_number=1.7, sensor_width=9.8, sensor_height=7.3,
                                h_px=4080, v_px=3060, tracking=False),
    "Phone · ultrawide": dict(focal_length=2.1, f_number=1.9, sensor_width=5.7, sensor_height=4.3,
                              h_px=4080, v_px=3060, tracking=False),
    "DSLR/mirrorless · 24 mm lens": dict(focal_length=24.0, f_number=2.8, sensor_width=36.0, sensor_height=24.0,
                                         h_px=6000, v_px=4000, tracking=False),
    "DSLR/mirrorless · 50 mm lens": dict(focal_length=50.0, f_number=1.8, sensor_width=36.0, sensor_height=24.0,
                                         h_px=6000, v_px=4000, tracking=False),
    "8-inch telescope + astro camera": dict(focal_length=1000.0, f_number=5.0, sensor_width=11.3,
                                            sensor_height=7.1, h_px=3008, v_px=3008, tracking=True),
}


@st.cache_data(show_spinner="Running the optics model…")
def optics_for(site: str, when: str, spec: dict, name: str):
    """Same call the pipeline makes in step A: optical_geometry on the site's fine ephemeris."""
    with contextlib.redirect_stdout(io.StringIO()):
        return optical_geometry(str(HERE / f"{site}_conj_raw.csv"), Instrument(name=name, **spec),
                                peak=pd.Timestamp(when).to_pydatetime(), min_elev=cfg.MIN_ELEV, sky_mag=cfg.SKY_MAG,
                                only=pipeline.ALL_NAMES, verbose=False)


@st.cache_data
def frame_positions(site: str, when: str):
    """Planet offsets (deg) on the sky plane around the group centre, celestial north up."""
    import numpy as np
    with contextlib.redirect_stdout(io.StringIO()):
        pl = load_clean(str(HERE / f"{site}_conj_raw.csv"))
    pl["dt"] = pd.to_datetime(pl["date"], format=FMT)
    pl = pl[pl["dt"] == pl.loc[(pl["dt"] - pd.Timestamp(when)).abs().idxmin(), "dt"]]
    a, d = np.radians(pd.to_numeric(pl["ra_app"])), np.radians(pd.to_numeric(pl["dec_app"]))
    s = np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], axis=1).sum(axis=0)
    a0, d0 = np.arctan2(s[1], s[0]), np.arcsin(s[2] / np.linalg.norm(s))
    D = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    return pd.DataFrame({"name": pl["planet"].map(PLANETS).values,
                         "x": np.degrees(np.cos(d) * np.sin(a - a0) / D),
                         "y": np.degrees((np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / D)})


def secs(t):
    return "—" if t is None or pd.isna(t) else (f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s")


def verdict(site, spec, name):
    import math
    inst = Instrument(name=name, **spec)
    r = optics_for(site, str(LOC["dt"]), spec, name)
    fisheye = inst.projection != "rectilinear"
    req_h, req_v = r["required_deg"]
    fov_h, fov_v = math.degrees(inst.fov_h), math.degrees(inst.fov_v)
    fits = r["fits_single_frame"]
    fits_rot = (not fisheye) and fov_v >= req_h and fov_h >= req_v
    ex = r["exposure"].reset_index()
    vis = ex[ex["visible_samples"] > 0]
    t_lo, t_hi = vis["t_snr_s"].max(), vis["t_max_s"].min()
    one_shot = len(vis) == 4 and vis["ok_alone"].all() and t_lo <= t_hi
    alone_ok = len(vis) == 4 and vis["ok_alone"].all()

    # ---- verdict banner
    if (fits or fits_rot) and one_shot:
        st.success(f"✅ **{name} meets the requirements** — all four planets fit in one frame"
                   f"{' (camera turned 90°)' if fits_rot and not fits else ''} and a single "
                   f"{secs(math.sqrt(t_lo * t_hi))} exposure records every planet without saturating.")
    elif (fits or fits_rot) and alone_ok:
        st.warning(f"🟡 **{name} can do it with an exposure bracket** — all four fit in the frame, but no single "
                   "exposure suits every planet (Venus saturates before the faintest one has enough signal). "
                   "Take a short and a long frame from the same spot and combine them.")
    elif fits or fits_rot:
        bad = ", ".join(vis.loc[~vis["ok_alone"], "planet"]) or "a planet below the horizon limit"
        st.error(f"❌ **{name}: frame is wide enough, but the exposure fails for {bad}.**")
    else:
        n = math.ceil(req_h / fov_h) * math.ceil(req_v / fov_v)
        st.error(f"❌ **{name} is too narrow for the full grouping** — " +
                 (f"it would need a mosaic of ≈{n} frames." if n <= 50 else
                  f"its {fov_h * 60:.0f}′ field is ~{req_h / fov_h:.0f}× smaller than the grouping.") +
                 " Use it for close-ups of single planets instead.")

    # ---- checklist
    c1, c2, c3 = st.columns(3)
    if fisheye:
        c1.metric("Field of view", "All-sky fisheye", "✅ every planet inside the circle" if fits else
                  "❌ a planet falls outside", delta_color="off")
    else:
        c1.metric("Field of view (H × V)", f"{fov_h:.1f}° × {fov_v:.1f}°",
                  f"needs {req_h:.1f}° × {req_v:.1f}° · " + ("✅ fits" if fits else "✅ fits rotated 90°"
                                                            if fits_rot else "❌ too small"), delta_color="off")
    c2.metric("One exposure for all four", secs(math.sqrt(t_lo * t_hi)) if one_shot else "Not possible",
              f"usable range {secs(t_lo)} – {secs(t_hi)}" if one_shot else "needs separate exposures",
              delta_color="off")
    pix = r["sampling"]["n_x_peak"]
    c3.metric("Planet detail", f"{pix.max():.1f} px", f"largest disk ({pix.idxmax()}) · {r['sampling']['quality'][pix.idxmax()]}",
              delta_color="off")

    left, right = st.columns(2)
    # ---- frame vs planets
    with left:
        st.markdown("**What the camera sees** (celestial north up)")
        if fisheye:
            st.info("All-sky lens: the check is that every planet lands inside the image circle above "
                    f"{cfg.MIN_ELEV}°.")
        else:
            pos = frame_positions(site, str(LOC["dt"]))
            cx, cy = (pos["x"].max() + pos["x"].min()) / 2, (pos["y"].max() + pos["y"].min()) / 2
            w, h = (fov_v, fov_h) if fits_rot and not fits else (fov_h, fov_v)
            sp = max(w, h, req_h, req_v, 1e-3) / 2 * 1.15
            box = lambda bw, bh: pd.DataFrame({"x": [cx - bw / 2], "x2": [cx + bw / 2],
                                               "y": [cy - bh / 2], "y2": [cy + bh / 2]})
            xs = alt.Scale(domain=[cx + sp, cx - sp], nice=False)       # east on the left, as on the sky
            ys = alt.Scale(domain=[cy - sp, cy + sp], nice=False)
            frame = alt.Chart(box(w, h)).mark_rect(fill=SERIES[0], fillOpacity=0.12, stroke=SERIES[0],
                                                   strokeWidth=2, clip=True).encode(
                x=alt.X("x:Q", scale=xs, title="← east   offset (deg)   west →"), x2="x2:Q",
                y=alt.Y("y:Q", scale=ys, title="offset (deg)"), y2="y2:Q")
            need = alt.Chart(box(req_h, req_v)).mark_rect(fill=None, stroke=MUTED, strokeDash=[4, 4],
                                                          strokeWidth=1).encode(x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q")
            p = alt.Chart(pos).encode(x="x:Q", y="y:Q")
            chart = alt.layer(frame, need, p.mark_circle(color=INK, size=90, opacity=1),
                              p.mark_text(align="left", dx=9, dy=-7, color=INK, fontSize=12).encode(text="name:N")
                              ).properties(width=430, height=430)
            st.altair_chart(style(chart), use_container_width=False, theme=None)
            st.caption("Blue = camera frame · dashed = space the grouping needs (incl. "
                       f"{inst.margin:.0%} margin) · dots = planets at the best moment from {site}.")

    # ---- exposure windows
    with right:
        st.markdown("**Usable exposure per planet** (enough signal → before saturation)")
        e = vis[["planet", "t_snr_s", "t_max_s", "ok_alone"]].copy()
        e["lo"], e["hi"] = e[["t_snr_s", "t_max_s"]].min(axis=1), e[["t_snr_s", "t_max_s"]].max(axis=1)
        e["state"] = e["ok_alone"].map({True: "usable", False: "no usable exposure"})
        e["snr"], e["max"] = e["t_snr_s"].map(secs), e["t_max_s"].map(secs)
        bars = alt.Chart(e).mark_bar(height=16, cornerRadius=3).encode(
            y=alt.Y("planet:N", title=None, sort=pipeline.ALL_NAMES),
            x=alt.X("lo:Q", scale=alt.Scale(type="log"), title="exposure time (s, log scale)"), x2="hi:Q",
            color=alt.Color("state:N", title=None, scale=alt.Scale(domain=["usable", "no usable exposure"],
                                                                 range=[SERIES[2], SERIES[1]])),
            tooltip=[alt.Tooltip("planet:N"), alt.Tooltip("snr:N", title="min for SNR"),
                     alt.Tooltip("max:N", title="max before saturating/trailing")])
        layers = [bars]
        if one_shot:
            layers.append(alt.Chart(pd.DataFrame({"t": [math.sqrt(t_lo * t_hi)]})).mark_rule(
                color=INK, strokeDash=[4, 3]).encode(x="t:Q"))
        st.altair_chart(style(alt.layer(*layers).properties(height=260)), use_container_width=True, theme=None)
        st.caption("A single shot works only if one vertical line crosses every green bar "
                   "(dashed line = suggested exposure). Red = the planet saturates before it reaches SNR "
                   f"{inst.target_snr:g}" + ("" if inst.tracking else ", or trails on a fixed tripod") + ".")

    with st.expander("Full numbers"):
        st.dataframe(r["exposure"].round(4), use_container_width=True)
        st.dataframe(r["sampling"], use_container_width=True)


with tab_optics:
    st.subheader("Can this instrument capture the four-planet grouping?")
    if not LOC["has_sky"]:
        st.info(f"**{LOC['short']}** has no four-planet window, so there is nothing to frame. "
                "Pick another location in the sidebar.")
    else:
        mode = st.radio("Instrument", ["Our mission instruments", "Try your own"], horizontal=True,
                        label_visibility="collapsed")
        c1, c2 = st.columns(2)
        site = LOC["id"]
        c1.markdown(f"**Observed from** {LOC['label']}  \nbest moment {LOC['dt']:%b %d %H:%M} UTC "
                    "(change it in the sidebar)")
        if mode == "Our mission instruments":
            keys = list(cfg.INSTRUMENTS)
            default = cfg.SITE_INSTRUMENT.get(site) or "GZ05_S25U_UW"
            key = c2.selectbox("Instrument", keys, index=keys.index(default) if default in keys else 0,
                               key=f"inst_{site}",           # new location -> its own instrument preselected
                               format_func=lambda k: k + ("  ← assigned to this site" if k == default else ""))
            spec, name = cfg.INSTRUMENTS[key], key
        else:
            preset = c2.selectbox("Start from", list(PRESETS))
            p = PRESETS[preset]
            k = preset          # widget keys tied to the preset, so picking a preset refills the boxes
            a1, a2, a3 = st.columns(3)
            fl = a1.number_input("Focal length (mm)", 1.0, 20000.0, float(p["focal_length"]), key=f"fl{k}")
            fn = a1.number_input("f-number (f/…)", 0.95, 40.0, float(p["f_number"]), key=f"fn{k}")
            sw = a2.number_input("Sensor width (mm)", 1.0, 60.0, float(p["sensor_width"]), key=f"sw{k}")
            sh = a2.number_input("Sensor height (mm)", 1.0, 60.0, float(p["sensor_height"]), key=f"sh{k}")
            hp = a3.number_input("Pixels across", 100, 20000, int(p["h_px"]), key=f"hp{k}")
            vp = a3.number_input("Pixels down", 100, 20000, int(p["v_px"]), key=f"vp{k}")
            tr = st.toggle("Mount tracks the sky (telescope mount / star tracker)", p["tracking"], key=f"tr{k}")
            with st.expander("Sensor details (optional — typical values filled in)"):
                b1, b2, b3 = st.columns(3)
                rn = b1.number_input("Read noise (e⁻)", 0.5, 30.0, 3.0, key=f"rn{k}")
                fw = b2.number_input("Full well (e⁻)", 1000, 200000, 20000, step=1000, key=f"fw{k}")
                qe = b3.number_input("Quantum efficiency", 0.1, 1.0, 0.5, key=f"qe{k}")
            spec = dict(focal_length=fl, aperture_mm=fl / fn, sensor_width=sw, sensor_height=sh,
                        h_px=int(hp), v_px=int(vp), read_noise=rn, dark_current=0.1, fwc=fw, qe=qe,
                        t_opt=0.85, sky_mag=18.5, tracking=tr)
            name = "Your instrument"
        verdict(site, spec, name)
        st.caption("Same model the pipeline uses (instrument_analyzer.optical_geometry) on the JPL ephemeris for "
                   f"the chosen site; planets below {cfg.MIN_ELEV}° are ignored; twilight sky for Mercury.")