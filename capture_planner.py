"""
capture_planner.py
Single source of truth shared by mission_planner (test.py), instrument_analyzer,
irf_fisheye_projector and diagnostic_test.
Nothing here is duplicated anywhere else: change a value once and every stage sees it.
"""
from __future__ import annotations
from datetime import datetime

# ---- time window (UTC) and sampling --------------------------------------------------------
START, STOP = datetime(2026, 11, 1), datetime(2026, 11, 20)
SCAN_STEP = "30m"             # coarse scan over the whole window
REFINE_STEP = "1m"            # fine scan around the chosen time (also feeds the optics model)
REFINE_HALF_WINDOW_MIN = 30   # +/- minutes around the chosen time
REFRESH = False               # True = ignore cached *_raw.csv files and download everything again

# ---- visibility rules (used by the conjunction search AND the capture planner) -------------
MIN_ELEV = 5                                              # deg, every planet
SUN_LIMIT = {"Mercury": -4, "Venus": -6, "Mars": -6, "Jupiter": -6}   # max Sun elevation, deg
MAX_SUN_ALL = min(SUN_LIMIT.values())                     # all four at once -> strictest rule (-6)

# ---- sites: lon (deg E), lat (deg), alt (km); None = not set yet ----------------------------
SITES = {
    "GZ01": (130.87521, 33.88342, 0.0),
    "GZ02": (-110.792, 38.406, 1.332),
    "GZ03": (-78.470897, 37.238431, 0.149),
    "GZ04": (20.4111236, 67.8407189, 0.425),
    "GZ05": (-100.193220, 25.716058, 0.54),
    "GZ05B": (-100.247889, 25.594833, 0.54),
    "GZ05C": (-103.4677372,25.5255908,1.25),
    "GZ06":(-70.8066909,-30.1695219,2.2)
}
# Extra Skynet-only telescopes: used by mosaic_planner.py only, NOT scanned by test.py.
SKYNET_SITES = {
    "MOREHEAD": (-79.05, 35.914, 0.145),     # Skynet Morehead (UNC Chapel Hill), from the API
    "TYMCE": (23.207, 50.193, 0.240),        # Skynet OAUJ-TYMCE (Poland), from the API
}
SITE_INSTRUMENT = {          # None = no camera defined, so no optics check for that site
    "GZ01": "GZ01_7D_10mm",
    "GZ02": "MDRS_WF",
    "GZ03": "Hampden_Sydney",
    "GZ04": "IRF_Kiruna_ASC",
    "GZ05": "GZ05_S25U_UW",
    "GZ05B": "EOSR_200mm",
    "GZ05C":"GZ05_S25U_UW",
    "GZ06": "PROMPT_CTIO6",
}

# ---- instruments (keys must be Instrument fields) -------------------------------------------
INSTRUMENTS = {
    "GZ01_7D_10mm": dict(                      # EF-S 10-18mm f/4.5-5.6 IS STM, at 10 mm f/4.5
        focal_length=10.0, aperture_mm=10.0 / 4.5,
        sensor_width=22.3, sensor_height=14.9, h_px=5184, v_px=3456,
        read_noise=3.0, dark_current=0.1, fwc=20_000, qe=0.40,
        t_opt=0.90, sky_mag=18.5, tracking=False,
    ),
    "GZ01_7D_50mm": dict(                      # EF 50mm f/1.4 (Mercury + Venus close-up)
        focal_length=50.0, aperture_mm=50.0 / 1.4,
        sensor_width=22.3, sensor_height=14.9, h_px=5184, v_px=3456,
        read_noise=3.0, dark_current=0.1, fwc=20_000, qe=0.40,
        t_opt=0.90, sky_mag=18.5, tracking=False,
    ),
    "MDRS_WF": dict(
        focal_length=336.0, aperture_mm=100.0,   # f/3.36 as written; the note says f/4.8
        sensor_width=9.024, sensor_height=9.024, h_px=3008, v_px=3008,
        read_noise=3.0, dark_current=0.1, fwc=50_000, qe=0.50,   # unverified
        t_opt=0.80, sky_mag=18.5, tracking=True,                 # tracking assumed; confirm
    ),
    "Hampden_Sydney": dict(
        focal_length=1949.0, aperture_mm=432.0,
        sensor_width=36.0, sensor_height=24.0, h_px=9576, v_px=6388,
        read_noise=0.86, dark_current=0.0017, fwc=51_400, qe=0.91,  # ASI6200MM Pro
        t_opt=0.80, sky_mag=18.5, tracking=True,                    # tracking assumed; confirm
    ),
    "Morehead": dict(                          # Skynet API 2026-11-13: OTA + camera + readout mode
        focal_length=8478.0, aperture_mm=609.6,
        sensor_width=3056 * 0.012, sensor_height=3056 * 0.012, h_px=3056, v_px=3056,   # 12 um px
        read_noise=10.0, dark_current=0.0, fwc=50_000, qe=0.50,   # qe unverified
        t_opt=0.80, sky_mag=18.5, tracking=True,
    ),
    "OAUJ_Tymce": dict(                        # Skynet API 2026-11-13 (pixel 3.0 um as reported)
        focal_length=2854.0, aperture_mm=355.0,
        sensor_width=9576 * 0.003, sensor_height=6388 * 0.003, h_px=9576, v_px=6388,
        read_noise=10.0, dark_current=0.0, fwc=65_535, qe=0.50,    # qe unverified
        t_opt=0.80, sky_mag=18.5, tracking=True,
    ),
    "IRF_Kiruna_ASC": dict(
        focal_length=8.0, aperture_mm=8.0 / 2.8,
        sensor_width=2832 * 8.396e-3, sensor_height=2832 * 8.404e-3,   # = h_px * pitch (mm)
        h_px=2832, v_px=2832,
        pixel_pitch_x_um=8.396, pixel_pitch_y_um=8.404,
        # Equidistant fisheye: 90 deg zenith distance -> 1350 px (R of the 2700 px sky circle).
        # This calibrated focal_px overrides focal_length/pitch (which would give ~953 px/rad)
        # and is the same model irf_fisheye_projector.py uses (rho = R * z / 90).
        projection="equidistant_fisheye",
        focal_px=1350.0 / (3.141592653589793 / 2),
        image_circle_diameter_px=2700.0,
        optical_center_x_px=1416.0, optical_center_y_px=1416.0,
        boresight_az_deg=0.0, boresight_elev_deg=90.0, roll_deg=0.0,   # zenith: N top, E left
        margin=0.0,                  # the horizon margin is MIN_ELEV, not a fraction of the circle
        read_noise=3.0, dark_current=0.1, fwc=50_000, qe=0.50,        # unverified
        t_opt=0.80, sky_mag=18.5, tracking=False,
    ),
        # ---- GZ05 Monterrey: phone cameras. Sensor/focal derived from published specs (approx.);
    # pixels = default 12 MP binned output. Noise / full well / QE are ASSUMED typical phone values.
    "GZ05_S25U_UW": dict(                      # Galaxy S25 Ultra ultrawide, 13 mm-equiv f/1.9 (all four)
        focal_length=2.1, aperture_mm=2.1 / 1.9,
        sensor_width=4080 * 0.0014, sensor_height=3060 * 0.0014, h_px=4080, v_px=3060,
        read_noise=2.0, dark_current=0.1, fwc=6_000, qe=0.60,
        t_opt=0.85, sky_mag=18.5, tracking=False,
    ),
    "GZ05_S25U_MAIN": dict(                    # Galaxy S25 Ultra main, 23 mm-equiv f/1.7 (Mercury + Venus)
        focal_length=6.3, aperture_mm=6.3 / 1.7,
        sensor_width=4080 * 0.0024, sensor_height=3060 * 0.0024, h_px=4080, v_px=3060,
        read_noise=2.0, dark_current=0.1, fwc=10_000, qe=0.60,
        t_opt=0.85, sky_mag=18.5, tracking=False,
    ),
    "GZ05_IP15P_UW": dict(                     # iPhone 15 Pro ultrawide, 13 mm-equiv f/2.2 (all four)
        focal_length=2.2, aperture_mm=2.2 / 2.2,
        sensor_width=4032 * 0.0014, sensor_height=3024 * 0.0014, h_px=4032, v_px=3024,
        read_noise=2.0, dark_current=0.1, fwc=6_000, qe=0.60,
        t_opt=0.85, sky_mag=18.5, tracking=False,
    ),
    "GZ05_IP15P_MAIN": dict(                   # iPhone 15 Pro main, 24 mm-equiv f/1.78 (Mercury + Venus)
        focal_length=6.86, aperture_mm=6.86 / 1.78,
        sensor_width=4032 * 0.00244, sensor_height=3024 * 0.00244, h_px=4032, v_px=3024,
        read_noise=2.0, dark_current=0.1, fwc=10_000, qe=0.60,
        t_opt=0.85, sky_mag=18.5, tracking=False,
    ),
    # ---- Canon EOS R (full frame 36 x 24 mm, 6720 x 4480 px, 5.36 um) + EF 70-200mm f/2.8L (Mk I)
    # via EF-RF adapter. Noise / full well / QE are typical published values, NOT measured.
    "EOSR_70mm": dict(                         # 70 mm f/2.8 -> Mercury + Venus pair (28.8 x 19.5 deg)
        focal_length=70.0, aperture_mm=70.0 / 2.8,
        sensor_width=36.0, sensor_height=24.0, h_px=6720, v_px=4480,
        read_noise=3.0, dark_current=0.1, fwc=60_000, qe=0.50,
        t_opt=0.90, sky_mag=18.5, tracking=False,
    ),
    "EOSR_200mm": dict(                        # 200 mm f/2.8 -> Mars + Jupiter pair (10.3 x 6.9 deg)
        focal_length=200.0, aperture_mm=200.0 / 2.8,
        sensor_width=36.0, sensor_height=24.0, h_px=6720, v_px=4480,
        read_noise=3.0, dark_current=0.1, fwc=60_000, qe=0.50,
        t_opt=0.90, sky_mag=18.5, tracking=False,
    ),
    # ---- GZ06 Cerro Tololo: Skynet PROMPT-CTIO-6 (API 2026-11-18). OTA 0.4064 m, f = 7.016 m (f/17.3);
    # FLI MicroLine ML4240, 2048 x 2048 px, 13.5 um -> 0.40"/px, 13.6' field. Readout "500KHz":
    # 11 e- read noise, 0 dark, 100k e- full well. Mount min altitude 10 deg, German equatorial.
    # Filters: U B V R I g' r' i' lum h-alpha OIII SII.
    "PROMPT_CTIO6": dict(
        focal_length=7016.0, aperture_mm=406.4,
        sensor_width=2048 * 0.0135, sensor_height=2048 * 0.0135, h_px=2048, v_px=2048,
        read_noise=11.0, dark_current=0.0, fwc=100_000, qe=0.50,   # qe unverified (not in the API)
        t_opt=0.80, sky_mag=18.5, tracking=True,
    ),
}
SKY_MAG = {"Mercury": 16.5}      # brighter sky for twilight Mercury (planet name -> mag/arcsec^2)

# ---- real observations (filled in by hand; copied into results.json) -----------------------
# status: "delivered" | "pending" | "failed". time_utc = ISO 8601 UTC. None = not known yet.
OBSERVATIONS = [
    dict(planet="Venus", source="Skynet", telescope="PROMPT-CTIO-6", observation="venus #aff7b27c",
         time_utc="2026-11-15T08:51Z", filter="V", exposure_s=0.1, status="delivered",
         note="crescent; JPL: 13.7% lit, 50.2 arcsec, alt 10.0 deg, airmass 5.5",
         image="assets/venus_ctio6_20261115.png"),
    dict(planet="Venus", source="Skynet", telescope="PROMPT-CTIO-6", observation="venus #aff7b27c",
         time_utc="2026-11-17T08:47Z", filter="V", exposure_s=0.1, status="failed",
         note="blank frame (raw and reduced): no Venus light. JPL check: Venus alt 10.4 deg, "
              "az 95.4, Sun -10.7 -> geometry OK; likely dome/east horizon at the 10 deg limit"),
    dict(planet="Jupiter", source="Skynet", telescope=None, observation="jupiter", time_utc=None,
         filter="V", exposure_s=None, status="delivered", note="disk + 3 moons, saturated"),
    dict(planet="Mars", source="Skynet", telescope=None, observation="mars", time_utc=None,
         filter="V", exposure_s=None, status="delivered", note="saturated point"),
    dict(planet="Mercury", source="GZ05 camera", telescope=None, observation=None, time_utc=None,
         filter=None, exposure_s=None, status="pending", note="Skynet impossible (Sun <= -10 limit)"),
]

assert set(SITE_INSTRUMENT) <= set(SITES), "SITE_INSTRUMENT has a site missing from SITES"
assert {v for v in SITE_INSTRUMENT.values() if v} <= set(INSTRUMENTS), "unknown instrument"

# ---- Skynet request planning (mosaic_planner.py, skynet_client.py) -------------------------
# Skynet 2 API: https://docs.skynetgo.org/developers/  (token in env var SKYNET_TOKEN).
# skynet_name must match the telescope name/shortName/slug in your Skynet observing access.
# With --remote, filters / min altitude / horizon / readout come from Skynet and override these.
# Values marked ASSUMED are fallbacks used only when Skynet is not queried.
MOSAIC_SITES = ["GZ02", "GZ03"]
MOSAIC_OVERLAP = 0.10            # tile overlap for the contiguous-mosaic comparison
SKYNET_MAX_SUN_DEG = -10        # Skynet "optical safety threshold" (seen in :observability)
SKYNET_SPEC_KIND = ("observation_spec", 1)   # confirmed via :validate on 2026-11-13
SKYNET = {
    "GZ02": dict(skynet_name="MDRS-WF", legacy_id=44,        # legacy api.skynet.unc.edu id
                 filters=["red", "green", "blue", "lum", "h-alpha"],   # legacy site listing
                 binning=1,
                 min_exp_s=0.1,        # ASSUMED minimum exposure (not exposed by the API)
                 horizon_deg=None,     # ASSUMED = MIN_ELEV unless Skynet reports mount/horizon
                 overhead_s=30.0,      # ASSUMED per-exposure overhead unless Skynet reports readout
                 count=5),             # exposures per target per filter
    "GZ03": dict(skynet_name="HSC", legacy_id=9,
                 filters=["g'", "r'", "i'", "h-alpha", "OIII", "SII", "open", "cls-ccd"],
                 binning=3,            # Skynet note: "pixels are very small, so bin it to 3x3"
                 min_exp_s=0.1, horizon_deg=None, overhead_s=30.0, count=5),
    "MOREHEAD": dict(skynet_name="Morehead", legacy_id=10, instrument="Morehead",
                     filters=["U", "B", "V", "R", "open"],          # no narrowband on this wheel
                     binning=2,            # 0.58"/px, still < seeing (2"); doubles full well area
                     min_exp_s=0.1, horizon_deg=None, overhead_s=1.0, count=5),
    "TYMCE": dict(skynet_name="OAUJ-TYMCE", legacy_id=None, instrument="OAUJ_Tymce",
                  filters=["blue", "green", "h-alpha", "lum", "red"],
                  binning=2, min_exp_s=0.1, horizon_deg=None, overhead_s=2.5, count=5),
}
# Tried in order per planet; the first filter whose exposure is feasible wins
# (bright planets fall through to narrowband when broadband saturates). Skynet filter ids.
FILTER_PREFERENCE = {
    "GZ02": ["green", "red", "blue", "lum", "h-alpha"],
    "GZ03": ["g'", "r'", "i'", "h-alpha", "OIII", "SII"],
    "MOREHEAD": ["V", "B", "R", "U"],
    "TYMCE": ["green", "red", "blue", "h-alpha"],
}
# Photon throughput relative to the V-band model in instrument_analyzer (88 nm wide):
# bandpassNm / 88 from GET https://api.skynetgo.org/v1/filter-specifiers (open, cls-ccd: rough).
# "<id>-<n>nm" variants found on a wheel (e.g. h-alpha-5nm) use n / 88 automatically.
FILTER_THROUGHPUT = {
    "red": 102 / 88, "green": 84 / 88, "blue": 89 / 88, "lum": 282 / 88,
    "g'": 150 / 88, "r'": 150 / 88, "i'": 150 / 88,
    "h-alpha": 3 / 88, "OIII": 3 / 88, "SII": 3 / 88,
    "open": 3.5, "cls-ccd": 2.0,
    "U": 76 / 88 * 0.3, "B": 88 / 88, "V": 78 / 88, "R": 137 / 88,   # U also low QE
}

assert set(MOSAIC_SITES) <= set(SKYNET), "MOSAIC_SITES without a SKYNET entry"
assert set(SKYNET) <= set(SITES) | set(SKYNET_SITES), "SKYNET site without coordinates"
assert not set(SKYNET_SITES) & set(SITES), "a site is in both SITES and SKYNET_SITES"
assert all(SITE_INSTRUMENT.get(s) or SKYNET[s].get("instrument") for s in SKYNET),     "SKYNET site without an instrument (SITE_INSTRUMENT or SKYNET[...]['instrument'])"
assert all(set(FILTER_PREFERENCE[s]) <= set(SKYNET[s]["filters"]) for s in FILTER_PREFERENCE),     "FILTER_PREFERENCE lists a filter the telescope does not have"
assert all(f in FILTER_THROUGHPUT for s in FILTER_PREFERENCE for f in FILTER_PREFERENCE[s]),     "FILTER_PREFERENCE filter missing from FILTER_THROUGHPUT"