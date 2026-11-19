"""
instrument_analyzer.py
Instrument / optics feasibility model for planetary imaging.

Reads the CSV written by emphimeris_analyzer.JPL_request() and answers:
  - how big is each planet in pixels with this instrument?
  - what resolution limit applies (diffraction vs seeing vs pixels)?
  - what exposure range works for each planet (SNR vs saturation vs trailing)?

All instrument settings live in an Instrument object passed in (see capture_planner.INSTRUMENTS).
"""
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from emphimeris_analyzer import PLANETS, cols, FMT

TARGETS = {k: v for k, v in PLANETS.items() if k != "10"}      # drop the Sun
FOUR = list(TARGETS)                                           # ["199","299","499","599"]
NAMES = [PLANETS[k] for k in FOUR]
AU_KM = 149597870.7
RADIUS_KM = {"199": 2439.7, "299": 6051.8, "499": 3396.2, "599": 71492.0}   # equatorial
LABELS = ["Severely undersampled", "Very limited representation", "Basic disk representation",
          "Better shape representation", "More useful planetary detail potential",
          "High sampling potential"]

WAVELENGTH_M = 550e-9
SEEING_ARCSEC = 2.0            # assumed atmospheric seeing
FGAMMA0_V = 8.8e9              # photons/s/m^2 at V=0 (~550 nm, ~88 nm wide)
K_EXT = 0.2                    # V-band extinction, mag per airmass
SAT_FRAC = 0.8                 # stay below 80% of full well
SIDEREAL_ARCSEC_PER_S = 15.041
RA_IN_HOURS = False            # emphimeris_analyzer requests ANG_FORMAT=DEG, so degrees


@dataclass
class Instrument:
    name: str = "Default 50mm f/2.8"
    focal_length: float = 50.0          # mm
    aperture_mm: float = 50.0 / 2.8     # mm (f-number is derived from this)
    sensor_width: float = 36.0          # mm
    sensor_height: float = 24.0         # mm
    h_px: int = 3840
    v_px: int = 2160
    read_noise: float = 3.0             # e- RMS
    dark_current: float = 0.1           # e-/pixel/s
    fwc: float = 50000                  # e-/pixel
    qe: float = 0.50
    t_opt: float = 0.80
    sky_mag: float = 18.5               # V mag/arcsec^2 (suburban dawn ~18.5, dark ~21)
    tracking: bool = False              # False = fixed tripod, adds a star-trail limit
    margin: float = 0.10                # framing margin for the single-frame check
    target_snr: float = 20

    # Fisheye projection ("rectilinear" or "equidistant_fisheye")
    projection: str = "rectilinear"
    focal_px: float | None = None

    image_circle_diameter_px: float | None = None
    optical_center_x_px: float | None = None
    optical_center_y_px: float | None = None

    boresight_az_deg: float = 0.0
    boresight_elev_deg: float = 90.0
    roll_deg: float = 0.0
    pixel_pitch_x_um: float | None = None
    pixel_pitch_y_um: float | None = None

    @property
    def f_px(self):
        """(fx, fy) focal length in pixels. A calibrated focal_px wins; otherwise
        focal_length / pixel pitch (pitch from pixel_pitch_*_um or sensor size / pixels)."""
        if self.focal_px is not None:
            return self.focal_px, self.focal_px
        px_x = self.pixel_pitch_x_um or self.sensor_width * 1000 / self.h_px
        px_y = self.pixel_pitch_y_um or self.sensor_height * 1000 / self.v_px
        return self.focal_length / (px_x / 1000), self.focal_length / (px_y / 1000)

    @property
    def f_number(self):
        return self.focal_length / self.aperture_mm

    @property
    def aperture_m(self):
        return self.aperture_mm / 1000.0

    @property
    def area_m2(self):
        return math.pi * (self.aperture_m / 2) ** 2

    @property
    def fov_h(self):                    # radians (rectilinear approximation)
        if self.projection == "equidistant_fisheye" and self.image_circle_diameter_px:
            return self.image_circle_diameter_px / self.f_px[0]       # equidistant: r = f * theta
        return 2 * math.atan(self.sensor_width / (2 * self.focal_length))

    @property
    def fov_v(self):
        if self.projection == "equidistant_fisheye" and self.image_circle_diameter_px:
            return self.image_circle_diameter_px / self.f_px[1]
        return 2 * math.atan(self.sensor_height / (2 * self.focal_length))

    @property
    def plate_x(self):                  # arcsec/pixel (field centre)
        if self.projection == "equidistant_fisheye":
            return 206265 / self.f_px[0]
        return 206.265 * (1000 * self.sensor_width / self.h_px) / self.focal_length

    @property
    def plate_y(self):
        if self.projection == "equidistant_fisheye":
            return 206265 / self.f_px[1]
        return 206.265 * (1000 * self.sensor_height / self.v_px) / self.focal_length

    


# ------------------------------------------------------------------ loading
def _table(df, col):
    p = df.pivot(index="date", columns="planet", values=col).reindex(columns=FOUR)
    p.index = pd.to_datetime(p.index, format=FMT)
    return p.sort_index().astype(float)


def load_planets(csv_base):
    """CSV from JPL_request -> dict of time x planet tables, only timestamps with all four planets
    (same rule as emphimeris_analyzer.load_clean) and identical rows in every table."""
    df = pd.read_csv(csv_base, header=None, names=cols,
                     skipinitialspace=True, dtype={"planet": str})
    df["planet"] = df["planet"].str.strip()
    df["date"] = df["date"].str.strip()
    df = df[df["planet"].isin(FOUR)].drop_duplicates(["date", "planet"])

    tables = {c: _table(df, c) for c in ("ra_app", "dec_app", "azi", "elev", "apmag", "delta")}
    complete_mask = pd.concat(tables.values(), axis=1).notna().all(axis=1)   # ALL fields, ALL planets
    complete = complete_mask[complete_mask].index
    if len(complete) == 0:
        raise ValueError("No timestamps contain all four planets "
                         "(JPL_request may have skipped a body; check its 'Skipping ...' messages).")
    return {c: v.loc[complete] for c, v in tables.items()}


def _resolve_peak(idx, peak):
    """peak: None (middle sample), 'YYYY-Mon-DD HH:MM' string, or datetime -> a timestamp in idx."""
    if peak is None:
        return idx[len(idx) // 2]
    ts = pd.to_datetime(peak, format=FMT) if isinstance(peak, str) else pd.Timestamp(peak)
    if ts in idx:
        return ts
    nearest = idx[idx.get_indexer([ts], method="nearest")[0]]
    print(f"[warn] {ts} not in the ephemeris; using nearest sample {nearest}.")
    return nearest


# ------------------------------------------------------------------ geometry

def fisheye_project(inst, azi_deg, elev_deg, margin=None):
    """
    Project a horizontal sky direction onto an equidistant fisheye image.

    Returns:
        x, y: image pixel coordinates
        inside: whether the projected point is inside the usable image circle
    """
    if inst.projection != "equidistant_fisheye":
        raise ValueError("Instrument is not configured for fisheye projection")

    fx, fy = inst.f_px

    # Planet direction in East-North-Up coordinates.
    A = math.radians(azi_deg)
    h = math.radians(elev_deg)

    v = np.array([
        math.cos(h) * math.sin(A),
        math.cos(h) * math.cos(A),
        math.sin(h)
    ])

    # Camera optical axis.
    A0 = math.radians(inst.boresight_az_deg)
    h0 = math.radians(inst.boresight_elev_deg)

    b = np.array([
        math.cos(h0) * math.sin(A0),
        math.cos(h0) * math.cos(A0),
        math.sin(h0)
    ])

    # Angular distance from optical axis.
    dot = np.clip(np.dot(v, b), -1.0, 1.0)
    theta = math.acos(dot)

    # A direction behind the camera cannot be projected.
    if theta >= math.pi / 2:
        return np.nan, np.nan, False

    # Construct camera image axes.
    right = np.cross(b, np.array([0.0, 0.0, 1.0]))

    if np.linalg.norm(right) < 1e-10:
        # Zenith-pointing: standard all-sky orientation (N top, E left, W right),
        # same as irf_fisheye_projector.project_altaz.
        right = np.array([-1.0, 0.0, 0.0])
    else:
        right /= np.linalg.norm(right)

    up = np.cross(right, b)
    up /= np.linalg.norm(up)

    # Camera roll.
    roll = math.radians(inst.roll_deg)

    right_r = right * math.cos(roll) + up * math.sin(roll)
    up_r = -right * math.sin(roll) + up * math.cos(roll)

    # Direction in the camera image plane.
    if theta < 1e-10:
        dx, dy = 0.0, 0.0
    else:
        tangent = (v - dot * b) / math.sin(theta)
        dx = np.dot(tangent, right_r)
        dy = np.dot(tangent, up_r)

    # Equidistant fisheye projection.
    x = inst.optical_center_x_px + fx * theta * dx
    y = inst.optical_center_y_px - fy * theta * dy

    # Usable circular image boundary.
    radius = inst.image_circle_diameter_px / 2
    margin = inst.margin if margin is None else margin
    usable_radius = radius * (1 - margin)

    distance = math.hypot(
        x - inst.optical_center_x_px,
        y - inst.optical_center_y_px
    )

    inside = distance <= usable_radius

    return x, y, inside

def fisheye_frame_coverage(inst, azi_row, elev_row, delta_row, min_elev=None, only=None):
    """
    Check whether the planetary disks fit inside the usable circular image area.
    Only planets in `only` (names; default all four) that are >= min_elev are required,
    matching the visibility rule used by the rest of the pipeline.

    Returns:
        fits: bool
        positions: DataFrame with pixel coordinates and boundary status
    """
    fx, fy = inst.f_px

    # Conservative radial pixel scale.
    f_px = max(fx, fy)

    usable_radius = (
        inst.image_circle_diameter_px / 2
    ) * (1 - inst.margin)

    diameters = disk_arcsec(
        np.asarray(delta_row.values, dtype=float)
    )

    def required(name, body):
        return ((only is None or name in only)
                and (min_elev is None or elev_row[body] >= min_elev))

    records = []

    for i, body in enumerate(FOUR):
        name = PLANETS[body]

        x, y, center_inside = fisheye_project(
            inst,
            azi_row[body],
            elev_row[body]
        )

        if not np.isfinite(x) or not np.isfinite(y):
            records.append({
                "planet": name,
                "x_px": np.nan,
                "y_px": np.nan,
                "disk_radius_px": np.nan,
                "inside": False,
                "required": required(name, body)
            })
            continue

        # Approximate angular disk radius in pixels.
        disk_radius_px = (
            f_px * math.radians(diameters[i] / 3600 / 2)
        )

        center_distance = math.hypot(
            x - inst.optical_center_x_px,
            y - inst.optical_center_y_px
        )

        # Entire planetary disk must remain inside the safe circle.
        inside = (
            center_distance + disk_radius_px <= usable_radius
        )

        records.append({
            "planet": name,
            "x_px": x,
            "y_px": y,
            "disk_radius_px": disk_radius_px,
            "inside": inside,
            "required": required(name, body)
        })

    positions = pd.DataFrame(records).set_index("planet")
    need = positions[positions["required"]]

    return bool(need["inside"].all()), positions

def configuration_extent(ra_vals, dec_vals):
    """Horizontal and vertical extent (radians) of the four-planet configuration."""
    ra_deg = np.asarray(ra_vals, dtype=float) * (15.0 if RA_IN_HOURS else 1.0)
    a = np.radians(ra_deg)
    d = np.radians(np.asarray(dec_vals, dtype=float))
    s = np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], axis=1)
    s0 = s.sum(axis=0)
    s0 /= np.linalg.norm(s0)
    a0, d0 = math.atan2(s0[1], s0[0]), math.asin(s0[2])

    da = a - a0
    D = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(da)
    xi = np.cos(d) * np.sin(da) / D
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(da)) / D
    return xi.max() - xi.min(), eta.max() - eta.min()


def disk_arcsec(delta_au):
    """Apparent angular diameter (arcsec), shape (n_times, 4)."""
    R = np.array([RADIUS_KM[k] for k in FOUR])
    return np.degrees(2 * np.arctan(R / (np.asarray(delta_au, float) * AU_KM))) * 3600


def airmass(elev_deg):
    h = np.clip(np.asarray(elev_deg, float), 1.0, 90.0)        # Kasten-Young
    return 1 / (np.sin(np.radians(h)) + 0.50572 * (h + 6.07995) ** -1.6364)


def _agg(fn, a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # all-NaN slice = planet never visible
        return fn(a, axis=0)


# ------------------------------------------------------------------ sampling / resolution / photons
def planetary_disk_sampling(inst, t, peak_i):
    theta = disk_arcsec(t["delta"].values)
    fov_h_as = math.degrees(inst.fov_h) * 3600
    out = pd.DataFrame({
        "planet": NAMES,
        "theta_min": theta.min(axis=0),
        "theta_peak": theta[peak_i],
        "theta_max": theta.max(axis=0),
        "n_x_peak": theta[peak_i] / inst.plate_x,
        "n_y_peak": theta[peak_i] / inst.plate_y,
        "pct_of_FOV_width": 100 * theta[peak_i] / fov_h_as,
    }).set_index("planet")
    out["quality"] = pd.cut(out["n_x_peak"], [0, 2, 5, 10, 30, 100, np.inf],
                            labels=LABELS, right=False)
    return out.round(3)


def diffraction_sampling(inst, wavelength_m=WAVELENGTH_M):
    theta_R = 206265 * 1.22 * wavelength_m / inst.aperture_m          # arcsec
    N_R = theta_R / max(inst.plate_x, inst.plate_y)                   # pixels per Rayleigh element
    theta_eff = math.sqrt(theta_R ** 2 + SEEING_ARCSEC ** 2)
    return {
        "D_mm": inst.aperture_mm,
        "theta_R_arcsec": theta_R,
        "N_R": N_R,
        "criterion": "sensor-limited (undersampled)" if N_R < 2 else "optics-limited (>= 2 px per element)",
        "theta_eff_arcsec": theta_eff,                                # diffraction + seeing
        "eff_res_px": theta_eff / max(inst.plate_x, inst.plate_y),
    }


def photon_model(inst, t, peak_i):
    mag = t["apmag"].iloc[peak_i]
    F = FGAMMA0_V * 10 ** (-0.4 * mag)                                # photons/s/m^2
    Ne = F * inst.area_m2 * inst.t_opt * inst.qe                      # e-/s
    return pd.DataFrame({
        "planet": NAMES, "mag": mag.values, "F_gamma": F.values, "Ne_rate": Ne.values,
        "flux_vs_brightest": 10 ** (-0.4 * (mag.values - mag.min())),
    }).set_index("planet")


def _sky_array(inst, sky_mag):
    """Scalar, or {planet name: mag/arcsec^2} (e.g. a brighter sky for twilight Mercury)."""
    if sky_mag is None:
        sky_mag = inst.sky_mag
    if isinstance(sky_mag, dict):
        return np.array([sky_mag.get(n, inst.sky_mag) for n in NAMES], float)
    return np.full(len(FOUR), float(sky_mag))


# ------------------------------------------------------------------ exposure model
def exposure_model(inst, t, vis, blur_arcsec, sky_mag=None):
    """Per-planet exposure limits, using only timestamps where the planet is above min_elev."""
    ps_x, ps_y = inst.plate_x, inst.plate_y
    mag = t["apmag"].values + K_EXT * airmass(t["elev"].values)
    theta = disk_arcsec(t["delta"].values)

    S = FGAMMA0_V * 10 ** (-0.4 * mag) * inst.area_m2 * inst.t_opt * inst.qe     # e-/s
    th_eff = np.sqrt(theta ** 2 + blur_arcsec ** 2)
    n_pix = np.maximum(1.0, math.pi / 4 * (th_eff / ps_x) * (th_eff / ps_y))

    sky_pix = _sky_array(inst, sky_mag) - 2.5 * math.log10(ps_x * ps_y)          # per planet
    B = FGAMMA0_V * 10 ** (-0.4 * sky_pix) * inst.area_m2 * inst.t_opt * inst.qe  # e-/pixel/s

    peak_rate = S / n_pix + B + inst.dark_current
    t_sat = SAT_FRAC * inst.fwc / peak_rate

    snr2 = inst.target_snr ** 2
    Q = snr2 * (S + n_pix * (B + inst.dark_current))
    t_snr = (Q + np.sqrt(Q ** 2 + 4 * S ** 2 * snr2 * n_pix * inst.read_noise ** 2)) / (2 * S ** 2)

    if inst.tracking:
        t_trail = np.inf
    else:
        dec_max = np.abs(t["dec_app"].values).max()
        t_trail = max(ps_x, ps_y) / (SIDEREAL_ARCSEC_PER_S * math.cos(math.radians(dec_max)))

    m = vis.values
    mask = lambda a: np.where(m, a, np.nan)
    out = pd.DataFrame({
        "planet": NAMES,
        "visible_samples": m.sum(axis=0),
        "mag_eff": _agg(np.nanmax, mask(mag)),
        "n_pix": _agg(np.nanmax, mask(n_pix)),
        "Ne_rate": _agg(np.nanmin, mask(S)),
        "B_sky": B,
        "t_sat_s": _agg(np.nanmin, mask(t_sat)),
        "t_snr_s": _agg(np.nanmax, mask(t_snr)),
    }).set_index("planet")
    t_hi = np.minimum(out["t_sat_s"], t_trail)
    out["t_max_s"] = t_hi
    out["ok_alone"] = (out["visible_samples"] > 0) & (out["t_snr_s"] <= t_hi)
    out["t_suggest_s"] = np.where(out["ok_alone"], np.sqrt(out["t_snr_s"] * t_hi), np.nan)
    return out, t_trail


# ------------------------------------------------------------------ main entry point
def optical_geometry(csv_base, inst=None, peak=None, min_elev=5.0,
                     sky_mag=None, only=None, verbose=True):

    inst = inst or Instrument()

    t = load_planets(csv_base)
    idx = t["ra_app"].index
    n = len(idx)

    peak_t = _resolve_peak(idx, peak)
    peak_i = idx.get_loc(peak_t)

    vis = t["elev"] >= min_elev

    pick = (
        lambda df: df.loc[[p for p in only if p in df.index]]
    ) if only else (lambda df: df)

    # Angular configuration extent, retained for comparison.
    ext = np.array([
        configuration_extent(
            t["ra_app"].iloc[i].values,
            t["dec_app"].iloc[i].values
        )
        for i in range(n)
    ])

    Theta_H = ext[:, 0].max()
    Theta_V = ext[:, 1].max()

    req_H = Theta_H * (1 + inst.margin)
    req_V = Theta_V * (1 + inst.margin)

    # ---------------------------------------------------------
    # Instrument-specific image coverage
    # ---------------------------------------------------------

    fisheye_positions = None

    if inst.projection == "equidistant_fisheye":

        coverage = []

        for timestamp in idx:
            fits, positions = fisheye_frame_coverage(
                inst,
                t["azi"].loc[timestamp],
                t["elev"].loc[timestamp],
                t["delta"].loc[timestamp],
                min_elev=min_elev,
                only=only
            )

            coverage.append(fits)

            if timestamp == peak_t:
                fisheye_positions = positions

        fits = bool(all(coverage))

    elif inst.projection == "rectilinear":

        fits = (
            inst.fov_h >= req_H
            and inst.fov_v >= req_V
        )

    else:
        raise ValueError(
            f"Unsupported projection: {inst.projection}"
        )

    # ---------------------------------------------------------
    # Existing optical calculations
    # ---------------------------------------------------------

    sampling = planetary_disk_sampling(inst, t, peak_i)
    diff = diffraction_sampling(inst)
    photons = photon_model(inst, t, peak_i)

    expo, t_trail = exposure_model(
        inst, t, vis, diff["theta_eff_arcsec"], sky_mag
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    if verbose:

        print(
            f"Instrument: {inst.name} | "
            f"f={inst.focal_length:g} mm, "
            f"D={inst.aperture_mm:.1f} mm, "
            f"f/{inst.f_number:.2f}"
        )

        print(
            f"Projection: {inst.projection}"
        )

        if inst.projection == "rectilinear":

            print(
                f"Plate scale: "
                f"{inst.plate_x:.3f}\" x "
                f"{inst.plate_y:.3f}\"/px"
            )

            print(
                f"FOV: {math.degrees(inst.fov_h):.3f} deg x "
                f"{math.degrees(inst.fov_v):.3f} deg"
            )

            print(
                f"\nAll four in ONE frame: "
                f"worst extent {math.degrees(Theta_H):.2f} x "
                f"{math.degrees(Theta_V):.2f} deg, needs "
                f"{math.degrees(req_H):.2f} x "
                f"{math.degrees(req_V):.2f} deg -> "
                f"{'fits' if fits else 'does NOT fit'}"
            )

        else:

            print(
                f"Usable image circle: "
                f"{inst.image_circle_diameter_px:.0f} px"
            )

            print(
                f"Optical center: "
                f"({inst.optical_center_x_px:.1f}, "
                f"{inst.optical_center_y_px:.1f})"
            )

            print(
                f"\nFisheye image coverage at all "
                f"{n} timestamps (planets >= {min_elev:g} deg"
                f"{', only ' + ', '.join(only) if only else ''}): "
                f"{'fits' if fits else 'does NOT fit'}"
            )

            print(
                f"\nPlanet positions at reference time "
                f"{peak_t}:"
            )

            print(fisheye_positions.to_string())

        print(f"\nWindow: {idx[0]} -> {idx[-1]} "
              f"({n} samples) | reference time {peak_t}")

        print("\nDisk sampling at reference time:")
        print(pick(sampling).to_string())

        print(
            f"\nDiffraction: Rayleigh "
            f"{diff['theta_R_arcsec']:.2f}\" | "
            f"{diff['N_R']:.2f} px per element "
            f"({diff['criterion']}) | "
            f"diffraction+seeing "
            f"{diff['theta_eff_arcsec']:.2f}\" "
            f"= {diff['eff_res_px']:.2f} px"
        )

        print(
            f"\nCollecting area A = "
            f"{inst.area_m2 * 1e4:.2f} cm^2"
        )

        print(
            photons.to_string(
                float_format=lambda x: f"{x:.3g}"
            )
        )

        print(
            f"\nExposure model "
            f"(elevation >= {min_elev:g} deg only, "
            f"SNR {inst.target_snr:g}, "
            f"{'tracking' if inst.tracking else f'tripod, trail limit {t_trail:.2f} s'}):"
        )

        print(
            pick(expo).to_string(
                float_format=lambda x: f"{x:.3g}"
            )
        )

    return {
        "sampling": sampling,
        "diffraction": diff,
        "photons": photons,
        "exposure": expo,

        "extent_deg": (
            math.degrees(Theta_H),
            math.degrees(Theta_V)
        ),

        "required_deg": (
            math.degrees(req_H),
            math.degrees(req_V)
        ),

        "fits_single_frame": bool(fits),
        "fisheye_positions": fisheye_positions,
        "reference": peak_t
    }