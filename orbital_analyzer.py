from emphimeris_analyzer import JPL_request
from datetime import datetime
import pandas as pd
import math
import numpy as np
import re
import numpy as np, pandas as pd,matplotlib
import matplotlib.pyplot as plt


K = 0.01720209895   # Gaussian gravitational constant [rad/day], for a in AU
cols=["Body","JDTDB","Date","x","y","z","vx","vy","vz"]

start = datetime(2026, 11, 9, 0, 0)      # already UTC, so skip local_to_utc_gz1
stop  = datetime(2026, 11, 19, 0, 0)
PLANETS = {"199": "Mercury", "299": "Venus", "399": "Earth","499": "Mars", "599": "Jupiter"}
#PLANETS.pop("10")
#PLANETS.update("399")

num = r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?"
mu = 0.0002959122083 #Sun's gravitational parameter in AU/day units 

EPSILON = math.radians(23.4392911) #expressed in rads
R = np.array([
    [1,  0,                 0],
    [0,  math.cos(EPSILON),   math.sin(EPSILON)],
    [0, -math.sin(EPSILON),   math.cos(EPSILON)],
])


"""*******************************************************************************
            JDTDB,            Calendar Date (TDB),                      X,                      Y,                      Z,                     VX,                     VY,                     VZ,
**************************************************************************************************************************************************************************************************"""
def JPL_Orbital_Position_request(start=start, stop=stop, step="1 h"):
    orbital_data = JPL_request(
        start, stop, step,                   # step: "1 h", "10 m", "1 d", ...
        "planetary_orbital_positions.txt",
        bodies=PLANETS,
        EPHEM_TYPE="VECTORS",
        CENTER="500@10",                     # Sun body center
        REF_PLANE="ECLIPTIC",                   # use REF_SYSTEM as the frame (default would be ecliptic)
        REF_SYSTEM="ICRF",
        VEC_TABLE="2",                       # state vector: x, y, z, vx, vy, vz
        VEC_CORR="NONE",                     # geometric states (no light-time correction)
        OUT_UNITS="AU-D",                    # or "KM-S"
        TIME_TYPE="UT",                      # interpret START/STOP as UTC
        CSV_FORMAT="NO",
        # observer-only keys that BASE_PARAMS may carry; None makes requests drop them
        QUANTITIES=None, SITE_COORD=None, ANG_FORMAT=None, CAL_FORMAT=None,
        ELEV_CUT=None, SKIP_DAYLT=None, EXTRA_PREC=None,
    )

    print(f"orbital request: {orbital_data.count(chr(10))} lines")
    rows = []
    with open("planetary_orbital_positions.txt") as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 3):
        l1, l2, l3 = lines[i:i + 3]

        body, rest = l1.split(",", 1)
        jd_str, date = rest.split("=", 1)
        jd = float(jd_str.split(",")[-1].strip()) if "," in jd_str else float(jd_str.strip())

        x, y, z = map(float, re.findall(r"[XYZ]\s*=\s*(" + num + ")", l2))
        vx, vy, vz = map(float, re.findall(r"V[XYZ]\s*=\s*(" + num + ")", l3))

        rows.append([int(body), jd, date.strip(), x, y, z, vx, vy, vz])

    df = pd.DataFrame(rows, columns=["Body","JDTDB","Date","x","y","z","vx","vy","vz"])
    df.to_csv("planetary_orbital_positions.csv", index=False)



#Orbital Element Derivation
#calcualte the position and velocity magnitudes
def indep_backup(csv_base,csv_out):
    df = pd.read_csv(csv_base)
    df["Position"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    df["Velocity"] = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
    df["h_x"] = df["y"] * df["vz"] - df["z"] * df["vy"]
    df["h_y"] = df["z"] * df["vx"] - df["x"] * df["vz"]
    df["h_z"] = df["x"] * df["vy"] - df["y"] * df["vx"]
    df["Angular Momentum"] = np.sqrt(df["h_x"]**2 + df["h_y"]**2 + df["h_z"]**2)
    df["Specific Energy"] = (0.5 * df["Velocity"]**2- mu / df["Position"])
    df["Semimajor Axis"] = -(mu/(2*df["Specific Energy"] ))
    df["e_x"] = (df["vy"]*df["h_z"] - df["vz"]*df["h_y"])/mu - df["x"]/df["Position"]
    df["e_y"] = (df["vz"]*df["h_x"] - df["vx"]*df["h_z"])/mu - df["y"]/df["Position"]
    df["e_z"] = (df["vx"]*df["h_y"] - df["vy"]*df["h_x"])/mu - df["z"]/df["Position"]
    df["Eccentricity"] = np.sqrt(df["e_x"]**2 + df["e_y"]**2 + df["e_z"]**2)
    df["Inclination"] = np.degrees(np.arccos(df["h_z"] / df["Angular Momentum"]))
    df["n_x"] = -df["h_y"]
    df["n_y"] = df["h_x"]
    df["n_z"] = 0
    df["Ascending Node"] = np.degrees(np.arctan2(df["n_y"], df["n_x"])) % 360
    cos_w = (df["n_x"]*df["e_x"] + df["n_y"]*df["e_y"] + df["n_z"]*df["e_z"]) / (np.sqrt(df["n_x"]**2 + df["n_y"]**2 + df["n_z"]**2)* df["Eccentricity"])
    df["Argument of Periapsis"] = np.degrees(np.arccos(np.clip(cos_w, -1, 1)))
    df.loc[df["e_z"] < 0, "Argument of Periapsis"] = (360 - df.loc[df["e_z"] < 0, "Argument of Periapsis"])
    cos_nu = (df["e_x"]*df["x"] + df["e_y"]*df["y"] + df["e_z"]*df["z"]) / (df["Eccentricity"]*df["Position"])
    df["True Anomaly"] = np.degrees(np.arccos(np.clip(cos_nu, -1, 1)))
    df.loc[df["x"]*df["vx"] + df["y"]*df["vy"] + df["z"]*df["vz"] < 0,"True Anomaly"] = (360 - df.loc[df["x"]*df["vx"] + df["y"]*df["vy"] + df["z"]*df["vz"] < 0,"True Anomaly"])
    nu_rad = np.radians(df["True Anomaly"])
    e = df["Eccentricity"]
    df["Eccentric Anomaly"] = np.degrees(2 * np.arctan2(np.sqrt(1-e) * np.sin(nu_rad/2),np.sqrt(1+e) * np.cos(nu_rad/2))) % 360
    E_rad = np.radians(df["Eccentric Anomaly"])
    df["Mean Anomaly"] = np.degrees(E_rad - df["Eccentricity"]*np.sin(E_rad)) % 360

    results = df[["Body", "JDTDB", "Date", "Position", "Velocity","h_x","h_y","h_z",
                "Angular Momentum","Specific Energy","Semimajor Axis","e_x","e_y","e_z",
                "Eccentricity","Inclination","n_x","n_y","n_z","Ascending Node",
                "Argument of Periapsis","True Anomaly","Eccentric Anomaly","Mean Anomaly"]]
    results.to_csv(csv_out, index=False)



def JPL_Orbital_Elements_request(start=start, stop=stop, step="1 h"):
    orbital_data = JPL_request(
    start, stop, step,                   # step: "1 h", "10 m", "1 d", ...
    "planetary_orbital_elements.csv",
    bodies=PLANETS,
    EPHEM_TYPE="ELEMENTS",
    CENTER="500@10",                     # Sun body center
    REF_PLANE="ECLIPTIC",                   # use REF_SYSTEM as the frame (default would be ecliptic)
    REF_SYSTEM="ICRF",
    OUT_UNITS="AU-D",                    # or "KM-S"
    TIME_TYPE="TDB",                      #Barycentric Dynamical Time 
    CSV_FORMAT="YES",
    # observer-only keys that BASE_PARAMS may carry; None makes requests drop them
    QUANTITIES=None, SITE_COORD=None, ANG_FORMAT=None, CAL_FORMAT=None,
    ELEV_CUT=None, SKIP_DAYLT=None, EXTRA_PREC=None,
    )
    print(f"orbital request: {orbital_data.count(chr(10))} lines")
    columns = [
        "Body", "JDTDB", "Date",
        "EC", "QR", "IN", "OM", "W", "Tp",
        "N", "MA", "TA", "A", "AD", "PR"
    ]

    df = pd.read_csv(
        "planetary_orbital_elements.csv",
        header=None,
        names=columns,
        skipinitialspace=True,
        index_col=False
    )

    df.drop(
        columns=["QR", "Tp", "N", "TA", "AD", "PR"],
        inplace=True
    )
    df.to_csv("planetary_orbital_elements.csv",index=False)    



def solve_kepler(M, e, tol=1e-12):
    M = np.mod(M, 2*np.pi)
    E = M.copy() if e < 0.8 else np.full_like(M, np.pi)
    for _ in range(100):
        dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E

def kepler_state(el, jd):
    jd = np.atleast_1d(jd).astype(float)
    a, e = el.A, el.EC
    inc, Om, w, M0 = np.radians([el.IN, el.OM, el.W, el.MA])
    n = K / a**1.5

    E = solve_kepler(M0 + n*(jd - el.JDTDB), e)

    x  = a*(np.cos(E) - e)
    y  = a*np.sqrt(1 - e**2)*np.sin(E)
    f  = n / (1 - e*np.cos(E))
    vx = -a*np.sin(E)*f
    vy =  a*np.sqrt(1 - e**2)*np.cos(E)*f

    cO, sO, ci, si, cw, sw = np.cos(Om), np.sin(Om), np.cos(inc), np.sin(inc), np.cos(w), np.sin(w)
    R = np.array([[cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci],
                  [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci],
                  [sw*si,             cw*si]])
    return (R @ np.vstack([x, y])).T, (R @ np.vstack([vx, vy])).T

def kepler_propagate(elements_csv, out_csv, jd_grid=None):
    """If jd_grid is None, uses each body's own JDTDB column."""
    elements = pd.read_csv(elements_csv)
    results = []
    for body, g in elements.groupby("Body"):
        el = g.iloc[0]
        jd = g["JDTDB"].values if jd_grid is None else jd_grid
        pos, vel = kepler_state(el, jd)
        out = pd.DataFrame({"Body": body, "JDTDB": jd})
        out[["x","y","z"]] = pos
        out[["vx","vy","vz"]] = vel
        results.append(out)
    kepler = pd.concat(results, ignore_index=True)
    kepler["Position"] = np.linalg.norm(kepler[["x","y","z"]], axis=1)
    kepler["Velocity"] = np.linalg.norm(kepler[["vx","vy","vz"]], axis=1)
    kepler.to_csv(out_csv, index=False)
    return elements


def load_positions(csv="planetary_orbital_positions.csv"):
    """JPL heliocentric vectors with a real UTC datetime column 'dt'."""
    df = pd.read_csv(csv)
    df["dt"] = pd.to_datetime(df["Date"].str.replace("A.D.", "").str.replace("UTC", "").str.strip(),
                              format="%Y-%b-%d %H:%M:%S.%f")
    return df


def _lon(x, y):
    return np.degrees(np.arctan2(y, x)) % 360


def _spread(lons):
    """Smallest arc (deg) that holds every longitude -- handles the 360/0 wrap."""
    s = np.sort(np.asarray(lons) % 360)
    gaps = np.diff(np.append(s, s[0] + 360))
    return float(360 - gaps.max())


def alignment(when, csv="planetary_orbital_positions.csv", earth=399):
    """Snapshot at the sample nearest `when` (UTC). Frame: heliocentric ecliptic J2000, AU.
    helio_lon = where the planet is around the Sun; geo_lon = where Earth sees it on the ecliptic;
    elong = angle from the Sun as seen from Earth. The 'alignment' you photograph is geo_spread."""
    df = load_positions(csv)
    t = df.loc[(df["dt"] - pd.Timestamp(when)).abs().idxmin(), "dt"]
    now = df[df["dt"] == t].set_index("Body")
    E = now.loc[earth, ["x", "y", "z"]].to_numpy(float)
    rows = []
    for body, p in now.iterrows():
        P = p[["x", "y", "z"]].to_numpy(float)
        row = dict(body=int(body), name=PLANETS.get(str(body), str(body)), x=P[0], y=P[1], z=P[2],
                   r_au=float(np.linalg.norm(P)), helio_lon=float(_lon(P[0], P[1])))
        if body != earth:
            d = P - E
            row.update(dist_au=float(np.linalg.norm(d)), geo_lon=float(_lon(d[0], d[1])),
                       geo_lat=float(np.degrees(np.arcsin(d[2] / np.linalg.norm(d)))),
                       elong=float(np.degrees(np.arccos(np.dot(d, -E) / np.linalg.norm(d) / np.linalg.norm(E)))))
        rows.append(row)
    bodies = pd.DataFrame(rows)
    planets = bodies[bodies["body"] != earth]
    return {"when_utc": t.to_pydatetime(),
            "frame": "heliocentric ecliptic J2000 (JPL VECTORS, CENTER 500@10), AU",
            "geo_spread_deg": round(_spread(planets["geo_lon"]), 2),
            "helio_spread_deg": round(_spread(planets["helio_lon"]), 2),
            "bodies": bodies.round(5)}


if __name__ == "__main__":
    JPL_Orbital_Position_request()
    JPL_Orbital_Elements_request()
    # separate file: do not overwrite the JPL vectors written by JPL_Orbital_Position_request()
    elements = kepler_propagate("planetary_orbital_elements.csv", "planetary_orbital_positions_kepler.csv")
