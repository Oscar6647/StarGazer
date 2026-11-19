import numpy as np, pandas as pd
import matplotlib.pyplot as plt

NAMES = {199: "Mercury", 299: "Venus", 399: "Earth", 499: "Mars", 599: "Jupiter"}

def rot_matrix(inc, Om, w):
    """Perifocal -> heliocentric ecliptic: Rz(Om) Rx(inc) Rz(w)"""
    cO, sO, ci, si, cw, sw = np.cos(Om), np.sin(Om), np.cos(inc), np.sin(inc), np.cos(w), np.sin(w)
    return np.array([
        [cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,  sO*si],
        [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
        [sw*si,             cw*si,             ci   ]])

def ellipse_points(el, N=1000):
    a, e = el.A, el.EC
    inc, Om, w = np.radians([el.IN, el.OM, el.W])
    E = np.linspace(0, 2*np.pi, N)
    pts = np.vstack([a*(np.cos(E) - e),
                     a*np.sqrt(1 - e**2)*np.sin(E),
                     np.zeros_like(E)])
    R = rot_matrix(inc, Om, w)
    return (R @ pts).T, R @ [a*(1 - e), 0, 0], R @ [-a*e, 0, 0]

def plot_ellipses(elements, bodies=None, title="Osculating orbits"):
    fig, ax = plt.subplots(figsize=(8, 8))
    for body, g in elements.groupby("Body"):
        if bodies and body not in bodies:
            continue
        el = g.iloc[0]
        orbit, peri, center = ellipse_points(el)
        line, = ax.plot(orbit[:, 0], orbit[:, 1], label=f"{NAMES.get(body, body)} (e={el.EC:.3f})")
        ax.scatter(*peri[:2],   marker="^", s=25, color=line.get_color())
        ax.scatter(*center[:2], marker="x", s=25, color=line.get_color())
    ax.scatter(0, 0, c="gold", s=100, edgecolors="k", zorder=5, label="Sun (focus)")
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("X (au)"); ax.set_ylabel("Y (au)"); ax.set_title(title)
    ax.legend(fontsize=8)
    return fig, ax


if __name__ == "__main__":
    elements = pd.read_csv("planetary_orbital_elements.csv")
    plot_ellipses(elements, bodies=[199, 299, 399, 499], title="Inner planets")
    plot_ellipses(elements, title="All planets")
    plt.show()