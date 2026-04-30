"""
ray_analysis.py
---------------
Plots velocity along rays emanating from the virtual origin (pole) of the jet.

Three subplots per figure:
  1. Velocity u along each ray vs axial distance
  2. Centreline velocity u_c vs axial distance
  3. Normalised velocity u / u_c along each ray vs axial distance

10 rays total (5 left, 5 right).  The jet half-width direction is the *central*
ray on each side; the other four are evenly spread inside / outside it.

The left and right halfwidth slopes are treated independently because the jet
may not be perfectly symmetric. Slopes are given as  dr/dz  evaluated at the
pole, i.e. the same coefficient that appears in:

    r(z) = slope * z + intercept

where that line also passes through the pole  (pole_r, pole_z).

Provided line equations
-----------------------
    Left  half-width:  r = -0.005114219025903502 * z  +  (-0.48011822450086505)
    Right half-width:  r =  0.004427383697712419 * z  +    0.5691104380882352

Usage
-----
    from ray_analysis import plot_ray_analysis
    plot_ray_analysis(cloud)                          # uses default slopes
    plot_ray_analysis(cloud, slope_half_spread=0.3)   # tighter fan
    plot_ray_analysis(cloud, fig_path="rays.png")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from typing import TYPE_CHECKING
from pointcloud import PointClouds

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Default halfwidth slopes (dr / dz) from the provided line fits
# ---------------------------------------------------------------------------
_DEFAULT_SLOPE_RIGHT =  0.004427383697712419
_DEFAULT_SLOPE_LEFT  = -0.005114219025903502   # negative → left of centreline


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_ray_analysis(
    cloud,
    pole: tuple[float, float] = (0.0823, -109.964),
    n_rays: int = 5,
    slope_half_spread: float = 0.4,
    vel_attr: str = "velocity_mean",
    fig_path: str | None = "ray_analysis.png",
    slope_hw_right: float | None = None,
    slope_hw_left:  float | None = None,
) -> plt.Figure:
    """
    Parameters
    ----------
    cloud             : PointCloud (or any object with a .points attribute
                        shaped as list[list[point]])
    pole              : (r_pole, z_pole)  virtual origin in (x/d, z/d) units.
                        Default: (0.0823, -109.964)
    n_rays            : rays per side – must be odd so the half-width ray is
                        the centre ray of the fan.  Default: 5
    slope_half_spread : fractional spread of the fan around the half-width ray.
                        E.g. 0.4 → factors [0.60, 0.80, 1.00, 1.20, 1.40].
                        Default: 0.4
    vel_attr          : point attribute that holds the mean velocity.
                        Falls back to np.mean(point.voltage_data) if absent.
    fig_path          : save path for the figure; None → do not save.
    slope_hw_right    : dr/dz for the right halfwidth ray.  If None, uses the
                        default from the provided line fit.
    slope_hw_left     : dr/dz for the left halfwidth ray (expected negative).
                        If None, uses the default from the provided line fit.
    """
    if n_rays % 2 == 0:
        raise ValueError("n_rays must be odd so the half-width ray is the centre.")

    pole_r, pole_z = pole

    # Use defaults when caller does not override
    if slope_hw_right is None:
        slope_hw_right = _DEFAULT_SLOPE_RIGHT
    if slope_hw_left is None:
        slope_hw_left = _DEFAULT_SLOPE_LEFT

    # ------------------------------------------------------------------
    # 1.  Collect data:  z  →  (radials_sorted, vmeans_sorted)
    # ------------------------------------------------------------------
    stations: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for station_list in cloud.points:
        if not station_list:
            continue
        z = float(station_list[0].axial)
        radials, vmeans = [], []
        for p in station_list:
            radials.append(float(p.radial))
            if hasattr(p, vel_attr):
                vmeans.append(float(getattr(p, vel_attr)))
            else:
                vmeans.append(float(np.mean(p.voltage_data)))

        order = np.argsort(radials)
        stations[z] = (np.array(radials)[order], np.array(vmeans)[order])

    z_vals = sorted(stations.keys())

    # ------------------------------------------------------------------
    # 2.  Centreline velocity at every axial station
    # ------------------------------------------------------------------
    uc: dict[float, float] = {}

    for z in z_vals:
        r, v = stations[z]
        uc[z] = float(np.interp(0.0, r, v))

    # ------------------------------------------------------------------
    # 3.  Build ray slopes
    #
    #     A ray through the pole with slope  s  satisfies:
    #         r(z) = pole_r + s * (z - pole_z)
    #
    #     The half-width line  r = slope_hw * z + intercept  already passes
    #     through the pole, so  s = slope_hw  (the coefficient is identical
    #     in both pole-centred and origin-centred form).
    #
    #     The fan factors spread symmetrically around 1.0, e.g.
    #     [0.60, 0.80, 1.00, 1.20, 1.40] for spread=0.4, n=5.
    # ------------------------------------------------------------------
    factors  = np.linspace(1.0 - slope_half_spread,
                           1.0 + slope_half_spread,
                           n_rays)
    hw_idx   = n_rays // 2   # index of the half-width (central) ray

    right_slopes = slope_hw_right * factors   # all positive
    left_slopes  = slope_hw_left  * factors   # all negative

    # ------------------------------------------------------------------
    # 4.  Helper: velocity and normalised velocity along one ray
    # ------------------------------------------------------------------
    uc_interp = interp1d(
        z_vals,
        [uc[z] for z in z_vals],
        bounds_error=False,
        fill_value=np.nan,
    )

    def ray_data(slope: float):
        """Returns (z_valid, u_ray, u_norm) for a given pole-centred slope."""
        zs, us = [], []
        for z in z_vals:
            r_data, v_data = stations[z]
            r_ray = pole_r + slope * (z - pole_z)
            if r_ray < r_data[0] or r_ray > r_data[-1]:
                continue
            zs.append(z)
            us.append(float(np.interp(r_ray, r_data, v_data)))
        zs = np.array(zs)
        us = np.array(us)
        uc_ray = uc_interp(zs)
        with np.errstate(invalid="ignore", divide="ignore"):
            u_norm = us / uc_ray
        return zs, us, u_norm

  # ------------------------------------------------------------------
    # 5.  Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Ray analysis  |  pole = (r={pole_r}, z={pole_z})  |  "
        f"slope_right={slope_hw_right:.5f},  slope_left={slope_hw_left:.5f}",
        fontsize=12,
        fontweight="bold",
    )

    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    # Colour maps – warm reds for right, cool blues for left
    # Brighter end = ray further from axis
    c_right = cm.YlOrRd(np.linspace(0.35, 0.85, n_rays))
    c_left  = cm.YlGnBu(np.linspace(0.35, 0.85, n_rays))

    z_arr  = np.array(z_vals)
    uc_arr = np.array([uc[z] for z in z_vals])

    # Helper to build a readable label for each ray
    def ray_label(side: str, i: int) -> str:
        if i == hw_idx:
            return f"{side} half-width"
        return f"{side}  ×{factors[i]:.2f}"

    # ---- Subplot 1 : raw velocity along each ray ----------------------
    ax1.plot(z_arr, uc_arr, "k-", lw=2.5, zorder=6, label="Centreline $u_c$")

    for i, slope in enumerate(right_slopes):
        zs, us, _ = ray_data(slope)
        lw = 2.2 if i == hw_idx else 1.2
        ls = "-"  if i == hw_idx else "--"
        ax1.plot(zs, us, color=c_right[i], lw=lw, ls=ls,
                 label=ray_label("R", i))

    for i, slope in enumerate(left_slopes):
        zs, us, _ = ray_data(slope)
        lw = 2.2 if i == hw_idx else 1.2
        ls = "-"  if i == hw_idx else "--"
        ax1.plot(zs, us, color=c_left[i], lw=lw, ls=ls,
                 label=ray_label("L", i))

    ax1.set_xlabel("Axial distance  $z/d$")
    ax1.set_ylabel("Velocity  $u$")
    ax1.set_title("Velocity along rays")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.25)

    # ---- Subplot 2 : centreline velocity only -------------------------
    ax2.plot(z_arr, uc_arr, "k-", lw=2.5)
    ax2.set_xlabel("Axial distance  $z/d$")
    ax2.set_ylabel("Centreline velocity  $u_c$")
    ax2.set_title("Centreline velocity")
    ax2.grid(True, alpha=0.25)

    # ---- Subplot 3 : normalised velocity u / u_c ---------------------
    ax3.axhline(0.5, color="grey", ls=":", lw=1.2,
                label="$u/u_c = 0.5$  (half-width definition)")
    ax3.axhline(1.0, color="grey", ls=":", lw=0.8, alpha=0.5)

    for i, slope in enumerate(right_slopes):
        zs, _, u_norm = ray_data(slope)
        lw = 2.2 if i == hw_idx else 1.2
        ls = "-"  if i == hw_idx else "--"
        ax3.plot(zs, u_norm, color=c_right[i], lw=lw, ls=ls,
                 label=ray_label("R", i))

    for i, slope in enumerate(left_slopes):
        zs, _, u_norm = ray_data(slope)
        lw = 2.2 if i == hw_idx else 1.2
        ls = "-"  if i == hw_idx else "--"
        ax3.plot(zs, u_norm, color=c_left[i], lw=lw, ls=ls,
                 label=ray_label("L", i))

    ax3.set_xlabel("Axial distance  $z/d$")
    ax3.set_ylabel("$u \\ / \\ u_c$")
    ax3.set_title("Normalised velocity  $u/u_c$")
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.25)

    # ---- Subplot 4 : Ray Geometry in Physical Space ------------------
    z_max = max(z_vals) if z_vals else 100
    z_extrap = np.linspace(pole_z, z_max, 100)

    ax4.axvline(0, color="k", lw=1.5, label="Jet axis (r=0)")

    for i, slope in enumerate(right_slopes):
        r_line = pole_r + slope * (z_extrap - pole_z)
        lw = 2.2 if i == hw_idx else 1.2
        ls = "-"  if i == hw_idx else "--"
        ax4.plot(r_line, z_extrap, color=c_right[i], lw=lw, ls=ls,
                 label=ray_label("R", i) if i == hw_idx else "")

    for i, slope in enumerate(left_slopes):
        r_line = pole_r + slope * (z_extrap - pole_z)
        lw = 2.2 if i == hw_idx else 1.2
        ls = "-"  if i == hw_idx else "--"
        ax4.plot(r_line, z_extrap, color=c_left[i], lw=lw, ls=ls,
                 label=ray_label("L", i) if i == hw_idx else "")

    ax4.plot(pole_r, pole_z, 'ro', markersize=8, label="Estimated Pole Position")
    ax4.axhline(pole_z, color="grey", ls=":", alpha=0.7)

    ax4.set_xlabel("Radial Distance / Halfwidth  $r$")
    ax4.set_ylabel("Axial distance  $z$")
    ax4.set_title("Ray Geometry Extrapolation")
    
    # Filter duplicate legend labels
    handles, labels = ax4.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax4.legend(by_label.values(), by_label.keys(), fontsize=7)
    ax4.grid(True, alpha=0.25)

    plt.tight_layout()

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {fig_path}")

    plt.show()
    return fig