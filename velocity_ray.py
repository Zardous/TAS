"""
ray_analysis.py
---------------
Plots velocity along rays emanating from the virtual origin (pole) of the jet.

Three subplots per figure:
  1. Velocity u along each ray vs axial distance
  2. Centreline velocity u_c vs axial distance
  3. Normalised velocity u / u_c along each ray vs axial distance
  4. Ray geometry in physical space (optionally overlaid with core/halfwidth data)

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

Overlay option
--------------
Pass a dict to `overlay_data` to overlay core/halfwidth scatter and fit lines
onto the ray geometry subplot (ax4).  Build this dict in your main script:

    from ray_analysis import build_overlay_data, plot_ray_analysis

    overlay = build_overlay_data(
        axial_dist        = axial_dist,
        left_core         = left_core,
        right_core        = right_core,
        left_halfwidths   = left_halfwidths,
        right_halfwidths  = right_halfwidths,
        # core fit lines
        m_left=m_left, c_left=c_left,
        m_right=m_right, c_right=c_right,
        x_intersect=x_intersect, y_intersect=y_intersect,
        x_inter_left=x_inter_left, x_inter_right=x_inter_right,
        # halfwidth fit lines
        m_haleft=m_haleft, c_haleft=c_haleft,
        m_haright=m_haright, c_haright=c_haright,
        x_hal_intersect=x_hal_intersect, y_hal_intersect=y_hal_intersect,
        x_intercept_left=x_intercept_left, x_intercept_right=x_intercept_right,
    )
    plot_ray_analysis(cloud, overlay_data=overlay)

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
from typing import TYPE_CHECKING, Any
from pointcloud import PointCloud

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Default halfwidth slopes (dr / dz) from the provided line fits
# ---------------------------------------------------------------------------
_DEFAULT_SLOPE_RIGHT =  0.004427383697712419
_DEFAULT_SLOPE_LEFT  = -0.005114219025903502   # negative → left of centreline


# ---------------------------------------------------------------------------
# Helper: build the overlay_data dict from main-script variables
# ---------------------------------------------------------------------------

def build_overlay_data(
    axial_dist: np.ndarray,
    left_core: np.ndarray,
    right_core: np.ndarray,
    left_halfwidths: np.ndarray,
    right_halfwidths: np.ndarray,
    # --- core fit ---
    m_left: float,
    c_left: float,
    m_right: float,
    c_right: float,
    x_intersect: float,
    y_intersect: float,
    x_inter_left: float,
    x_inter_right: float,
    # --- halfwidth fit ---
    m_haleft: float,
    c_haleft: float,
    m_haright: float,
    c_haright: float,
    x_hal_intersect: float,
    y_hal_intersect: float,
    x_intercept_left: float,
    x_intercept_right: float,
) -> dict[str, Any]:
    """
    Convenience function to bundle all combined-graph arrays into the dict
    expected by plot_ray_analysis(overlay_data=...).

    All parameters match the variable names used in the original main script.
    Returns a dict that can be passed directly to plot_ray_analysis.
    """
    return dict(
        axial_dist        = np.asarray(axial_dist),
        left_core         = np.asarray(left_core),
        right_core        = np.asarray(right_core),
        left_halfwidths   = np.asarray(left_halfwidths),
        right_halfwidths  = np.asarray(right_halfwidths),
        m_left            = float(m_left),
        c_left            = float(c_left),
        m_right           = float(m_right),
        c_right           = float(c_right),
        x_intersect       = float(x_intersect),
        y_intersect       = float(y_intersect),
        x_inter_left      = float(x_inter_left),
        x_inter_right     = float(x_inter_right),
        m_haleft          = float(m_haleft),
        c_haleft          = float(c_haleft),
        m_haright         = float(m_haright),
        c_haright         = float(c_haright),
        x_hal_intersect   = float(x_hal_intersect),
        y_hal_intersect   = float(y_hal_intersect),
        x_intercept_left  = float(x_intercept_left),
        x_intercept_right = float(x_intercept_right),
    )


# ---------------------------------------------------------------------------
# Internal: draw the combined-graph overlay onto an existing Axes
# ---------------------------------------------------------------------------


    """
    Reproduce the "messy combined graph" (vertical orientation, r on x-axis,
    z on y-axis) on top of the ray geometry subplot.

    Scatter points, fit-line extrapolations, collapse/pole markers, and
    reference lines are all added with reduced alpha so they sit behind the
    ray fan without obscuring it.
    """
    axial_dist       = od["axial_dist"]
    left_core        = od["left_core"]
    right_core       = od["right_core"]
    left_halfwidths  = od["left_halfwidths"]
    right_halfwidths = od["right_halfwidths"]

    m_left, c_left   = od["m_left"],   od["c_left"]
    m_right, c_right = od["m_right"],  od["c_right"]
    x_intersect      = od["x_intersect"]
    y_intersect      = od["y_intersect"]
    x_inter_left     = od["x_inter_left"]
    x_inter_right    = od["x_inter_right"]

    m_haleft,  c_haleft  = od["m_haleft"],  od["c_haleft"]
    m_haright, c_haright = od["m_haright"], od["c_haright"]
    x_hal_intersect  = od["x_hal_intersect"]
    y_hal_intersect  = od["y_hal_intersect"]
    x_intercept_left  = od["x_intercept_left"]
    x_intercept_right = od["x_intercept_right"]

    # ---- shared extrapolation domain ----
    z_max_core = x_intersect * 1.2
    x_extrap_core = np.linspace(0, z_max_core, 100)
    x_extrap_half = np.linspace(x_hal_intersect * 1.2, 100, 101)

    # ---- core scatter & fits ----
    ax.scatter(left_core,  axial_dist, color="steelblue",  s=25, alpha=0.55,
               zorder=2, label="Left core (data)")
    ax.scatter(right_core, axial_dist, color="darkorange", s=25, alpha=0.55,
               zorder=2, label="Right core (data)")
    ax.plot(m_left  * x_extrap_core + c_left,  x_extrap_core,
            color="steelblue",  ls="--", lw=1.2, alpha=0.6,
            label="Left core fit")
    ax.plot(m_right * x_extrap_core + c_right, x_extrap_core,
            color="darkorange", ls="--", lw=1.2, alpha=0.6,
            label="Right core fit")
    ax.scatter(y_intersect, x_intersect, color="red", s=60, zorder=5,
               label=f"Core collapse  z={x_intersect:.1f}")

    # ---- halfwidth scatter & fits ----
    ax.scatter(left_halfwidths,  axial_dist, color="mediumseagreen", s=25,
               alpha=0.55, zorder=2, label="Left halfwidth (data)")
    ax.scatter(right_halfwidths, axial_dist, color="mediumpurple",   s=25,
               alpha=0.55, zorder=2, label="Right halfwidth (data)")
    ax.plot(m_haleft  * x_extrap_half + c_haleft,  x_extrap_half,
            color="mediumseagreen", ls="--", lw=1.2, alpha=0.6,
            label="Left halfwidth fit")
    ax.plot(m_haright * x_extrap_half + c_haright, x_extrap_half,
            color="mediumpurple",   ls="--", lw=1.2, alpha=0.6,
            label="Right halfwidth fit")
    ax.scatter(y_hal_intersect, x_hal_intersect, color="red",
               marker="^", s=70, zorder=5,
               label=f"Halfwidth pole  z={x_hal_intersect:.1f}")

    # ---- pole / outlet reference lines ----
    ax.axhline(y=0,                 color="gray",  ls="--", lw=0.8,
               alpha=0.5, label="Jet outlet  (z=0)")
    ax.axhline(y=x_inter_left,      color="blue",  ls=":",  lw=0.9,
               alpha=0.6, label=f"Core pole L  z={x_inter_left:.1f}")
    ax.axhline(y=x_inter_right,     color="green", ls=":",  lw=0.9,
               alpha=0.6, label=f"Core pole R  z={x_inter_right:.1f}")
    ax.axhline(y=x_intercept_left,  color="blue",  ls="-.", lw=0.9,
               alpha=0.6, label=f"HW pole L  z={x_intercept_left:.1f}")
    ax.axhline(y=x_intercept_right, color="green", ls="-.", lw=0.9,
               alpha=0.6, label=f"HW pole R  z={x_intercept_right:.1f}")
    ax.axvline(x=0,                 color="gray",  ls="--", lw=0.8, alpha=0.5)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_ray_analysis(
    cloud,
    pole: tuple[float, float] = (0.0823, -109.964),
    n_rays: int = 3,
    slope_half_spread: float = 0.4,
    vel_attr: str = "velocity_mean",
    fig_path: str | None = "ray_analysis.png",
    slope_hw_right: float | None = None,
    slope_hw_left:  float | None = None,
    overlay_data: dict[str, Any] | None = None,
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
    overlay_data      : dict produced by build_overlay_data().  When supplied,
                        the core radius and halfwidth scatter, fit lines, and
                        reference markers from the combined graph are drawn
                        behind the ray fan in the geometry subplot (ax4).
                        Pass None (default) to show the clean ray geometry only.
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
    z_max = max(z_vals) if z_vals else 300
    z_extrap = np.linspace(pole_z, z_max, 300)

    # Draw combined-graph data first (behind rays) when overlay supplied
    if overlay_data is not None:
        _draw_combined_overlay(ax4, overlay_data)

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

    ax4.plot(pole_r, pole_z, 'ro', markersize=8, zorder=6,
             label="Estimated Pole Position")
    ax4.axhline(pole_z, color="grey", ls=":", alpha=0.7)

    ax4.set_xlabel("Radial Distance / Halfwidth  $r$")
    ax4.set_ylabel("Axial distance  $z$")
    overlay_suffix = " + combined graph" if overlay_data is not None else ""
    ax4.set_title(f"Ray Geometry Extrapolation{overlay_suffix}")

    # Deduplicate legend entries
    handles, labels = ax4.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax4.legend(by_label.values(), by_label.keys(),
               fontsize=6, ncol=2, loc="best")
    ax4.grid(True, alpha=0.25)

    plt.tight_layout()

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {fig_path}")

    plt.show()

    # ---- Subplot 3 : normalised velocity u / u_c ---------------------
    plt.plot()
    plt.axhline(0.5, color="grey", ls=":", lw=1.2,
                label="$u/u_c = 0.5$  (half-width definition)")
    plt.axhline(1.0, color="grey", ls=":", lw=0.8, alpha=0.5)

    for i, slope in enumerate(right_slopes):
        zs, _, u_norm = ray_data(slope)
        lw = 2.2 if i == hw_idx else 1.2
        ls = "-"  if i == hw_idx else "--"
        plt.plot(zs, u_norm, color=c_right[i], lw=lw, ls=ls,
                    label=ray_label("R", i))

    for i, slope in enumerate(left_slopes):
        zs, _, u_norm = ray_data(slope)
        lw = 2.2 if i == hw_idx else 1.2
        ls = "-"  if i == hw_idx else "--"
        plt.plot(zs, u_norm, color=c_left[i], lw=lw, ls=ls,
                    label=ray_label("L", i))

    plt.xlabel("Axial distance  $z/d$")
    plt.ylabel("$u \\ / \\ u_c$")
    plt.title("Normalised velocity  $u/u_c$")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(True, alpha=0.25)
    plt.show()

    plt.plot()
    z_max = max(z_vals) if z_vals else 100
    z_extrap = np.linspace(pole_z, z_max, 100)

    # Draw combined-graph data first (behind rays) when overlay supplied
    if overlay_data is not None:
        _draw_combined_overlay(ax4, overlay_data)

    plt.axvline(0, color="k", lw=1.5, label="Jet axis (r=0)")

    for i, slope in enumerate(right_slopes):
        r_line = pole_r + slope * (z_extrap - pole_z)
        lw = 2.2 if i == hw_idx else 1.2
        ls = "-"  if i == hw_idx else "--"
        plt.plot(r_line, z_extrap, color=c_right[i], lw=lw, ls=ls,
                 label=ray_label("R", i) if i == hw_idx else "")

    for i, slope in enumerate(left_slopes):
        r_line = pole_r + slope * (z_extrap - pole_z)
        lw = 2.2 if i == hw_idx else 1.2
        ls = "-"  if i == hw_idx else "--"
        plt.plot(r_line, z_extrap, color=c_left[i], lw=lw, ls=ls,
                 label=ray_label("L", i) if i == hw_idx else "")

    plt.plot(pole_r, pole_z, 'ro', markersize=8, zorder=6,
             label="Estimated Pole Position")
    plt.axhline(pole_z, color="grey", ls=":", alpha=0.7)

    plt.xlabel("Radial Distance / Halfwidth  $r$")
    plt.ylabel("Axial distance  $z$")
    overlay_suffix = " + combined graph" if overlay_data is not None else ""
    plt.title(f"Ray Geometry Extrapolation{overlay_suffix}")

    # Deduplicate legend entries
    handles, labels = ax4.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               fontsize=6, ncol=2, loc="best")
    plt.grid(True, alpha=0.25)
    plt.show()

    c_overlay = 5.0   # <-- adjust this constant to shift the curve up/down

    fig_uc, ax_uc = plt.subplots(figsize=(8, 5))

    ax_uc.plot(z_arr, uc_arr, "k-", lw=2.5, label="Centreline $u_c$")

    # Only plot the overlay where z != 0 to avoid division by zero
    z_nonzero = z_arr[z_arr != 0]
    overlay = (1.0 / (z_nonzero-2)) * c_overlay + 7
    ax_uc.plot(z_nonzero, overlay, "r--", lw=1.8,
            label=f"$c \\,/\\, z$  ($c = {c_overlay}$)")

    ax_uc.set_xlabel("Axial distance  $z/d$")
    ax_uc.set_ylabel("Centreline velocity  $u_c$")
    ax_uc.set_title("Centreline velocity with $1/z$ decay overlay")
    ax_uc.legend()
    ax_uc.grid(True, alpha=0.25)
    fig_uc.tight_layout()

    if fig_path:
        _uc_path = fig_path.replace(".png", "_centreline.png")
        fig_uc.savefig(_uc_path, dpi=150, bbox_inches="tight")
        print(f"Centreline figure saved → {_uc_path}")

    plt.show()

    return fig






if __name__ == "__main__":
    cloud = PointCloud()
    cloud.read_test_data()
    plot_ray_analysis(cloud)




