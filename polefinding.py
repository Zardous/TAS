import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from pointcloud import PointCloud
from pointcloud import find_edge

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — EXTRACT ARRAYS FROM POINTCLOUD
# ══════════════════════════════════════════════════════════════════════════════

def extract_arrays(pc: PointCloud):
    """
    Convert PointCloud into flat arrays for the optimiser.

    Returns
    -------
    axial_distances : np.ndarray (N_stations,)
        Sorted unique axial positions.
    x_positions : list of np.ndarray
        x_positions[i] = radial positions at station i, sorted ascending.
    velocities : list of np.ndarray
        velocities[i] = velocity_mean at each radial point at station i.
    centerline_vel : np.ndarray (N_stations,)
        velocity_mean at the point closest to x=0 at each station.
    """
    axial_distances = []
    x_positions     = []
    velocities      = []
    centerline_vel  = []

    for station in pc.points:
        if not station:
            continue

        z    = station[0].axial
        xs   = np.array([p.radial        for p in station])
        us   = np.array([p.velocity_mean for p in station])

        # sort radially so interp1d works correctly
        sort_idx = np.argsort(xs)
        xs = xs[sort_idx]
        us = us[sort_idx]

        # centreline velocity = point closest to x=0
        centre_idx = np.argmin(np.abs(xs))
        uc = us[centre_idx]

        axial_distances.append(z)
        x_positions.append(xs)
        velocities.append(us)
        centerline_vel.append(uc)

    # sort all stations by axial distance
    sort_idx       = np.argsort(axial_distances)
    axial_distances = np.array(axial_distances)[sort_idx]
    x_positions    = [x_positions[i]  for i in sort_idx]
    velocities     = [velocities[i]   for i in sort_idx]
    centerline_vel = np.array(centerline_vel)[sort_idx]

    return axial_distances, x_positions, velocities, centerline_vel


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LINE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def make_line(x_pole: float, z_pole: float, slope: float):
    """
    Return callable x(z) for the line through (x_pole, z_pole) with
    slope m = dx/dz.
    """
    return lambda z: x_pole + slope * (z - z_pole)


def slopes_from_x_at_z(x_targets: np.ndarray,
                        z_ref: float,
                        x_pole: float,
                        z_pole: float) -> np.ndarray:
    """
    Compute slopes so each line passes through (x_pole, z_pole) AND
    hits x_targets[i] at axial plane z_ref.

    Parameters
    ----------
    x_targets : radial positions you want the lines to pass through at z_ref
    z_ref     : the axial reference plane (typically the last station)
    x_pole    : candidate pole x position
    z_pole    : candidate pole z position
    """
    return (x_targets - x_pole) / (z_ref - z_pole)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SAMPLE VELOCITY ALONG A LINE
# ══════════════════════════════════════════════════════════════════════════════

def sample_line(x_pole: float,
                z_pole: float,
                slope: float,
                axial_distances: np.ndarray,
                x_positions: list,
                velocities: list) -> dict:
    """
    Walk every axial station and interpolate the velocity at the exact
    x position where the line crosses that station.

    Parameters
    ----------
    x_pole, z_pole : candidate pole position
    slope          : dx/dz for this line
    axial_distances: 1-D array of axial station positions
    x_positions    : list of radial position arrays, one per station
    velocities     : list of mean velocity arrays, one per station

    Returns
    -------
    dict with keys 'z', 'x', 'u', 'x_gap'
        z     : axial positions where a valid sample was found
        x     : exact x position of the line at each z
        u     : interpolated velocity at each (x, z)
        x_gap : distance to nearest real data point (quality indicator)
    """
    x_on_line = make_line(x_pole, z_pole, slope)

    z_out, x_out, u_out, gap_out = [], [], [], []

    for i, z in enumerate(axial_distances):

        # never sample below the pole
        if z <= z_pole:
            continue

        x_target = x_on_line(z)
        xp = x_positions[i]
        up = velocities[i]

        # skip if line exits the measurement domain at this station
        if x_target < xp.min() or x_target > xp.max():
            continue

        # interpolate velocity at the exact crossing point
        u_val = float(
            interp1d(xp, up, kind='linear', bounds_error=False,
                     fill_value=np.nan)(x_target)
        )

        if np.isnan(u_val):
            continue

        z_out.append(z)
        x_out.append(x_target)
        u_out.append(u_val)
        gap_out.append(float(np.min(np.abs(xp - x_target))))

    return {k: np.array(v) for k, v in
            zip(('z', 'x', 'u', 'x_gap'),
                (z_out, x_out, u_out, gap_out))}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — COMPUTE u/uc RATIOS ALONG A LINE
# ══════════════════════════════════════════════════════════════════════════════

def velocity_ratios(x_pole: float,
                    z_pole: float,
                    slope: float,
                    axial_distances: np.ndarray,
                    x_positions: list,
                    velocities: list,
                    centerline_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute u(z) / uc(z) at every station crossed by the line.

    If the pole is correct, these ratios should be constant along the line.

    Returns
    -------
    z_valid : axial positions where a valid ratio was computed
    ratios  : u/uc at each valid station
    """
    pts = sample_line(x_pole, z_pole, slope,
                      axial_distances, x_positions, velocities)

    if pts['z'].size == 0:
        return np.array([]), np.array([])

    # interpolate centreline velocity at sampled axial positions
    uc_interp = interp1d(axial_distances, centerline_vel,
                         kind='linear', bounds_error=False, fill_value=np.nan)
    uc = uc_interp(pts['z'])

    # keep only points with valid, non-zero centreline velocity
    valid = (~np.isnan(uc)) & (uc != 0) & (~np.isnan(pts['u']))

    return pts['z'][valid], pts['u'][valid] / uc[valid]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — ERROR FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def error_fn(ratios: np.ndarray) -> float:
    """
    Measure how non-constant the velocity ratios are along a line.

    For 3 points this exactly reproduces your original:
        err1  = |r1 - r2| / r1
        err2  = |r1 - r3| / r1
        error = (err1 + err2) / 2

    For N points it generalises naturally by averaging all N-1 deviations.

    Parameters
    ----------
    ratios : array of u/uc values along one line

    Returns
    -------
    scalar error — 0 means perfectly constant (ideal pole)
    """
    if len(ratios) < 2:
        return np.inf

    r_ref = ratios[0]                      # u1/uc1 — first downstream point

    if np.abs(r_ref) < 1e-10:             # guard against near-zero reference
        return np.inf

    # |r1 - ri| / r1  for all i != 0
    relative_errors = np.abs(r_ref - ratios[1:]) / np.abs(r_ref)

    return float(np.mean(relative_errors))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — POLE ERROR (OBJECTIVE FUNCTION FOR OPTIMISER)
# ══════════════════════════════════════════════════════════════════════════════

def pole_error(params: np.ndarray,
               slopes: np.ndarray,
               axial_distances: np.ndarray,
               x_positions: list,
               velocities: list,
               centerline_vel: np.ndarray) -> float:
    """
    Objective function for the pole-finding optimisation.

    For a candidate pole position [x_pole, z_pole]:
        1. Walk every line (slope)
        2. Compute u/uc ratios along it
        3. Score with error_fn
        4. Return mean error across all lines

    A perfect pole gives error = 0.

    Parameters
    ----------
    params         : [x_pole, z_pole] — candidate pole position
    slopes         : 1-D array of line slopes to evaluate
    axial_distances: from extract_arrays()
    x_positions    : from extract_arrays()
    velocities     : from extract_arrays()
    centerline_vel : from extract_arrays()
    """
    x_pole, z_pole = params
    errors = []

    for m in slopes:
        _, ratios = velocity_ratios(x_pole, z_pole, m,
                                    axial_distances, x_positions,
                                    velocities, centerline_vel)
        if ratios.size >= 2:
            errors.append(error_fn(ratios))

    return float(np.mean(errors)) if errors else np.inf


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pole_optimisation(pc: PointCloud,
                          x_pole_init: float,
                          z_pole_init: float) -> tuple[float, float]:
    """
    Full pipeline: PointCloud → optimised pole position.

    Parameters
    ----------
    pc           : your loaded PointCloud
    x_pole_init  : initial x guess (e.g. 0.0 from halfwidth extrapolation)
    z_pole_init  : initial z guess (e.g. -111.0, midpoint of your two estimates)

    Returns
    -------
    (x_opt, z_opt) : optimised pole position
    """

    # ── 1. extract arrays ────────────────────────────────────────────────────
    axial_distances, x_positions, velocities, centerline_vel = extract_arrays(pc)

    SKIP = 4
    axial_distances = axial_distances[SKIP:]
    x_positions     = x_positions[SKIP:]
    velocities      = velocities[SKIP:]
    centerline_vel  = centerline_vel[SKIP:]


    print("Loaded stations:")
    for i, z in enumerate(axial_distances):
        print(f"  z={z:.2f} | {len(x_positions[i])} radial points | "
              f"x=[{x_positions[i].min():.3f}, {x_positions[i].max():.3f}] | "
              f"uc={centerline_vel[i]:.4f}")

    # ── 2. define fan of lines across the full jet width ─────────────────────
    # fan from leftmost to rightmost x at the last (furthest) axial station
    x_sweep = np.linspace(x_positions[-1].min(), #NEED TO CHECK THIS, MAYBE SHOULD BE WITHIN EDGES OF THE CONE ?
                          x_positions[-1].max(),
                          9)

    slopes = slopes_from_x_at_z(x_sweep,
                                 z_ref=axial_distances[-1],
                                 x_pole=x_pole_init,
                                 z_pole=z_pole_init)

    print(f"\nInitial pole guess:  x={x_pole_init:.4f},  z={z_pole_init:.4f}")
    print(f"Testing {len(slopes)} lines")
    print(f"Initial error: {pole_error([x_pole_init, z_pole_init], slopes, axial_distances, x_positions, velocities, centerline_vel):.6f}")

    # ── 3. optimise ──────────────────────────────────────────────────────────
    line_params = [
        {
            'slope'      : m,
            'x_intercept': x_pole_init - m * z_pole_init,   # x at z=0
            'x_at_ref'   : x_sweep[i],                       # x at last station
        }
        for i, m in enumerate(slopes)
    ]
    result = minimize(
        pole_error,
        x0=[x_pole_init, z_pole_init],
        args=(slopes, axial_distances, x_positions, velocities, centerline_vel),
        method='Nelder-Mead',
        options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 2000}
    )
    x_opt, z_opt = result.x

    print(f"\nOptimised pole:  x={x_opt:.4f},  z={z_opt:.4f}")
    print(f"Final error:     {result.fun:.6f}")
    print(f"Converged:       {result.success}")
    print(f"Iterations:      {result.nit}")

    # ── 6. update line_params with optimised pole ────────────────────────────
    # recompute slopes and intercepts using the optimised pole position
    slopes_opt = slopes_from_x_at_z(x_sweep,
                                     z_ref=axial_distances[-1],
                                     x_pole=x_opt,
                                     z_pole=z_opt)

    line_params_opt = [
        {
            'slope'      : m,
            'x_intercept': x_opt - m * z_opt,
            'x_at_ref'   : x_sweep[i],
        }
        for i, m in enumerate(slopes_opt)
    ]

    return x_opt, z_opt, line_params


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # load data
    pc = PointCloud()
    pc.read_test_data()

    # initial pole guess from your halfwidth extrapolation (Image 1)
    # x=0 by symmetry, z = midpoint of your two estimates (-93.88, -128.54)
    X_POLE_INIT = 0.0
    Z_POLE_INIT = (-93.88 + -128.54) / 2     # = -111.21

    x_opt, z_opt, line_params = run_pole_optimisation(pc, X_POLE_INIT, Z_POLE_INIT)

import matplotlib.pyplot as plt

x_opt, z_opt, line_params = run_pole_optimisation(pc, 0.0, -111.21)

axial_distances, x_positions, velocities, centerline_vel = extract_arrays(pc)

SKIP = 4
axial_distances = axial_distances[SKIP:]
x_positions     = x_positions[SKIP:]
velocities      = velocities[SKIP:]
centerline_vel  = centerline_vel[SKIP:]

z_plot = np.linspace(z_opt*1.2, axial_distances[-1]*1.2, 300)


plt.figure()
for lp in line_params:
    # x(z) = x_intercept + slope * z
    # (equivalent to x_pole + slope*(z - z_pole), just written from z=0)
    x_plot = lp['x_intercept'] + lp['slope'] * z_plot
    plt.plot(x_plot, z_plot, 'b--', alpha=0.4)

plt.scatter(x_opt, z_opt, color='red', zorder=5, label=f'Pole ({x_opt:.3f}, {z_opt:.3f})')
plt.xlabel('x/d')
plt.ylabel('axial distance z/d')
plt.legend()
plt.show()