import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pointcloud import PointCloud

# --- Load data ---
cloud = PointCloud()
cloud.read_test_data()

# --- Collect velocities at each axial station ---
stations = {}

for station_list in cloud.points:
    if not station_list:
        continue
    z = float(station_list[0].axial)
    radials, vmeans = [], []
    for p in station_list:
        radials.append(float(p.radial))
        if hasattr(p, "velocity_mean"):
            vmeans.append(float(p.velocity_mean))
        else:
            vmeans.append(float(np.mean(p.voltage_data)))

    order = np.argsort(radials)
    stations[z] = (np.array(radials)[order], np.array(vmeans)[order])

# --- Extract centreline velocity at each station ---
z_vals = sorted(stations.keys())
uc_vals = [float(np.interp(0.0, *stations[z])) for z in z_vals]

z_arr  = np.array(z_vals)
uc_arr = np.array(uc_vals)

# --- Fit 1/x curve to stations from z=4 onwards ---
#     Model:  u_c(z) = a / (z - z0) + c
#     z0 allows a virtual origin offset, c allows a baseline offset

mask = z_arr >= 4
z_fit  = z_arr[mask]
uc_fit = uc_arr[mask]

def inv_model(z, a, z0, c):
    return a / (z - z0) + c

# Initial guesses: a~5, z0~-100 (virtual origin), c~0
p0 = [5.0, -100.0, 0.0]
popt, pcov = curve_fit(inv_model, z_fit, uc_fit, p0=p0)
a_fit, z0_fit, c_fit = popt

print(f"Best fit:  a = {a_fit:.4f},  z0 = {z0_fit:.4f},  c = {c_fit:.4f}")
print(f"Model:  u_c = {a_fit:.4f} / (z - ({z0_fit:.4f})) + {c_fit:.4f}")

# --- Dense curve for smooth overlay ---
z_smooth = np.linspace(z_fit[0], 12, 300)
uc_smooth = inv_model(z_smooth, *popt)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(z_arr, uc_arr, "k-o", lw=2, markersize=5, label="Centreline $u_c$")
ax.plot(z_smooth, uc_smooth, "r--", lw=2,
        label=f"Fit: $\\frac{{{a_fit:.2f}}}{{z - ({z0_fit:.2f})}} + {c_fit:.2f}$")

# Mark the fitted region
ax.axvline(4, color="grey", ls=":", lw=1, alpha=0.7, label="Fit start (z=4)")

ax.set_xlim(left=z_arr[0], right=12)
ax.set_ylim(bottom=0)
ax.set_xlabel("Axial distance  $z/d$")
ax.set_ylabel("Centreline velocity  $u_c$")
ax.set_title("Centreline velocity with $1/z$ best-fit overlay")
ax.legend()
ax.grid(True, alpha=0.25)

fig.tight_layout()
plt.show()

# --- Nozzle exit velocity: average of values within 97% of max at station 0 ---
z0_station = z_vals[0]
r0, v0 = stations[z0_station]
v_max_0 = np.max(v0)
core_vals_0 = v0[v0 >= 0.97 * v_max_0]
u_exit = np.mean(core_vals_0)

print(f"Nozzle exit velocity (station z={z0_station}): {u_exit:.4f}")