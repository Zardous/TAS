import numpy as np
from pointcloud import PointCloud
from point import point
import matplotlib.pyplot as plt

cloud = PointCloud()
cloud.read_test_data()

n = len(cloud.points)

fig, ((ax1, ax2)) = plt.subplots(1, 2)
cloud.plot_2Dgraph_from_attr_name('velocity_mean', None, ax1)
cloud.plot_2Dgraph_from_attr_name('velocity_norm', None, ax2)

#mass_flux, momentum_flux, energy_flux = pointcloud_testdata.flux_integrals(np.array(p.radial for lst in pointcloud_testdata.points for p in lst), np.array(p.velocity_mean for lst in pointcloud_testdata.points for p in lst))


mass_flux_arr = np.zeros(n)
momentum_flux_arr = np.zeros(n)
energy_flux_arr = np.zeros(n)
xi_s = []
f_s = []
axial_dist = np.array([0, 6, 12, 24, 48, 84, 96])


for i in range(n):
    print()
    print(f"Axial position {i}")
    lst = cloud.points[i]
    vel = np.array([p.velocity_mean for p in lst])
    pos = np.array([p.radial for p in lst])  
    
    mf, momf, ef, xi_i, f_i = cloud.flux_integrals(vel, pos)

    mass_flux_arr[i] = mf
    momentum_flux_arr[i] = momf
    energy_flux_arr[i] = ef

    xi_s.append(xi_i)
    f_s.append(f_i)

mass_flux_arr /= mass_flux_arr[0] 
momentum_flux_arr /= momentum_flux_arr[0]
energy_flux_arr /= energy_flux_arr[0]

plt.figure()
plt.plot(axial_dist, mass_flux_arr, label='Mass Flux')
plt.plot(axial_dist, momentum_flux_arr, label='Momentum Flux')
plt.plot(axial_dist, energy_flux_arr, label='Energy Flux')
plt.title("Normalized Flux Integrals")
plt.xlabel("Axial Distance [mm]")
plt.ylabel("Normalized Flux []")
plt.annotate(f'Final mass flux value: {round(mass_flux_arr[-1],1)}', xy = (1,2.95), fontsize = 16, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"))
plt.annotate(f'Final momentum flux value: {round(momentum_flux_arr[-1],2)}', xy = (1,2.7), fontsize = 16, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"))
plt.annotate(f'Final energy flux value: {round(energy_flux_arr[-1],2)}', xy = (1, 2.45), fontsize = 16, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"))
plt.legend()
plt.grid(True)

#plt.figure()
#for i in range(len(xi_s)):
#    plt.plot(xi_s[i], f_s[i], label=f"xi_s {i}")
#plt.ylim(0.95,1.05)
#plt.xlim(0,2)
#plt.legend()
plt.show()
