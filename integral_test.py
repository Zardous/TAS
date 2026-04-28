import numpy as np
from pointcloud import PointCloud
from point import point
import matplotlib.pyplot as plt

pointcloud_testdata = PointCloud()
pointcloud_testdata.read_test_data()

n = len(pointcloud_testdata.points)

#mass_flux, momentum_flux, energy_flux = pointcloud_testdata.flux_integrals(np.array(p.radial for lst in pointcloud_testdata.points for p in lst), np.array(p.velocity_mean for lst in pointcloud_testdata.points for p in lst))


mass_flux_arr = np.zeros(n)
momentum_flux_arr = np.zeros(n)
energy_flux_arr = np.zeros(n)
axial_dist = np.array([0, 6, 12, 24, 48, 84, 96])


for i in range(n):
    lst = pointcloud_testdata.points[i]
    vel = np.array([p.velocity_mean for p in lst])
    pos = np.array([p.radial for p in lst])  
    
    
    mass_flux_arr[i], momentum_flux_arr[i], energy_flux_arr[i] = pointcloud_testdata.flux_integrals(vel, pos)


plt.plot(axial_dist, mass_flux_arr, label='mass flux')
plt.plot(axial_dist, momentum_flux_arr, label='momentum flux')
plt.plot(axial_dist, energy_flux_arr, label='energy flux')
plt.title("flux integrals")
plt.legend()
plt.grid(True)
plt.show()
