import numpy as np
from pointcloud import PointCloud
from point import point
import matplotlib.pyplot as plt

pointcloud_testdata = PointCloud()
pointcloud_testdata.read_test_data()
#pointcloud_testdata.shift_velocities()
iteration_num = 0
n = len(pointcloud_testdata.points)
axial_dist = np.array([12, 6, 0, 24, 48, 84, 96])
halfwidths = np.zeros(n)
midpoints = np.zeros(n)
max_velocities = np.zeros(n)
max_widths = np.zeros(n)

for i in range(n):
    lst = pointcloud_testdata.points[i]
    vel = np.array([p.velocity_mean for p in lst])
    pos = np.array([p.radial for p in lst])  

    print(f'Axial distance {i}:')
    halfwidth_index, right_up, left_up, right_down, left_down = pointcloud_testdata.find_halfwidth(vel, pos)
    print(halfwidth_index)
    
    print(f'Left down: {pos[left_down]}, Left up: {pos[left_up]}, Right up: {pos[right_up]}, Right down: {pos[right_down]}')
    print(f'Left down: {left_down}, Left up: {left_up}, Right up: {right_up}, Right down: {right_down}')
    right_up_max, left_up_max, max, midpoint = pointcloud_testdata.find_mid(vel, pos)
    print(midpoint)

    print(f'Left up pos max: {pos[left_up_max]}, Right up pos max: {pos[right_up_max]}, Midpoint: {midpoint}, Max: {max}')
    print(f'Left up max: {left_up_max}, Right up max: {right_up_max}, Midpoint: {midpoint}, Max: {max}')
                             
    halfwidths[i] = (pos[right_up] - pos[left_up]) / 2 
    max_widths[i] = (pos[right_up_max] - pos[left_up_max]) / 2
    midpoints[i] = midpoint
    max_velocities[i] = max
    
    plt.scatter(pos, vel, label=f"Velocity profile{i}")
    plt.axhline(y=vel[left_up],     color='green', linestyle='--', label="Halfwidth velocity")
    plt.axhline(y=vel[left_up_max], color='blue',  linestyle=':',  label="Max region velocity")
    plt.title(f"Axial position {i}")
    plt.xlabel("Radial position")
    plt.ylabel("Velocity")
    plt.legend()   
    plt.show()

    iteration_num += 1

plt.scatter(axial_dist, max_widths, label="Max velocity width vs axial distance")
plt.title(f"Axial position {i}")
plt.xlabel("Axial distance")
plt.ylabel("Max velocity width")
plt.legend()   
plt.show()


plt.scatter(axial_dist, halfwidths, label="Jet halfwidth vs axial distance")
plt.title("Jet halfwidth vs Axial distance")
plt.xlabel("Axial distance")
plt.ylabel("Jet halfwidth")
#plt.show()

for j in range(len(pointcloud_testdata.points)):
    mean_velocity = np.zeros([len(pointcloud_testdata.points[j])])
    radial_pos = np.zeros([len(pointcloud_testdata.points[j])])
    for i, point in enumerate(pointcloud_testdata.points[j]):
        if point.velocity_mean is not np.nan:
            mean_velocity[i] = point.velocity_mean
            radial_pos[i] = point.radial
    
    #plt.plot(radial_pos, mean_velocity)
    plt.scatter(radial_pos, mean_velocity)

plt.title("Mean Velocity")
plt.show()
