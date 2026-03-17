import numpy as np
from pointcloud import PointCloud
from point import point
import matplotlib.pyplot as plt

pointcloud_testdata = PointCloud()
pointcloud_testdata.read_test_data()
#pointcloud_testdata.shift_velocities()
iteration_num = 0
for i in range(len(pointcloud_testdata.points)):
    lst = pointcloud_testdata.points[i]
    vel = np.array([p.velocity_mean for p in lst])
    pos = np.array([p.radial for p in lst])  
    empty_LD = np.zeros(len(pointcloud_testdata.points[i]))
    empty_LU = np.zeros(len(pointcloud_testdata.points[i]))
    empty_RD = np.zeros(len(pointcloud_testdata.points[i]))
    empty_RU = np.zeros(len(pointcloud_testdata.points[i]))
    empty_LUM = np.zeros(len(pointcloud_testdata.points[i]))
    empty_RUM = np.zeros(len(pointcloud_testdata.points[i]))
    empty_mid = np.zeros(len(pointcloud_testdata.points[i]))

    print(f'Axial distance {i}:')
    halfwidth_index, right_up, left_up, right_down, left_down = pointcloud_testdata.find_halfwidth(vel, pos)
    print(halfwidth_index)
    
    print(f'Left down: {pos[left_down]}, Left up: {pos[left_up]}, Right up: {pos[right_up]}, Right down: {pos[right_down]}')
    print(f'Left down: {left_down}, Left up: {left_up}, Right up: {right_up}, Right down: {right_down}')
    right_up_max, left_up_max, max, midpoint = pointcloud_testdata.find_mid(vel, pos)
    print(midpoint)
    empty_mid[:] = midpoint
    empty_LD[:] = pos[left_down]
    empty_LU[:] = pos[left_up]
    empty_RD[:] = pos[right_down]
    empty_RU[:] = pos[right_up]
    empty_LUM[:] = pos[left_up_max]
    empty_RUM[:] = pos[right_up_max]
    print(f'Left up pos max: {pos[left_up_max]}, Right up pos max: {pos[right_up_max]}, Midpoint: {midpoint}, Max: {max}')
    print(f'Left up max: {left_up_max}, Right up max: {right_up_max}, Midpoint: {midpoint}, Max: {max}')
    
    
    fig, ax = plt.subplots()
    ax.plot(pos, vel, label="Velocity profile")
    ax.axhline(y=vel[left_up],     color='green', linestyle='--', label="Halfwidth velocity")
    ax.axhline(y=vel[left_up_max], color='blue',  linestyle=':',  label="Max region velocity")
    ax.set_title(f"Axial position {i}")
    ax.set_xlabel("Radial position")
    ax.set_ylabel("Velocity")
    ax.legend()
    plt.show()

    iteration_num += 1


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
