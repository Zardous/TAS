import numpy as np
from pointcloud import PointCloud
from point import point
import matplotlib.pyplot as plt

pointcloud_testdata = PointCloud()
pointcloud_testdata.read_test_data()
#pointcloud_testdata.shift_velocities()
for i in range(len(pointcloud_testdata.points)):
    lst = pointcloud_testdata.points[i]
    vel = np.array([p.velocity_mean for p in lst])
    pos = np.array([p.radial for p in lst])  

    print(f'Axial line {i}:')
    halfwidth_index, right_up, left_up, right_down, left_down = pointcloud_testdata.find_halfwidth(vel, pos)
    print(halfwidth_index)
    print(f'Left down: {pos[left_down]}, Left up: {pos[left_up]}, Right up: {pos[right_up]}, Right down: {pos[right_down]}')
    print(f'Left down: {left_down}, Left up: {left_up}, Right up: {right_up}, Right down: {right_down}')
    right_up_max, left_up_max, max, midpoint = pointcloud_testdata.find_mid(vel, pos)
    print(midpoint)
    print(f'Left up pos max: {pos[left_up_max]}, Right up pos max: {pos[right_up_max]}, Midpoint: {midpoint}, Max: {max}')
    print(f'Left up max: {left_up_max}, Right up max: {right_up_max}, Midpoint: {midpoint}, Max: {max}')
    iteration_num = 0
    
    fig, ax = plt.subplots(2, 2)

    plt.scatter(pos, midpoint)
    plt.scatter(iteration_num, max)
    plt.scatter(pos, vel)
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
