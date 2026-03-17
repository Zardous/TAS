import numpy as np
from pointcloud import PointCloud
from point import point
import matplotlib.pyplot as plt

pointcloud_testdata = PointCloud()
pointcloud_testdata.read_test_data()
pointcloud_testdata.shift_velocities()
print(tuple(len(p) for p in pointcloud_testdata.points))

pointcloud_testdata.filter()

print(tuple(len(p) for p in pointcloud_testdata.points))

def plot(attribute, idx: None|np.ndarray|list):
    suffixes = {'velocity_mean': 'm/s',
                'velocity_skewness': '-'}
    
    plt.title(attribute)
    for i in range(7): 
        x = np.array([p.radial for p in pointcloud_testdata.points[i]])
        y = np.array([p.__getattribute__(attribute) for p in pointcloud_testdata.points[i]])
        # y = pointcloud_testdata.__check_for_filter(y)
        # x = x - pointcloud_testdata.find_mid(y, x)[2]

        plt.vlines(0.0,0,15, colors='black')
        plt.plot(x, y)
        plt.ylim(0, 12)

    plt.ylabel(suffixes[attribute])
    plt.xlabel('x/d')

    plt.show()



plot("velocity_mean", [0, 1, 2, 3])

# for j in range(len(pointcloud_testdata.points)):
#     mean_velocity = np.zeros([len(pointcloud_testdata.points[j])])
#     radial_pos = np.zeros([len(pointcloud_testdata.points[j])])
#     for i, point in enumerate(pointcloud_testdata.points[j]):
#         if point.velocity_mean is not np.nan:
#             mean_velocity[i] = point.velocity_mean
#             radial_pos[i] = point.radial

#     #plt.plot(radial_pos, mean_velocity)
#     plt.scatter(radial_pos, mean_velocity)

# plt.title("Mean Velocity")
# plt.show()
