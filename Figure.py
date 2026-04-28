from pointcloud import * # Also imports all the imports from pointcloud
import matplotlib.pyplot as plt

# cloud = PointCloud()
# cloud.read_test_data(filter_and_shift=False)

# fig, ((ax1, ax2)) = plt.subplots(1, 2)
# cloud.plot_2Dgraph_from_attr_name('velocity_mean', [0, 3, 6], ax1)

# cloud.read_test_data(filter_and_shift=True)

# cloud.plot_2Dgraph_from_attr_name('velocity_mean', [0, 3, 6], ax2)

# ax1.set_ylabel('Mean velocity [m/s]')
# ax2.set_ylabel('Mean velocity [m/s]')
# ax1.set_ylim(0)
# ax2.set_ylim(0)

# ax1.set_title('Data before filtering and shifting')
# ax2.set_title('Data after filtering and shifting')

# fig.tight_layout()
# fig.show()

# input()


cloud = PointCloud()

fig, ax1 = plt.subplots(1, 1)

cloud.read_test_data(filter_and_shift=True)

cloud.plot_2Dgraph_from_attr_name('velocity_mean', [0,], ax1)

ax1.set_ylabel('Mean velocity [m/s]')
ax1.set_ylim(0)

ax1.set_title('')

fig.tight_layout()
fig.show()

input()