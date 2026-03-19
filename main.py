from pointcloud import * # Also imports point.py

cloud = PointCloud()
cloud.read_test_data()
#cloud.plot()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
cloud.plot_2D('velocity_mean', None, ax1)
cloud.plot_2D('velocity_turb_int', None, ax2)
cloud.plot_2D('velocity_skewness', None, ax3)
cloud.plot_2D('velocity_kurtosis', None, ax4)

fig.tight_layout()
fig.show()

input()
