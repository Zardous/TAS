from pointcloud import * # Also imports point.py

cloud = PointCloud()
cloud.read_test_data()
cloud.plot('velocity_mean')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
cloud.plot_2D('velocity_mean', [0], ax1)
cloud.plot_2D('velocity_turb_int', [0], ax2)
cloud.plot_2D('velocity_skewness', [0], ax3)
cloud.plot_2D('velocity_kurtosis', [0], ax4)

fig.tight_layout()
fig.show()

input()
