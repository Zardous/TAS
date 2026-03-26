from pointcloud import * # Also imports point.py
import scipy as sp
import matplotlib.tri as tri
import matplotlib.axes
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D


cloud = PointCloud()
cloud.read_test_data()

fig, ax1 = plt.subplots(1, 1)
corr, main_point, main_val = cloud.correlate(6, 40, 'velocity_arr', cloud.kl_divergence)
ax1 = cloud.plot_surface_from_array(corr, ax1)
ax1.scatter(main_point.radial, main_point.axial, main_val, color='red') # type: ignore

fig.tight_layout()
fig.show()

input()
