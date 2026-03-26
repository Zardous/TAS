from pointcloud import * # Also imports point.py
import scipy as sp
import matplotlib.tri as tri
import matplotlib.axes
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

cloud = PointCloud()
cloud.read_test_data()

fig, (ax1, ax2) = plt.subplots(1, 2)


corr_kl, main_point, main_val = cloud.full_cross_correlation(4, 25, cloud.correlate_by_kl_divergence)
ax1 = cloud.plot_2Dcontour_from_array(corr_kl, ax1, transparency=1)

if isinstance(ax1, Axes3D): # plot is 3D
    ax1.scatter(main_point.radial, main_point.axial, main_val, color='red') # type: ignore
else: # Plot is 2D
    ax1.scatter(main_point.radial, main_point.axial, color='red')



corr_fb, main_point, main_val = cloud.full_cross_correlation(4, 25, cloud.correlate_by_freq_bins)
ax2 = cloud.plot_2Dcontour_from_array(corr_fb, ax2, transparency=1)

if isinstance(ax2, Axes3D): # plot is 3D
    ax2.scatter(main_point.radial, main_point.axial, main_val, color='red') # type: ignore
else: # Plot is 2D
    ax2.scatter(main_point.radial, main_point.axial, color='red')



fig.tight_layout()
plt.show()

