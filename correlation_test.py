from pointcloud import * # Also imports point.py
import scipy as sp
import matplotlib.tri as tri
import matplotlib.axes
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def scatter_points(layer, i):
    ax1.clear()
    all_radials = []
    all_axials = []
    cloud.points[layer][i].plot_distribution(ax1, 40)
    for layer_points in cloud.points:
        for p in layer_points:
            all_radials.append(p.radial)
            all_axials.append(p.axial)

    ax1.scatter(all_radials,all_axials)

if True: # Run cross correlation. If false it runs auto correlation (code below)
    cloud = PointCloud()
    cloud.read_test_data()

    scatter_points(5,30)
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, ax1 = plt.subplots()

    corr_kl, main_point, main_val = cloud.full_cross_correlation(4, 25, cloud.correlate_by_kl_divergence)
    ax1 = cloud.plot_2Dcontour_from_array(corr_kl, ax1, transparency=1)

    if isinstance(ax1, Axes3D): # plot is 3D
        ax1.scatter(main_point.radial, main_point.axial, main_val, color='red') # type: ignore
    else: # Plot is 2D
        ax1.scatter(main_point.radial, main_point.axial, color='red')


    #corr_fb, main_point, main_val = cloud.full_cross_correlation(4, 25, cloud.correlate_by_freq_bins)
    #ax2 = cloud.plot_2Dcontour_from_array(corr_fb, ax2, transparency=1)

    #if isinstance(ax2, Axes3D): # plot is 3D
    #    ax2.scatter(main_point.radial, main_point.axial, main_val, color='red') # type: ignore
    #else: # Plot is 2D
    #    ax2.scatter(main_point.radial, main_point.axial, color='red')



    fig.tight_layout()
    plt.show()
else:
    cloud = PointCloud()
    cloud.read_test_data()

    # p = cloud.points[1][40]
    # p.spectral_analysis()
    # Power spectral density

    fig, ax1 = plt.subplots(1, 1)

    point1 = cloud.points[1][40]
    point2 = cloud.points[1][40]
    corr_kl = cloud.pair_correlation(point1, point2, cloud.correlate_pair_by_convolution)
    ms = np.linspace(0, 5_000, corr_kl.size)
    ax1.plot(ms, corr_kl)

    ax1.set_xlabel('Lag [ms]')
    ax1.set_ylabel('Autocorrelation [-]')

    fig.tight_layout()
    plt.show()



