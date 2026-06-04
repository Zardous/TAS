from pointcloud import * # Also imports all the imports from pointcloud
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines

cloud = PointCloud()
cloud.read_test_data()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

current_index = 22
current_layer = 1

def draw(layer, i):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    tha_point = cloud.points[layer][i]
    corr_kl = cloud.pair_correlation(tha_point, tha_point, cloud.correlate_pair_by_convolution)
    ms = np.linspace(0, 5_000, corr_kl.size)
    ax1.set_ylim(-0.4, 1)
    ax1.plot(ms, corr_kl)
    ax1.grid(True)

    ax1.set_xlabel('Lag [ms]')
    ax1.set_ylabel('Autocorrelation [-]')
    ax1.set_xlim(0, 20)

    corr_kl, _, _ = cloud.full_cross_correlation(layer, i, cloud.correlate_by_kl_divergence)

    contour = cloud.plot_2Dcontour_from_array(corr_kl, ax2, transparency=1)

    ax2.set_xlabel("Radial distance r/d [-]")
    ax2.set_ylabel("Axial distance x/d [-]")
    ax2.set_xlim(-1.5, 1.5)

    #ax2_5 = cloud.plot_2Dcontour_from_array(corr_kl, ax2, transparency=1)

    #ax2_5.set_xlabel("Radial distance r/d [-]")
    #ax2_5.set_ylabel("Axial distance x/d [-]")
    #ax2_5.set_xlim(-1.5, 1.5)
    #cbar = fig.colorbar(contour, ax=ax2, pad=0.02)
    #cbar.set_label("Correlation Strength")
    vmin = contour.norm.vmin
    vmax = contour.norm.vmax
    #cbar.set_ticks([vmin, 0.25*vmax, 0.5*vmax, 0.75*vmax, vmax])
    #cbar.set_ticklabels([f"{vmin:.2f} (Strong)","6.08","12.16","18.23",f"{vmax:.2f} (Weak)"])

    all_radials = []
    all_axials = []

    for layer_points in cloud.points:
        for p in layer_points:
            all_radials.append(p.radial)
            all_axials.append(p.axial)

    ax2.scatter(all_radials, all_axials,
                s=20, facecolors="white", edgecolors="black")
    
    main_point = cloud.points[layer][i]
    ax2.scatter(main_point.radial, main_point.axial, s=50, color="red")
        
    ax2.set_ylim(-0.01,8.01)

    main_point.turbulence_power(ax3)

    main_point.Kolmogorov(ax4)

    ax1.set_title(f"Layer: {layer}, Point: {i}", y=1.05)

    fig.canvas.draw_idle()

    ax3.grid(True)
    ax3.set_ylim(-5, 2)
    ax3.set_xlim(0, 4000)
    ax4.grid(True)
    ax4.set_ylim(1e-7, 0.1)
    ax4.set_xlim(0, 7000)


def find_closest_index(layer_points, target_radial):
    return min(
        range(len(layer_points)),
        key=lambda j: abs(layer_points[j].radial - target_radial)
    )

def on_key(event):
    global current_index, current_layer

    if event.key == "right":
        current_index = (current_index + 1) % len(cloud.points[current_layer])

    elif event.key == "left":
        current_index = (current_index - 1) % len(cloud.points[current_layer])

    elif event.key == "up":
        current_radial = cloud.points[current_layer][current_index].radial

        current_layer = (current_layer + 1) % len(cloud.points)

        current_index = find_closest_index(cloud.points[current_layer], current_radial)

    elif event.key == "down":
        current_radial = cloud.points[current_layer][current_index].radial

        current_layer = (current_layer - 1) % len(cloud.points)

        current_index = find_closest_index(cloud.points[current_layer], current_radial)

    draw(current_layer, current_index)


fig.canvas.mpl_connect("key_press_event", on_key)

draw(current_layer, current_index)
plt.show()

input()
