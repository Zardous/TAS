from pointcloud import * # Also imports all the imports from pointcloud
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines

cloud = PointCloud()
cloud.read_test_data()


#axial_layer = 4
#for i in range(len(cloud.points[axial_layer])):
#    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
#    cloud.points[axial_layer][i].plot_distribution(ax,40)
#    fig.savefig(f'figure_[{axial_layer}][{i}].png')
#    plt.close(fig)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

cloud.points[4][3].plot_distribution(ax1, 40)

current_index = 0
current_layer = 0

def draw(layer, i):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    tha_point = cloud.points[layer][i]
    corr_kl = cloud.pair_correlation(tha_point, tha_point, cloud.correlate_pair_by_convolution)
    ms = np.linspace(0, 5_000, corr_kl.size)
    ax1.set_ylim(-1, 1)
    ax1.plot(ms, corr_kl)

    ax1.set_xlabel('Lag [ms]')
    ax1.set_ylabel('Autocorrelation [-]')
    ax1.set_xlim(0, 20)

    corr_kl, _, _ = cloud.full_cross_correlation(layer, i, cloud.correlate_by_kl_divergence)
    ax2_5 = cloud.plot_2Dcontour_from_array(corr_kl, ax2, transparency=1)

    all_radials = []
    all_axials = []

    for layer_points in cloud.points:
        for p in layer_points:
            all_radials.append(p.radial)
            all_axials.append(p.axial)

    ax2_5.scatter(all_radials, all_axials,
                s=20, facecolors="white", edgecolors="black")
    
    main_point = cloud.points[layer][i]
    ax2_5.scatter(main_point.radial, main_point.axial, s=50, color="red")
        
    ax2_5.set_ylim(-0.01,8.01)

    ax3.plot(ms, main_point.velocity_arr)
    ax3.set_xlabel('Time [ms]')
    ax3.set_xlim(0, 1000)

    # freq, ampls = main_point.spectral_analysis(False)
    # ax4.plot(freq, ampls)
    # ax4.set_xlabel(f'Frequency [Hz]')
    # ax4.set_ylabel(f'Amplitude [m/s]')
    # ax4.set_yscale('log')

    main_point.Kolmogorov(ax4)

    ax1.set_title(f"Layer: {layer}, Point: {i}")

    fig.canvas.draw_idle()

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


"""
def on_key(event):
    global current_index, current_layer

    if event.key == "right":
        current_index = (current_index + 1) % len(cloud.points[current_layer])

    elif event.key == "left":
        current_index = (current_index - 1) % len(cloud.points[current_layer])

    elif event.key == "up":
        current_layer = (current_layer + 1) % len(cloud.points)
        current_index = 0 

    elif event.key == "down":
        current_layer = (current_layer - 1) % len(cloud.points)
        current_index = 0  

    draw(current_layer, current_index)


fig.canvas.mpl_connect("key_press_event", on_key)

draw(current_layer, current_index)
plt.show()

#anim = FuncAnimation(fig, update, frames=len(cloud.points[axial_layer]), interval=200, repeat=True)
#fig.show() 
"""

"""
    def update(i):
    ax1.clear()  
    cloud.points[axial_layer][i].plot_distribution(ax1, 40)
    ax3 = cloud.plot_contour_attr('velocity_mean', ax2)

    main_point=cloud.points[axial_layer][i]
    ax3.scatter(main_point.radial, main_point.axial, s=50, color="red")
"""
# Save as GIF
#anim.save(f'axial_layer_{axial_layer}.gif', writer='pillow')

input()
