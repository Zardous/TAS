from pointcloud import * # Also imports all the imports from pointcloud
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import scipy.signal as sp

cloud = PointCloud()
cloud.read_test_data()


axial_layer = 4
for i in range(len(cloud.points[axial_layer])):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cloud.points[axial_layer][i].plot_distribution(ax,40)
    #fig.savefig(f'figure_[{axial_layer}][{i}].png')
    plt.close(fig)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

cloud.points[4][3].plot_distribution(ax1, 40)

current_index = 10
current_layer = 4

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

    lines = [
        (1/0.0103, 0.3792/0.0103, "Potential Core"),   # potential core
        (-1/0.0121, 0.3875/0.0121, "Potential Core"),  # potential core
        (-1/0.0051, -0.4801/0.0051, "Jet Half Width"),    # pole
        (1/0.0044, -0.5691/0.0044, "Jet Half Width")      # pole
    ]

    # Define color mapping
    color_map = {"Potential Core": "orange", "Jet Half Width": "purple"}

    # Plot all lines
    
    for a, b, category in lines:
        x_vals = np.array(ax3.get_xlim())
        if abs(a) <= 100:
            if a <= 0:
                x_vals[0] = -0.12
            else:
                x_vals[1] = 0.05
        else:
            x_vals = x_vals
        y_vals = (a * x_vals + b) / 12

        ax2.plot(x_vals, y_vals, color=color_map[category], linestyle="--", linewidth=2)
        
    nucleus_line = mlines.Line2D([], [], color="orange", linestyle="--", label="Potential Core")
    pole_line = mlines.Line2D([], [], color="purple", linestyle="--", label="Jet Half Width")

    ax2.legend(handles=[nucleus_line, pole_line], loc="upper right")
        
    ax2_5.set_ylim(-0.01,8.01)

    ax3.plot(ms, main_point.velocity_arr)
    ax3.set_ylabel('Velocity [m/s]')
    ax3.set_xlabel('Time [ms]')
    ax3.set_xlim(0, 5000)

    # freq, ampls = main_point.energy_spectrum(False)
    # ax3.plot(freq, ampls)
    # ax3.set_xlabel(f'Frequency [Hz]')
    # ax3.set_yscale('log')
    # ax3.set_ylim(10**-5, 10**0)

    main_point.turbulence_power(ax3)

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

point1 = cloud.points[2][5]
point2 = cloud.points[2][19]
point3 = cloud.points[1][22]
point4 = cloud.points[1][39]
point5 = cloud.points[3][16]
point6 = cloud.points[3][28]
v1 = point1.velocity_arr - point1.velocity_arr.mean()
v2 = point2.velocity_arr - point2.velocity_arr.mean()
v3 = point3.velocity_arr - point3.velocity_arr.mean()
v4 = point4.velocity_arr - point4.velocity_arr.mean()
v5 = point5.velocity_arr - point5.velocity_arr.mean()
v6 = point6.velocity_arr - point6.velocity_arr.mean()

var = (np.std(v1) * np.std(v2) * np.std(v3) * np.std(v4) * np.std(v5) * np.std(v6))**(1/3)
conversion_factor =  5/63
conversion_factor =  5000/100_000
correlate_sim2 = sp.correlate(v1, v2, mode = 'full') / (var * len(v1))
ms2 = sp.correlation_lags(len(v1), len(v2), mode='full')*conversion_factor

correlate_sim1 = sp.correlate(v3, v4, mode = 'full') / (var * len(v3))
ms1 = sp.correlation_lags(len(v3), len(v4), mode='full')*conversion_factor

correlate_sim3 = sp.correlate(v5, v6, mode = 'full') / (var * len(v5))
ms3 = sp.correlation_lags(len(v5), len(v6), mode='full')*conversion_factor

correlate_coreleft12 = sp.correlate(v1, v3, mode = 'full') / (var * len(v1))
ms12l = sp.correlation_lags(len(v1), len(v3), mode='full')*conversion_factor

correlate_coreleft23 = sp.correlate(v3, v5, mode = 'full') / (var * len(v3))
ms23l = sp.correlation_lags(len(v3), len(v5), mode='full')*conversion_factor

correlate_coreright12 = sp.correlate(v2, v4, mode = 'full') / (var * len(v2))
ms12r = sp.correlation_lags(len(v2), len(v4), mode='full')*conversion_factor

correlate_coreright23 = sp.correlate(v4, v6, mode = 'full') / (var * len(v4))
ms23r = sp.correlation_lags(len(v4), len(v6), mode='full')*conversion_factor

# ms = np.linspace(0, 5_000, correlate_coreright23.size) unused but would be okay

print(ms1.max(), ms2.max(), ms3.max()) # 7936.428571428571 if 5/63 else 5000
print(v2.shape)


#plt.title("Symmetry at Core Boundary")
plt.xlim(0,20)
# plt.xlabel("Samples (63 microseconds per sample)")
plt.xlabel("Lags [ms]")
plt.ylabel("Cross-Correlation [-]")
plt.plot(ms1, correlate_sim1, color='blue', label='Points 1,22 and 1,39')
plt.plot(ms2, correlate_sim2, color='red', label='Points 2,5 and 2,19')
plt.plot(ms3, correlate_sim3, color='green', label='Points 3,16 and 3,28')
plt.legend(loc='lower left', fontsize='small')
plt.grid()
plt.show()

#plt.title("Correlation of Different Layers Along Left Core Boundary")
plt.xlim(0,40)
plt.xlabel("Lags [ms]")
plt.ylabel("Cross-Correlation [-]")
plt.plot(ms12l, correlate_coreleft12, color='blue', label='LHS, Points 2,5 and 1,22')
plt.plot(ms23l, correlate_coreleft23, color='red', label='LHS, Points 3,16 and 2,5')
plt.legend(loc='lower left', fontsize='small')
plt.grid()
plt.show()

#plt.title("Correlation of Different Layers Along Right Core Boundary")
plt.xlim(0,40)
plt.xlabel("Lags [ms]")
plt.ylabel("Cross-Correlation [-]")
plt.plot(ms12r, correlate_coreright12, color='blue', label='RHS, Points 2,19 and 1,39')
plt.plot(ms23r, correlate_coreright23, color='red', label='RHS, Points 3,28 and 2,19')
plt.legend(loc='lower left', fontsize='small')
plt.grid()
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
