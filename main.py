from pointcloud import * # Also imports all the imports from pointcloud
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines

cloud = PointCloud()
cloud.read_test_data()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

cloud.plot_2D('velocity_mean', None, ax1)
cloud.plot_2D('velocity_turb_int', None, ax2)
cloud.plot_2D('velocity_skewness', None, ax3)
cloud.plot_2D('velocity_skewness', None, ax3)

#cloud.plot_2D('velocity_mean', [0], ax4, True)
#ax4 = cloud.plot_surface_attr('velocity_mean', ax4)

#fig.tight_layout()
fig.show()

#axial_layer = 4
#for i in range(len(cloud.points[axial_layer])):
#    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
#    cloud.points[axial_layer][i].plot_distribution(ax,40)
#    fig.savefig(f'figure_[{axial_layer}][{i}].png')
#    plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

cloud.points[4][3].plot_distribution(ax1, 40)

current_index = 0
current_layer = 0

def draw(layer, i):
    ax1.clear()
    ax2.clear()

    cloud.points[layer][i].plot_distribution(ax1, 40)

    ax3 = cloud.plot_contour_attr('velocity_mean', ax2)

    all_radials = []
    all_axials = []

    for layer_points in cloud.points:
        for p in layer_points:
            all_radials.append(p.radial)
            all_axials.append(p.axial)

    ax3.scatter(all_radials, all_axials,
                s=20, facecolors="white", edgecolors="black")
    
    main_point = cloud.points[layer][i]
    ax3.scatter(main_point.radial, main_point.axial, s=50, color="red")

  # List of (a, b, category)
    lines = [
        (1/0.0103, 0.3792/0.0103, "Potential Core"),   # potential core
        (-1/0.0121, 0.3875/0.0121, "Potential Core"),  # potential core
        (-1/0.0051, -0.4801/0.0051, "Jet Half Width"),    # pole
        (1/0.0044, -0.5691/0.0044, "Jet Half Width")      # pole
    ]

    # Define color mapping
    color_map = {"Potential Core": "red", "Jet Half Width": "orange"}

    # Plot all lines
    
    for a, b, category in lines:
        x_vals = np.array(ax3.get_xlim())
        #if lines[category] == "Potential Core":
        #    if lines[a] <= 0:
        #        x_vals = [x for x in x_vals if x <=0.5]
        #    else:
        #        x_vals = [x for x in x_vals if x >=-0.5]
        #else:
        #    x_vals = x_vals
        y_vals = (a * x_vals + b) / 12

        ax3.plot(x_vals, y_vals, color=color_map[category], linestyle="--", linewidth=2)
        
    nucleus_line = mlines.Line2D([], [], color="red", linestyle="--", label="Potential Core")
    pole_line = mlines.Line2D([], [], color="orange", linestyle="--", label="Jet Half Width")

    ax3.legend(handles=[nucleus_line, pole_line], loc="upper right")

    ax3.set_ylim(-1,9)


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
