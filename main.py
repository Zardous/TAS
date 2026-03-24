from pointcloud import * # Also imports all the imports from pointcloud
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

cloud = PointCloud()
cloud.read_test_data()
#cloud.plot('velocity_mean')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
cloud.plot_2D('velocity_norm', [4,5,6], ax1)
cloud.plot_2D('velocity_turb_int', None, ax2)
cloud.plot_2D('velocity_skewness', None, ax3)
cloud.plot_2D('velocity_mean', [0], ax4, True)

#fig.tight_layout()
fig.show()

#axial_layer = 4
#for i in range(len(cloud.points[axial_layer])):
#    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
#    cloud.points[axial_layer][i].plot_distribution(ax,40)
#    fig.savefig(f'figure_[{axial_layer}][{i}].png')
#    plt.close(fig)

axial_layer = 4

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

def update(i):
    ax.clear()  
    cloud.points[axial_layer][i].plot_distribution(ax, 40)

anim = FuncAnimation(fig, update, frames=len(cloud.points[axial_layer]), interval=200, repeat=True)
fig.show() 


# Save as GIF
#anim.save(f'axial_layer_{axial_layer}.gif', writer='pillow')

input()
