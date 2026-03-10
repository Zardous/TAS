
lists = ['0', '05', '1', '2', '4', '7', '8']
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from point import point

color_map = {4: "red", 7: "green", 8: "blue"}

point1 = point(4, 4, [2,2,2,2,2,2])
point2 = point(4, 5, [2,2,2,2,2,4])
point3 = point(4, 6, [2,2,2,2,2,6])
point4 = point(7, 4, [2,2,2,2,2,8])
point5 = point(7, 5, [2,2,2,2,2,10])
point6 = point(7, 6, [2,2,2,2,2,12])
point7 = point(8, 4, [2,2,2,2,2,14])
point8 = point(8, 5, [2,2,2,2,2,16])
point9 = point(8, 6, [2,2,2,2,2,18])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for point in [point1, point2, point3, point4, point5, point6, point7, point8, point9]: 
    x = point.axial
    y = point.radial
    z = point.velocity_mean
    
    ax.scatter(x, y, z, color=color_map[y])





ax.plot(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
