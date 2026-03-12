import matplotlib.pyplot as plt
import numpy as np
from point import point
from pointcloud import PointCloud
from collections import defaultdict


points = []

ptcld = PointCloud()


for p in ptcld.points:
    for i in range(len(p)):
        points.append((p[i].radial, p[i].axial, p[i].velocity_mean))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

profiles = defaultdict(list)

for p in points:
    profiles[p[1]].append(p)

for r,pts in profiles.items(): 
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    z = [p[2] for p in pts]
    ax.plot(x, y, z)

ax.set_xlabel('Radial Position')
ax.set_ylabel('Axial Position')
ax.set_zlabel('Velocity')
plt.show()
