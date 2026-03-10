import matplotlib.pyplot as plt
import numpy as np

with open("data\position\pos_hw0_5d.dat", "r") as file05:
    data05 = file05.readlines()
with open("data\position\pos_hw0d.dat", "r") as file0:
    data0 = file0.readlines()
with open("data\position\pos_hw1d.dat", "r") as file1:
    data1 = file1.readlines()
with open("data\position\pos_hw2d.dat", "r") as file2:
    data2 = file2.readlines()
with open("data\position\pos_hw4d.dat", "r") as file4:
    data4 = file4.readlines()
with open("data\position\pos_hw7d.dat", "r") as file7:
    data7 = file7.readlines()
with open("data\position\pos_hw8d.dat", "r") as file8:
    data8 = file8.readlines()

data05_floats = [float(line.strip()) for line in data05]
data0_floats = [float(line.strip()) for line in data0]
data1_floats = [float(line.strip()) for line in data1]
data2_floats = [float(line.strip()) for line in data2]
data4_floats = [float(line.strip()) for line in data4]
data7_floats = [float(line.strip()) for line in data7]
data8_floats = [float(line.strip()) for line in data8]

radial_dist = [data0_floats, data05_floats, data1_floats, data2_floats, data4_floats, data7_floats, data8_floats]

fig, ax = plt.subplots()
axial_dist = np.array([0, 5, 10, 20, 40, 70, 80])

for i in range(len(radial_dist)):
    ax.scatter(radial_dist[i], np.full(len(radial_dist[i]),axial_dist[i]))

ax.vlines(0,0,80)
plt.show()
