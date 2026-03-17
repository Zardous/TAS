import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent / "data" / "position"

FILE_NAMES = {
    "data05": "pos_hw0_5d.dat",
    "data0":   "pos_hw0d.dat",
    "data1":   "pos_hw1d.dat",
    "data2":   "pos_hw2d.dat",
    "data4":   "pos_hw4d.dat",
    "data7":   "pos_hw7d.dat",
    "data8":   "pos_hw8d.dat",
}

data = {}
for key, fname in FILE_NAMES.items():
    with open(BASE_DIR / fname, "r") as f:
        data[key] = f.readlines()

data05, data0, data1, data2, data4, data7, data8 = (
    data["data05"], data["data0"], data["data1"], data["data2"],
    data["data4"],  data["data7"], data["data8"]
)

data05_floats = [float(line.strip()) for line in data05]
data0_floats = [float(line.strip()) for line in data0]
data1_floats = [float(line.strip()) for line in data1]
data2_floats = [float(line.strip()) for line in data2]
data4_floats = [float(line.strip()) for line in data4]
data7_floats = [float(line.strip()) for line in data7]
data8_floats = [float(line.strip()) for line in data8]
len0 = len(data0_floats)
len05 = len(data05_floats)
len1 = len(data1_floats)
len2 = len(data2_floats)
len4 = len(data4_floats)
len7 = len(data7_floats)
len8 = len(data8_floats)


radial_dist = [data0_floats, data05_floats, data1_floats, data2_floats, data4_floats, data7_floats, data8_floats]

fig, ax = plt.subplots()
axial_dist = np.array([0, 6, 12, 24, 48, 84, 96])

for i in range(len(radial_dist)):
    ax.scatter(radial_dist[i], np.full(len(radial_dist[i]),axial_dist[i]))



ax.vlines(0,0,80)
plt.show()

