# File to test the point class

from point import *

point1 = point(4,6, [2,2,2,4])
print(point1.radial)
print(point1.axial)
print(point1.voltage_arr)
print(point1.velocity_arr)
print(point1.velocity_mean)
print(point1.velocity_std)
print(point1.velocity_rmsf)
print(point1.velocity_turb_int)
print(point1.velocity_kurtosis)
print(point1.velocity_skewness)