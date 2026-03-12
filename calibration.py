#imports
import numpy as np
from matplotlib import pyplot as plt
from pointcloud import PointCloud

ptcld = PointCloud()
ptcld.read_cal_data()

HW_voltage_calibration1 =[] #[V]
HW_voltage_calibration2 = [] #[V]
list1, list2 = ptcld.points
for p in list1:
    HW_voltage_calibration1.append(p.voltage_mean)
HW_voltage_calibration1 = np.array(HW_voltage_calibration1)
for p in list2:
    HW_voltage_calibration2.append(p.voltage_mean)
HW_voltage_calibration2 = np.array(HW_voltage_calibration2)

print(HW_voltage_calibration1, HW_voltage_calibration2)

# data and general values
g = 9.80665
water_density = 1000 #[kg/m^3]
air_density = 1.1977 #[kg/m^3]
water_column_height = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #[mmH2O] 
Valydine_voltage = [0.001, 0.065, 0.119, 0.182, 0.244, 0.299, 0.367, 0.424, 0.482, 0.535, 0.598] #[V]

VALYDINE_voltage_calibration1  = [0.002, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.091, 0.110, 0.131, 0.150, 0.191, 0.235, 0.274, 0.313, 0.343, 0.399, 0.454] #[V]

VALYDINE_voltage_calibration2 = [0.002,  0.011, 0.021, 0.029, 0.040, 0.050, 0.06, 0.071, 0.080, 0.09, 0.1, 0.13, 0.154, 0.190, 0.234, 0.272, 0.312, 0.349, 0.402, 0.451] #[V]


# pressure to velocity

def p_to_v(column_height):
    pressure = g * column_height # [Pa]
    velocity=np.sqrt(2*pressure/air_density) # [m/s]
    return velocity

velocity_values = []
for i in water_column_height:
    velocity_values.append(p_to_v(i))

# Valydine voltage to velocity

plt.plot(Valydine_voltage, velocity_values)
plt.show()

# f: velocity, xi: Valydine:
# x: valydine from calibrations, grid: valydine voltage for which we already know the velocity

def lagrange_basis_func(x, grid):
    n = len(grid)
    l = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                l[i] *= (x-grid[j])/(grid[i]-grid[j])
    return l

def phi(x, grid, f, basis):
    interpolant = np.zeros(len(x), dtype=float)
    
    for k in range(len(x)):
        l_k = basis(x[k], grid)
        interpolant[k] = np.sum(f * l_k)
        
    return interpolant

lagrange_poly = phi(Valydine_voltage, Valydine_voltage, velocity_values, lagrange_basis_func)

plt.plot(Valydine_voltage, phi(Valydine_voltage, Valydine_voltage, velocity_values, lagrange_basis_func)) 
plt.plot(Valydine_voltage, velocity_values, 'o')   
plt.show()

#chat
velocity_cal1 = phi(VALYDINE_voltage_calibration1, Valydine_voltage, velocity_values, lagrange_basis_func)
velocity_cal2 = phi(VALYDINE_voltage_calibration2, Valydine_voltage, velocity_values, lagrange_basis_func)
plt.plot(Valydine_voltage, velocity_values, 'o', label="known data")

plt.plot(VALYDINE_voltage_calibration1, velocity_cal1, 'x', label="calibration 1")
plt.plot(VALYDINE_voltage_calibration2, velocity_cal2, 's', label="calibration 2")

plt.xlabel("Valydine Voltage [V]")
plt.ylabel("Velocity [m/s]")
plt.legend()
plt.show()

