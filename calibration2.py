import numpy as np
from matplotlib import pyplot as plt
from pointcloud import PointCloud
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

ptcld = PointCloud()
ptcld.read_cal_data()

HW_voltage_calibration1 =[] #[V]
HW_voltage_calibration2 = [] #[V]
list1, list2 = ptcld.points


for p in list1:
    HW_voltage_calibration1.append(p.voltage_mean)
HW_voltage_calibration1.sort()
HW_voltage_calibration1 = np.array(HW_voltage_calibration1)


for p in list2:
    HW_voltage_calibration2.append(p.voltage_mean)
HW_voltage_calibration2.sort()
HW_voltage_calibration2 = np.array(HW_voltage_calibration2)


# data and general values
g = 9.80665
water_density = 1000 #[kg/m^3]
air_density = 1.1977 #[kg/m^3]
water_column_height = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #[mmH2O] 
Valydine_voltage = [0.001, 0.065, 0.119, 0.182, 0.244, 0.299, 0.367, 0.424, 0.482, 0.535, 0.598] #[V]



VALYDINE_voltage_calibration1  = [0.002, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.091, 0.110, 0.131, 0.150, 0.191, 0.235, 0.274, 0.313, 0.343, 0.399, 0.454] #[V]

VALYDINE_voltage_calibration2 = [0.002, 0.011, 0.021, 0.029, 0.040, 0.050, 0.06, 0.071, 0.080, 0.09, 0.1, 0.13, 0.154, 0.190, 0.234, 0.272, 0.312, 0.349, 0.402, 0.451] #[V]


# pressure to velocity

def p_to_v(column_height):
    pressure = g * column_height # [Pa]
    velocity=np.sqrt(2*pressure/air_density) # [m/s]
    return velocity

velocity_values = []
for i in water_column_height:
    velocity_values.append(p_to_v(i))



#print(HW_voltage_calibration1)
#print(HW_voltage_calibration2)


#Polynomial to fit the velocities from water column to val voltages
m = 10  #king looks best with m=10 or m=11
coeffs = np.polyfit(Valydine_voltage, velocity_values, m)

p=np.poly1d(coeffs)

#values for voltages to create the curve
voltages_test = np.linspace(0, 0.6, 100)



#velocities for calibration
velo_cal1 = p(VALYDINE_voltage_calibration1)
veloc_cal2 = p(VALYDINE_voltage_calibration2)


#King's Law
def kings_law(E, A, B, n):
    # Standard form: U = ((E^2 - A) / B)^(1/n)
    return ((E**2 - A) / B)**(1/n)

#making the fit
initial_guess = [0.5, 0.5, 0.45] 
popt, pcov = curve_fit(kings_law, HW_voltage_calibration1, velo_cal1, p0=initial_guess)

#Best fitting coefficients
A_best, B_best, n_best = popt


#show coeffs
print(f"Best coefficients: A={A_best:.4f}, B={B_best:.4f}, n={n_best:.4f}")

print(f"Hot Wire Voltages 1: {HW_voltage_calibration1}")
print(f"Hot Wire Voltages 2: {HW_voltage_calibration2}")



#Plotting
#plt.scatter(Valydine_voltage, velocity_values, label='data')
#plt.plot(voltages_test, p(voltages_test), label='poly curve')

def v_to_u_func(E_array, A, B, n):

    velo = np.maximum(0, E_array**2 - A) / B

    return velo**(1/n)

velo_test = v_to_u_func(HW_voltage_calibration1, A_best, B_best, n_best)

plt.scatter(HW_voltage_calibration1, velo_test, label='data')
plt.plot(HW_voltage_calibration1, velo_cal1, label='king function')


plt.show()