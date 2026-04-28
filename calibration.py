
import numpy as np
from matplotlib import pyplot as plt
from pointcloud import PointCloud
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


ptcld = PointCloud()
ptcld.read_cal_data()

#Hot wire voltages read from the files
#HW_voltage_calibration1:  [1.69504677 1.96513012 2.02282869 2.06378831 2.08783118 2.11210874
 #2.12805589 2.14470475 2.15924293 2.17331288 2.19387913 2.21405078
 #2.22950905 2.25879872 2.28525563 2.30502251 2.32247552 2.33457597
 #2.35556621 2.37291198]
#HW_voltage_calibration2:  [1.69942348 1.94905992 2.02007438 2.05205678 2.08261532 2.10627165
 #2.12594933 2.14418338 2.1575524  2.17067774 2.18448637 2.21371998
 #2.23518805 2.26132232 2.28688577 2.30512208 2.32442563 2.33863592
 #2.35931018 2.37480002]

HW_voltage_calibration1 =[] #[V]
HW_voltage_calibration2 = [] #[V]
list1, list2 = ptcld.points

for p in list1:
    HW_voltage_calibration1.append(p.voltage_mean)
HW_voltage_calibration1.sort()
HW_voltage_calibration1 = np.array(HW_voltage_calibration1)
#print("HW_voltage_calibration1: ", HW_voltage_calibration1)
# np.reshape(HW_voltage_calibration1)

for p in list2:
    HW_voltage_calibration2.append(p.voltage_mean)
HW_voltage_calibration2.sort()
HW_voltage_calibration2 = np.array(HW_voltage_calibration2)
#print("HW_voltage_calibration2: ", HW_voltage_calibration2)
# np.reshape(HW_voltage_calibration2)

#HW_voltage_calibration2 = np.sort(HW_voltage_calibration2).flatten()

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

# Valydine voltage to velocity
#plt.plot(Valydine_voltage, velocity_values)
#plt.show()
# Valydine to velocity interpolation
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

'''plt.plot(Valydine_voltage, phi(Valydine_voltage, Valydine_voltage, velocity_values, lagrange_basis_func)) 
plt.plot(Valydine_voltage, velocity_values, 'o')   
plt.show()'''

# calibration Valydine to velocity
velocity_cal1 = phi(VALYDINE_voltage_calibration1, Valydine_voltage, velocity_values, lagrange_basis_func)
velocity_cal2 = phi(VALYDINE_voltage_calibration2, Valydine_voltage, velocity_values, lagrange_basis_func)

'''plt.plot(Valydine_voltage, velocity_values, 'o', label="known data")
plt.plot(VALYDINE_voltage_calibration1, velocity_cal1, 'x', label="calibration 1")
plt.plot(VALYDINE_voltage_calibration2, velocity_cal2, 's', label="calibration 2")
plt.xlabel("Valydine Voltage [V]")
plt.ylabel("Velocity [m/s]")
plt.legend()
plt.show()'''

# HWA to velocity
#HW_voltage_calibration1 = HW_voltage_calibration1.tolist()
#HW_voltage_calibration2 = HW_voltage_calibration2.tolist()

voltage = np.append(HW_voltage_calibration1,  HW_voltage_calibration2)
velocity = np.append(velocity_cal1, velocity_cal2)

'''print("velocity:", velocity)
print("voltage:", voltage)
plt.plot(voltage, velocity)  
plt.scatter(voltage, velocity)
plt.show()'''

def model(x,y):

    model = LinearRegression()
    degree = PolynomialFeatures(degree=4)
    polynomial = degree.fit_transform(x.reshape(-1,1))
    model.fit(polynomial,y)

    return model

poly = model(voltage, velocity)

#voltage = voltage.reshape(-1, 1)
#voltage_poly = PolynomialFeatures.transform(voltage)
degree = PolynomialFeatures(degree=4)
test = np.linspace(0,2.5,50)
voltage_5D = degree.fit_transform(test.reshape(-1,1)) 
v_pred = poly.predict(voltage_5D)
print(poly.coef_)
print(poly.intercept_)
#print(r2_score(velocity, v_pred))

def Kings(HW):
    A=2.4746
    B=1.1525
    n=0.4194
    velo = np.maximum(0, HW**2 - A) / B
    return velo**(1/n)

v_king = Kings(test)

plt.scatter(voltage, velocity, color = "green")      
plt.plot(test, v_pred, color = "red", label = "Polynomial Fit")
plt.title("Polynomial Fit")
plt.xlabel("Hot Wire Voltage [V]")
plt.ylabel("Velocity [m/s]")
plt.xlim(left=1.5)
plt.ylim(bottom=0, top=11.5)
plt.show()

plt.scatter(voltage, velocity, color = "green")   
plt.plot(test, v_king, color = "blue", label = "King's Law")
plt.title("King's Law Fit")
plt.xlabel("Hot Wire Voltage [V]")
plt.ylabel("Velocity [m/s]")
plt.xlim(left=1.5)
plt.ylim(bottom=0, top=11.5)
plt.show()
   
#def poly_func(x, y):
#    coeffs = np.polyfit(x,y,10)
#    return coeffs
       

#print("COEFFICIENTS:",poly_func(Valydine_voltage, velocity_values))
#my try
#my_coeffs = poly_func(Valydine_voltage, velocity_values)
#fitted_poly = np.poly1d(my_coeffs)
#x_smooth = np.linspace(min(Valydine_voltage), max(Valydine_voltage), 200)
#y_smooth = fitted_poly(x_smooth)
#plt.scatter(Valydine_voltage, velocity_values, label="Data Points", color='black')
#plt.plot(x_smooth, y_smooth, label="10th Degree Fit", color='red')
#plt.legend()
#plt.show()