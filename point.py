import math
import numpy as np
from scipy import stats

class point:
    def __init__(self, radial_pos, axial_pos, voltage_data):
        self.radial = radial_pos
        self.axial = axial_pos
        self.voltage_arr = voltage_data
        self.voltage_mean = self.__mean_voltage() 
        self.velocity_arr = self.__voltage_arr_to_velocity_arr()
        self.velocity_mean = self.__mean_velocity() 
        self.velocity_std = self.__std_velocity()
        self.velocity_rmsf = self.__rms_fluctuations()
        self.velocity_turb_int = self.__turbulence_intensity()
        self.velocity_skewness = self.__skewness()
        self.velocity_kurtosis = self.__kurtosis()

    def __voltage_arr_to_velocity_arr(self):
        velocity_arr = np.zeros(len(self.voltage_arr))

        velocity_arr = self.__voltage_to_velocity(self.voltage_arr)
        return velocity_arr

    def __voltage_to_velocity(self, voltage):
        velocity = -15.577001774682708+38.2584093*voltage-23.36414635*voltage**2+0.35160582*voltage**3+1.97694339*voltage**4
        return velocity

    def __mean_voltage(self):
        mean_voltage = self.voltage_arr.mean()
        return mean_voltage

    def __mean_velocity(self):
        mean_velocity = self.velocity_arr.mean()
        return mean_velocity

    def __std_velocity(self):
        std_velocity = self.velocity_arr.std()
        return std_velocity

    def __rms_fluctuations(self):
        velocity_error = np.zeros(len(self.velocity_arr))
        velocity_error = (self.velocity_arr - self.velocity_mean )**2
        rms_fluctuations = math.sqrt((velocity_error.sum())/len(self.velocity_arr))
        return rms_fluctuations
    
    def __turbulence_intensity(self):
        turbulence_intensity = self.__rms_fluctuations() / self.velocity_mean
        return turbulence_intensity
    
    def __skewness(self):
        velocity_error_nume = np.zeros(len(self.velocity_arr))
        velocity_error_denom = np.zeros(len(self.velocity_arr))
        velocity_error_nume = (self.velocity_arr - self.velocity_mean)**3
        velocity_error_denom = (self.velocity_arr - self.velocity_mean)**2
        skewness = (velocity_error_nume.sum() / len(self.velocity_arr)) / ((velocity_error_denom.sum() / len(self.velocity_arr))**(3/2))
        return skewness
    
    def __kurtosis(self):
        velocity_error_nume = np.zeros(len(self.velocity_arr))
        velocity_error_denom = np.zeros(len(self.velocity_arr))
        velocity_error_nume = (self.velocity_arr - self.velocity_mean)**4
        velocity_error_denom = (self.velocity_arr - self.velocity_mean)**2
        kurtosis = (velocity_error_nume.sum() / len(self.velocity_arr)) / ((velocity_error_denom.sum() / len(self.velocity_arr))**2)
        return kurtosis

# TODO: make correlation matrix