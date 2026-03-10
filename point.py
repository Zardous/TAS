import math
import numpy as np

class point:
    def __init__(self, radial_pos, axial_pos, voltage_data):
        self.radial = radial_pos
        self.axial = axial_pos
        self.voltage_arr = voltage_data
        self.velocity_arr = self.__voltage_arr_to_velocity_arr()
        self.velocity_mean = self.__mean_velocity() 
        self.velocity_std = self.__std_velocity()
        self.velocity_rmsf = self.__rms_fluctuations()
        self.velocity_turb_int = self.__turbulence_intensity()
        self.velocity_skewness = self.__skewness()
        self.velocity_kurtosis = self.__kurtosis()

    def __voltage_arr_to_velocity_arr(self):
        velocity_arr = np.zeros(len(self.voltage_arr))

        for i in range(len(self.voltage_arr)):
            velocity_arr[i] = self.__voltage_to_velocity(self.voltage_arr[i])
        return velocity_arr

    def __voltage_to_velocity(self, voltage):
        # Insert polynomial that was found from calibration
        velocity = voltage + 2
        return velocity

    def __mean_velocity(self):
        mean_velocity = self.velocity_arr.mean()
        return mean_velocity

    def __std_velocity(self):
        std_velocity = self.velocity_arr.std()
        return std_velocity

    def __rms_fluctuations(self):
        velocity_error = np.zeros(len(self.velocity_arr))
        for i in range(len(self.velocity_arr)):
            velocity_error[i] = (self.velocity_arr[i] - self.velocity_mean )**2
        rms_fluctuations = math.sqrt((velocity_error.sum())/len(self.velocity_arr))
        return rms_fluctuations
    
    def __turbulence_intensity(self):
        turbulence_intensity = self.__rms_fluctuations() / self.velocity_mean
        return turbulence_intensity
    
    def __skewness(self):
        velocity_error_nume = np.zeros(len(self.velocity_arr))
        velocity_error_denom = np.zeros(len(self.velocity_arr))
        for i in range(len(self.velocity_arr)):
            velocity_error_nume[i] = (self.velocity_arr[i] - self.velocity_mean)**3
        for j in range(len(self.velocity_arr)):
            velocity_error_denom[j] = (self.velocity_arr[j] - self.velocity_mean)**2
        skewness = (velocity_error_nume.sum() / len(self.velocity_arr)) / ((velocity_error_denom.sum() / len(self.velocity_arr))**(3/2))
        return skewness
    
    def __kurtosis(self):
        velocity_error_nume = np.zeros(len(self.velocity_arr))
        velocity_error_denom = np.zeros(len(self.velocity_arr))
        for i in range(len(self.velocity_arr)):
            velocity_error_nume[i] = (self.velocity_arr[i] - self.velocity_mean)**4
        for j in range(len(self.velocity_arr)):
            velocity_error_denom[j] = (self.velocity_arr[j] - self.velocity_mean)**2
        kurtosis = (velocity_error_nume.sum() / len(self.velocity_arr)) / ((velocity_error_denom.sum() / len(self.velocity_arr))**2)
        return kurtosis