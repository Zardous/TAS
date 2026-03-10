import math
import numpy as np

class point:
    def __init__(self, radial_pos, axial_pos, voltage_data):
        self.radial = radial_pos
        self.axial = axial_pos
        self.voltage_arr = voltage_data
        self.velocity_arr = self.__voltage_arr_to_velocity_arr()
        self.velocity_mean = self.__mean_velocity() 
    
    def __voltage_arr_to_velocity_arr(self):
        velocity_arr = np.array([])

        for i in range(len(self.voltage_arr)):
            velocity_arr = np.append(velocity_arr, self.__voltage_to_velocity(self.voltage_arr[i]))
        return velocity_arr

    def __voltage_to_velocity(self, voltage):
        velocity = voltage + 2
        return velocity

    def __mean_velocity(self):
        mean_velocity = self.velocity_arr.mean()
        return mean_velocity

point1 = point(4, 6, [2,2,3,2,2,2])

print(point1.radial)
print(point1.axial)
print(point1.voltage_arr)
print(point1.velocity_arr)
print(point1.velocity_mean)