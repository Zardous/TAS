import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.axes as axes

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
    
        velocity = np.where(
            voltage > 1.6,
            -15.577001774682708
            + 38.2584093 * voltage
            - 23.36414635 * voltage**2
            + 0.35160582 * voltage**3
            + 1.97694339 * voltage**4,
            0
        )
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
    
    def plot_distribution(self, ax, bin_number, color_code='blue'):
        ax.set_title(f"Point at Axial Dist: {self.axial} Radial Dist: {self.radial:0.2f}")
        ax.set_xlabel("Velocity [m/s]")
        ax.set_xlim(0,12)
        ax.set_ylim(0,0.3)
        ax.set_ylabel('Occurance Frequency []')
        ax.axvline(x=self.velocity_mean)

        counts, bin_edges = np.histogram(self.velocity_arr, bins=bin_number, range=(0, 12))
  
        counts = counts/counts.sum()

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.bar(bin_centers, counts, width=np.diff(bin_edges), color = color_code, alpha = 0.5, edgecolor='black')
        
        ax.text(
            0.01, 0.95,
            f"Radial Pos: {self.radial:.2f} m",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top'
        )

        ax.text(
            0.01, 0.90,
            f"Axial Pos: {self.axial:.2f} m",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top'
        )

        ax.text(
            0.01, 0.85,
            f"Standard Deviation: {self.velocity_std:.2f}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top'
        )

        ax.text(
            0.01, 0.80,
            f"Skewness: {self.velocity_skewness:.2f}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top'
        )

        ax.text(
            0.01, 0.75,
            f"Kurtosis: {self.velocity_kurtosis:.2f}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top'
        )

        ax.text(
            self.velocity_mean,          # x position = the vertical line
            ax.get_ylim()[1]*0.5,            # y position = top of y-axis
            f"Mean Velocity:", 
            verticalalignment='bottom',  # starts at the top
            horizontalalignment='left', # aligns to the line
            fontsize=14,
            color='black'
        )

        ax.text(
            self.velocity_mean,          # x position = the vertical line
            ax.get_ylim()[1]*0.45,            # y position = top of y-axis
            f"{self.velocity_mean:.2f} m/s", 
            verticalalignment='bottom',  # starts at the top
            horizontalalignment='left', # aligns to the line
            fontsize=14,
            color='black'
        )

        return ax

