try:
    from nptdms import TdmsFile
    import nptdms
except:
    raise RuntimeError(f'Please install nptdms: \npip install npTDMS')

from point import point
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.tri as tri
from collections import defaultdict
from typing import Callable
import scipy as sp

# ... (your imports and point class here)

class PointCloud:
    def __init__(self) -> None:
        self.points = []

    def read_test_data(self, filter_and_shift=True):
        # ... (your existing read logic)
        print(f'Done reading data.')

    def plot_volume_flux_development(self):
        """Now correctly placed inside the class"""
        x_coords = []
        q_values = []

        if not self.points:
            print("No data found. Did you run read_test_data()?")
            return

        for station in self.points:
            r = np.array([p.radial_pos for p in station])
            # Ensure your 'point' class has a 'velocity' attribute
            u = np.array([p.velocity for p in station])
            x_val = station[0].axial_pos
            
            idx = np.argsort(r)
            r_s, u_s = r[idx], u[idx]
            
            # Integral calculation
            q_val = 2 * np.pi * sp.integrate.trapezoid(r_s * u_s, x=r_s)
            
            x_coords.append(x_val)
            q_values.append(q_val)

        # Plotting logic remains the same...
        plt.scatter(x_coords, q_values, color='blue')
        plt.show()
        return x_coords, q_values

# --- EXECUTION FLOW ---
if __name__ == "__main__":
    # 1. Create the object
    my_cloud = PointCloud()
    
    # 2. Load the data (This populates self.points)
    my_cloud.read_test_data()
    
    # 3. Call the method using the object
    my_cloud.plot_volume_flux_development()