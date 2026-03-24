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

class PointCloud:
    def __init__(self) -> None:
        pass

    def read_test_data(self):
        self.points: list[list[point]] = []

        for folder_name in os.listdir(os.path.join('data')):
            if not folder_name.startswith('HW'): continue

            current_list: list[point] = []

            idx_name = folder_name[-2]
            if idx_name=='5': idx_name='0_5'
            axial = float(idx_name)
            if axial==5.0: axial=0.5

            file_dir = os.path.join('data', 'position', f'pos_hw{idx_name}d.dat')
            with open(file_dir) as file:
                lines = file.readlines()
                radials = np.array([float(p.strip()) for p in lines])

            idx = 0
            for file_name in os.listdir(os.path.join('data', folder_name)):
                if not file_name.endswith('.tdms'): continue
                data = self.__read_file(os.path.join('data', folder_name, file_name))

                if idx>=radials.size: 
                    print(f'Found a missing position. Ignoring')
                    continue

                radial = radials[idx]

                current_list.append(point(radial_pos=radial, axial_pos=axial, voltage_data=data))

                idx += 1
            if idx != radials.size: raise RuntimeError

            self.points.append(current_list)

        zerofive = self.points.pop(0)
        self.points.insert(1, zerofive)

        print(f'Done')
        self.__shift_velocities()
        self.__filter()

    def read_cal_data(self):
        self.points = []

        for folder_name in os.listdir(os.path.join('data')):
            if not folder_name.startswith('C'): continue

            current_list = []

            idx = 0
            for file_name in os.listdir(os.path.join('data', folder_name)):
                if not file_name.endswith('.tdms'): continue
                data = self.__read_file(os.path.join('data', folder_name, file_name))

                current_list.append(point(radial_pos=0., axial_pos=0., voltage_data=data))

                idx += 1

            self.points.append(current_list)

        print(f'Done')

    def __read_file(self, file_path: str):
        file = TdmsFile.read(file_path)
        (group, ) = file.groups()
        group: nptdms.tdms.TdmsGroup
        (channel, ) = group.channels()
        channel: nptdms.tdms.TdmsChannel
        data = channel.raw_data
        return data

    def __check_for_filter(self, array: np.ndarray):
        left = array.copy()
        left[:-1] = array[1:]
        right = array.copy()
        right[1:] = right[:-1]
        mid = (left+right)/2
        out = np.where(0.9<array/mid, True, False)
        return out
    
    def __check_for_tail_filter(self, array: np.ndarray):
        max_val = np.max(array)
        max_idx = np.argmax(array)
        left = np.where((array / max_val < 0.05) & (np.arange(len(array)) < max_idx))[0]
        right = np.where((array / max_val < 0.05) & (np.arange(len(array)) > max_idx))[0]
        
        print(f"left[-1]: {left[-1]}")
        print(f"right[0]: {right[0]}")
        return 
    
    def find_halfwidth(self, vel: np.ndarray, pos: np.ndarray):
        max = np.max(vel)
        over_half = np.where(vel/max>=0.5)
        right_up = over_half[0][-1]
        left_up = over_half[0][0]
        right_down = right_up+1
        left_down = left_up-1
        indice_over_half = np.where(vel/max<0.55)
        indice_under_half = np.where(vel/max>0.45)
        indice_half = np.intersect1d(indice_under_half, indice_over_half)
        print(f'Half width at: {pos[indice_half]}')
        return indice_half, right_up, left_up, right_down, left_down

    def find_mid(self, vel: np.ndarray, pos: np.ndarray):
        max = np.max(vel)
        over_half, = np.where(vel/max>=0.97)
        right_up_max = over_half[-1]
        left_up_max = over_half[0]
        midpoint = (pos[right_up_max] + pos[left_up_max])/2

        return right_up_max, left_up_max, midpoint, max
    
    def find_core(self, vel: np.ndarray, pos: np.ndarray):
        if not hasattr(self, '_max_val'):
            self._max_val = np.max(vel)
        over_half, = np.where(vel / self._max_val >= 0.97)
        right_up_max_ = over_half[-1]
        left_up_max_ = over_half[0]
        return right_up_max_, left_up_max_
        
    def __shift_velocities(self):
        for lst in self.points:
            _,_,mid,_ = self.find_mid(np.array([p.velocity_mean for p in lst]), np.array([p.radial for p in lst]))
            for p in lst:
                p.radial -= mid
        return 
    
    def __filter(self):
        for lst in self.points:
            tmp = []
            vels = np.array([p.velocity_mean for p in lst])
            check = self.__check_for_filter(vels)

            for p, c in zip(lst, check):
                if c and p.velocity_kurtosis<2000:
                    tmp.append(p)

            lst.clear()
            lst.extend(tmp)

            self.__check_for_tail_filter(vels)

        return 
    
    def correlate(self, axial_idx: int, radial_idx: int, attribute) -> tuple[np.ndarray, point, np.ndarray]:
        main_pt = self.points[axial_idx][radial_idx]
        pts = [p for lst in self.points for p in lst]
        idx = pts.index(main_pt)
        arr = np.array([p.__getattribute__(attribute) for lst in self.points for p in lst]) # shape (380, 100000)
        m = arr.mean(axis=1, keepdims=True)
        s = arr.std(axis=1, keepdims=True)

        arr: np.ndarray = (arr-m)/s
        # arr: np.ndarray = (arr-m)/m

        corr = arr@arr.T / (100000-1) # shape (380, 380)
        main_corr_value: np.ndarray = corr[idx, idx]
        return corr[idx], main_pt, main_corr_value

    def plot(self, attribute):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        col = ['#AA0000', '#FF0000', '#FF0078', '#FF00FF', '#7800FF', '#0000FF', '#0000AA']
        for i in range(7): 
            x = np.array([p.radial for p in self.points[i]])
            y = np.array([p.axial for p in self.points[i]])
            z = np.array([p.__getattribute__(attribute) for p in self.points[i]])
            ax.plot(x, y, z, color = col[i])

        ax.set_xlabel('Radial Position')
        ax.set_ylabel('Axial Position')
        ax.set_zlabel('Velocity')
        ax.set_zlim(0)
        plt.show()

    def plot_2D(self, attribute, idx: None|np.ndarray|list, ax):
        suffixes = {'velocity_mean': 'm/s',
                    'velocity_skewness': '-',
                    'velocity_kurtosis': '-',
                    'velocity_std': 'm/s',
                    'velocity_rmsf': 'm/s',
                    'velocity_turb_int': '-'}
        
        col = ['#AA0000', '#FF0000', '#FF0078', '#FF00FF', '#7800FF', '#0000FF', '#0000AA']
        ax.set_title(attribute)
        ax.set_ylabel(suffixes[attribute])
        ax.set_xlabel('x/d')

        if idx==None:
            for i in range(7): 
                x = np.array([p.radial for p in self.points[i]])
                y = np.array([p.__getattribute__(attribute) for p in self.points[i]])
                ax.scatter(x, y, color=col[i], label = str(self.points[i][0].axial))
                
        else:
            for i in idx: 
                x = np.array([p.radial for p in self.points[i]])
                y = np.array([p.__getattribute__(attribute) for p in self.points[i]])
                ax.scatter(x, y, color=col[i])

        ax.legend()
        return ax

    def plot_surface_attr(self, attribute, ax: axes._axes.Axes):
        fig = ax.get_figure()
        assert fig!=None
        ss = ax.get_subplotspec()
        fig.delaxes(ax)
        ax = fig.add_subplot(ss, projection='3d')
        suffixes = {'velocity_mean': 'm/s',
                    'velocity_skewness': '-',
                    'velocity_kurtosis': '-',
                    'velocity_std': 'm/s',
                    'velocity_rmsf': 'm/s',
                    'velocity_turb_int': '-'}
        
        ax.set_title(attribute)
        ax.set_ylabel(suffixes[attribute])
        ax.set_xlabel('x/d')

        x = np.array([p.radial for lst in self.points for p in lst])
        y = np.array([p.axial for lst in self.points for p in lst])
        z = np.array([p.__getattribute__(attribute) for lst in self.points for p in lst])
        ax.plot_trisurf(x, y, z, antialiased=False, edgecolor='none', cmap='viridis')
        return ax
    
    def plot_surface_array(self, arr, ax: axes._axes.Axes):
        fig = ax.get_figure()
        assert fig!=None
        ss = ax.get_subplotspec()
        fig.delaxes(ax)
        ax = fig.add_subplot(ss, projection='3d')
        suffixes = {'velocity_mean': 'm/s',
                    'velocity_skewness': '-',
                    'velocity_kurtosis': '-',
                    'velocity_std': 'm/s',
                    'velocity_rmsf': 'm/s',
                    'velocity_turb_int': '-'}
        
        ax.set_xlabel('x/d')

        x = np.array([p.radial for lst in self.points for p in lst])
        y = np.array([p.axial for lst in self.points for p in lst])
        z = arr
        ax.plot_trisurf(x, y, z, antialiased=False, edgecolor='none', cmap='viridis')
        return ax
    
    def plot_contour(self, attribute, ax: axes._axes.Axes):
        suffixes = {'velocity_mean': 'm/s',
                    'velocity_skewness': '-',
                    'velocity_kurtosis': '-',
                    'velocity_std': 'm/s',
                    'velocity_rmsf': 'm/s',
                    'velocity_turb_int': '-'}
        
        ax.set_title(attribute)
        ax.set_ylabel('axial distance')
        ax.set_xlabel('x/d')

        x = np.array([p.radial for lst in self.points for p in lst])
        y = np.array([p.axial for lst in self.points for p in lst])
        z = np.array([p.__getattribute__(attribute) for lst in self.points for p in lst])

        triang = tri.Triangulation(x.flatten(), y.flatten())
        
        highest = z.max()
        cont = ax.tricontour(triang, z.flatten(), levels=[0.2*highest, 0.4*highest, 0.6*highest, 0.8*highest, 0.97*highest], colors="#000000FF")
        cf2 = ax.tricontourf(triang, z.flatten(), levels=50, cmap='viridis', vmin=0, vmax=z.max())

        ax.legend()
        return ax
