try:
    from nptdms import TdmsFile
    import nptdms
except:
    raise RuntimeError(f'Please install nptdms: \npip install npTDMS')

from point import point
import sys, os
import numpy as np
import matplotlib.pyplot as plt
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
        '''
        Sets points to NaN if they are less than 10% of the average of its neighbours

        At endpoints of the array it uses only one neighbour
        '''
        left = array.copy()
        left[:-1] = array[1:]

        right = array.copy()
        right[1:] = right[:-1]

        mid = (left+right)/2

        out = np.where(0.9<array/mid, True, False)
        # print(f'Warning: filter interpolates data')
        return out
    
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

    def plot(self):
        points = []
        for p in self.points:
            for i in range(len(p)):
                points.append((p[i].radial, p[i].axial, p[i].velocity_mean))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        profiles = defaultdict(list)

        for p in points:
            profiles[p[1]].append(p)

        for r, pts in profiles.items(): 
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            z = np.array([p[2] for p in pts])
            ax.plot(x, y, z)

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
        
        col = ['#C80000', '#FF0040', '#FF00AA', '#FF00FF', '#AA00FF', '#6400FF', '#0000C8']
        ax.set_title(attribute)
        ax.set_ylabel(suffixes[attribute])
        ax.set_xlabel('x/d')

        if idx==None:
            for i in range(7): 
                x = np.array([p.radial for p in self.points[i]])
                y = np.array([p.__getattribute__(attribute) for p in self.points[i]])
                ax.plot(x, y, color=col[i], label = str(self.points[i][0].axial))
                
        else:
            for i in idx: 
                x = np.array([p.radial for p in self.points[i]])
                y = np.array([p.__getattribute__(attribute) for p in self.points[i]])
                ax.plot(x, y, color=col[i])

        ax.legend()
        return ax