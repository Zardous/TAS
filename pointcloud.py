try:
    from nptdms import TdmsFile
    import nptdms
except:
    raise RuntimeError(f'Please install nptdms: \npip install npTDMS')

from point import point
import sys, os
import numpy as np

class PointCloud:
    def __init__(self) -> None:
        pass

    def read_test_data(self):
        self.points = []

        for folder_name in os.listdir(os.path.join('data')):
            if not folder_name.startswith('HW'): continue

            current_list = []

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
                data = self.read(os.path.join('data', folder_name, file_name))

                if idx>=radials.size: 
                    print(f'Found a missing position. Ignoring')
                    continue

                radial = radials[idx]

                current_list.append(point(radial_pos=radial, axial_pos=axial, voltage_data=data))

                idx += 1
            if idx != radials.size: raise RuntimeError

            self.points.append(current_list)

        print(f'Done')

    def read_cal_data(self):
        self.points = []

        for folder_name in os.listdir(os.path.join('data')):
            if not folder_name.startswith('C'): continue

            current_list = []

            idx = 0
            for file_name in os.listdir(os.path.join('data', folder_name)):
                if not file_name.endswith('.tdms'): continue
                data = self.read(os.path.join('data', folder_name, file_name))

                current_list.append(point(radial_pos=0., axial_pos=0., voltage_data=data))

                idx += 1

            self.points.append(current_list)

        print(f'Done')

    def read(self, file_path: str):
        file = TdmsFile.read(file_path)
        (group, ) = file.groups()
        group: nptdms.tdms.TdmsGroup
        (channel, ) = group.channels()
        channel: nptdms.tdms.TdmsChannel
        data = channel.raw_data
        return data
