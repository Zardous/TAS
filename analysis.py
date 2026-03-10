try:
    from nptdms import TdmsFile
except:
    raise RuntimeError(f'Please install nptdms: \npip install npTDMS')

import sys, os

path = os.path.join('data', 'HW0D', 'HW0D (1).tdms')
file = TdmsFile.read(path)
dataframe = file.as_dataframe()
group = file


# print(file.properties)
# group = file['HW0D (1)']
# print(group.properties)