try:
    from nptdms import TdmsFile
    import nptdms
except:
    raise RuntimeError(f'Please install nptdms: \npip install npTDMS')

import sys, os
import numpy as np

path = os.path.join('data', 'HW0D', 'HW0D (1).tdms_index')
file = TdmsFile.read(path)

(group, ) = file.groups()
group: nptdms.tdms.TdmsGroup
(channel, ) = group.channels()
channel: nptdms.tdms.TdmsChannel
data = channel.raw_data
print(data)

