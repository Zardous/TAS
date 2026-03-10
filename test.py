try:
    from nptdms import TdmsFile
    import nptdms
except:
    raise RuntimeError(f'Please install nptdms: \npip install npTDMS')

import sys, os
import numpy as np
import matplotlib.pyplot as plt

y0 = []
y05 = []
y1 = []
y2 = []
y4 = []
y7 = []
y8 = []
filenumber = ['0','1','2','4','05','7','8']
for i in range(71):
    path = os.path.join('data', 'HW0D', 'HW0D (' + str(i) + ').tdms')
    file = TdmsFile.read(path)
    (group, ) = file.groups()
    group: nptdms.tdms.TdmsGroup
    (channel, ) = group.channels()
    channel: nptdms.tdms.TdmsChannel
    data = channel.raw_data
    point = round(sum(data)/len(data),5)
    y0.append(point)

for i in range(50):
    path = os.path.join('data', 'HW1D', 'HW1D (' + str(i) + ').tdms')
    file = TdmsFile.read(path)
    (group, ) = file.groups()
    group: nptdms.tdms.TdmsGroup
    (channel, ) = group.channels()
    channel: nptdms.tdms.TdmsChannel
    data = channel.raw_data
    point = round(sum(data)/len(data),5)
    y1.append(point)

for i in range(70):
    path = os.path.join('data', 'HW2D', 'HW2D (' + str(i) + ').tdms')
    file = TdmsFile.read(path)
    (group, ) = file.groups()
    group: nptdms.tdms.TdmsGroup
    (channel, ) = group.channels()
    channel: nptdms.tdms.TdmsChannel
    data = channel.raw_data
    point = round(sum(data)/len(data),5)
    y2.append(point)

for i in range(53):
    path = os.path.join('data', 'HW4D', 'HW4D (' + str(i) + ').tdms')
    file = TdmsFile.read(path)
    (group, ) = file.groups()
    group: nptdms.tdms.TdmsGroup
    (channel, ) = group.channels()
    channel: nptdms.tdms.TdmsChannel
    data = channel.raw_data
    point = round(sum(data)/len(data),5)
    y4.append(point)

for i in range(77):
    path = os.path.join('data', 'HW05D', 'HW05D (' + str(i) + ').tdms')
    file = TdmsFile.read(path)
    (group, ) = file.groups()
    group: nptdms.tdms.TdmsGroup
    (channel, ) = group.channels()
    channel: nptdms.tdms.TdmsChannel
    data = channel.raw_data
    point = round(sum(data)/len(data),5)
    y05.append(point)

for i in range(64):
    path = os.path.join('data', 'HW7D', 'HW7D (' + str(i) + ').tdms')
    file = TdmsFile.read(path)
    (group, ) = file.groups()
    group: nptdms.tdms.TdmsGroup
    (channel, ) = group.channels()
    channel: nptdms.tdms.TdmsChannel
    data = channel.raw_data
    point = round(sum(data)/len(data),5)
    y05.append(point)

for i in range(64):
    path = os.path.join('data', 'HW8D', 'HW8D (' + str(i) + ').tdms')
    file = TdmsFile.read(path)
    (group, ) = file.groups()
    group: nptdms.tdms.TdmsGroup
    (channel, ) = group.channels()
    channel: nptdms.tdms.TdmsChannel
    data = channel.raw_data
    point = round(sum(data)/len(data),5)
    y8.append(point)


plt.plot(y0)
plt.plot(y1)
plt.plot(y2)
plt.plot(y4)
plt.plot(y05)
plt.plot(y7)
plt.plot(y8)
plt.show()