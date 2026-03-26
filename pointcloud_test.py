import numpy as np
from pointcloud import PointCloud
from point import point
import matplotlib.pyplot as plt

pointcloud_testdata = PointCloud()
pointcloud_testdata.read_test_data()

for i in range(len(pointcloud_testdata.points)):
    print(pointcloud_testdata.points[i][0].axial)
#pointcloud_testdata.shift_velocities()
iteration_num = 0
n = len(pointcloud_testdata.points)
axial_dist = np.array([0, 6, 12, 24, 48, 84, 96])
halfwidths = np.zeros(n)
midpoints = np.zeros(n)
max_velocities = np.zeros(n)
max_widths = np.zeros(n)

left_halfwidths = np.zeros(n)
right_halfwidths = np.zeros(n)
left_core = np.zeros(n)
right_core = np.zeros(n)

for i in range(n):
    lst = pointcloud_testdata.points[i]
    vel = np.array([p.velocity_mean for p in lst])
    pos = np.array([p.radial for p in lst])  

    #print(f'Axial distance {i}:')
    halfwidth_index, right_up, left_up, right_down, left_down, _, _ = pointcloud_testdata.find_halfwidth(vel, pos)
    #print(halfwidth_index)
    
    #print(f'Positions: Left down: {pos[left_down]}, Left up: {pos[left_up]}, Right up: {pos[right_up]}, Right down: {pos[right_down]}')
    #print(f'Indexes: Left down: {left_down}, Left up: {left_up}, Right up: {right_up}, Right down: {right_down}')
    right_up_max, left_up_max, max, midpoint = pointcloud_testdata.find_mid(vel, pos)
    right_up_max_, left_up_max_ = pointcloud_testdata.find_core(vel, pos)
    #print(midpoint)
    
    
    '''
    print(f'Left up pos max: {pos[left_up_max_]}, Right up pos max: {pos[right_up_max_]}')
    print(f'Left up max: {left_up_max_}, Right up max: {right_up_max_}')
     '''                        
    halfwidths[i] = (pos[right_up] - pos[left_up]) / 2 + midpoint
    
    midpoints[i] = midpoint
    
    if right_up_max_ is not None and left_up_max_ is not None:
        max_widths[i] = (pos[right_up_max_] - pos[left_up_max_]) / 2
    else:
        max_widths[i] = np.nan
    if right_up_max_ is not None and left_up_max_ is not None:
        left_core[i] = pos[left_up_max_]
        right_core[i] = pos[right_up_max_]
    else:
        left_core[i] = np.nan
        right_core[i] = np.nan

    max_velocities[i] = max
    #need to remove the shift of the halfwidths at each axial position to get the pole
    left_halfwidths[i] = (pos[left_up] + pos[left_down])/2 - (pos[left_up] + pos[right_up] + pos[left_down] + pos[right_down])/4
    right_halfwidths[i] = (pos[right_up] + pos[right_down])/2 - (pos[left_up] + pos[right_up] + pos[left_down] + pos[right_down])/4
    '''
    plt.scatter(pos, vel, label=f"Velocity profile{i}")
    plt.axhline(y=vel[left_up],     color='green', linestyle='--', label="Halfwidth velocity")
    plt.axhline(y=vel[left_up_max_], color='blue',  linestyle=':',  label="Max region velocity")
    plt.title(f"Axial position {i}")
    plt.xlabel("Radial position")
    plt.ylabel("Velocity")
    '''

    iteration_num += 1
'''
plt.show()
plt.scatter(axial_dist, max_widths, label="Max velocity width vs axial distance")
plt.title(f"Axial position {i}")
plt.xlabel("Axial distance")
plt.ylabel("Max velocity width") 
plt.show()


plt.scatter(axial_dist, halfwidths, label="Jet halfwidth vs axial distance")
plt.title("Jet halfwidth vs Axial distance")
plt.xlabel("Axial distance")
plt.ylabel("Jet halfwidth")
plt.show()
'''
'''
for j in range(len(pointcloud_testdata.points)):
    mean_velocity = np.zeros([len(pointcloud_testdata.points[j])])
    radial_pos = np.zeros([len(pointcloud_testdata.points[j])])
    for i, point in enumerate(pointcloud_testdata.points[j]):
        if point.velocity_mean is not np.nan:
            mean_velocity[i] = point.velocity_mean
            radial_pos[i] = point.radial
    
    #plt.plot(radial_pos, mean_velocity)
    plt.scatter(radial_pos, mean_velocity)

plt.title("Mean Velocity")
plt.show()
'''
for j in range(len(left_core)):
    plt.scatter(axial_dist[j], left_core[j])
    plt.scatter(axial_dist[j], right_core[j])
plt.title("Potential core radius vs axial distance")
plt.xlabel("Axial distance")
plt.ylabel("Potential core radius")
plt.show()

for j in range(len(left_halfwidths)):
    plt.scatter(axial_dist[j], left_halfwidths[j])
    plt.scatter(axial_dist[j], right_halfwidths[j])
plt.title("Halfwidth vs axial distance")
plt.xlabel("Axial distance")
plt.ylabel("Halfwidth")
plt.show()

valid_idx = ~np.isnan(left_core) & ~np.isnan(right_core)

x_clean, y_leftcore_clean, y_rightcore_clean = axial_dist[valid_idx], left_core[valid_idx], right_core[valid_idx]

m_left, c_left = np.polyfit(x_clean, y_leftcore_clean, 1)
m_right, c_right = np.polyfit(x_clean, y_rightcore_clean, 1)

x_inter_left = -c_left / m_left
x_inter_right = -c_right / m_right

x_intersect = (c_right - c_left) / (m_left - m_right)
y_intersect = m_left * x_intersect + c_left

x_extrapolate = np.linspace(0, x_intersect * 1.2, 100)

y_left_line = m_left * x_extrapolate + c_left
y_right_line = m_right * x_extrapolate + c_right

plt.figure(figsize=(8, 5))
plt.scatter(axial_dist, left_core)
plt.scatter(axial_dist, right_core)
plt.plot(x_extrapolate, y_left_line, color='gray', linestyle='--', label='Left Core Extrapolation')
plt.plot(x_extrapolate, y_right_line, color='orange', linestyle='--', label='Right Core Extrapolation')
plt.scatter(x_intersect, y_intersect, color='red', label='Estimated Core Collapse Point')
plt.axvline(x=0, color='gray', linestyle='--', label='Jet Outlet')
plt.axvline(x=x_inter_left, color='blue', linestyle=':', label=f'Leftmost pole position: x={x_inter_left:.2f}')
plt.axvline(x=x_inter_right, color='green', linestyle=':', label=f'Rightmost pole position: x={x_inter_right:.2f}')
plt.title("Potential Core Radius Extrapolation")
plt.xlabel("Axial Distance")
plt.ylabel("Potential Core Radius")
plt.legend()
plt.show()

valid_idx_half = ~np.isnan(left_halfwidths) & ~np.isnan(right_halfwidths)

x_clean, y_haleft_clean, y_haright_clean = axial_dist[valid_idx_half], left_halfwidths[valid_idx_half], right_halfwidths[valid_idx_half]
m_haleft, c_haleft = np.polyfit(x_clean, y_haleft_clean, 1)
m_haright, c_haright = np.polyfit(x_clean, y_haright_clean, 1)



x_hal_intersect = (c_haright - c_haleft) / (m_haleft - m_haright)

y_hal_intersect = m_haleft * x_hal_intersect + c_haleft

x_intercept_left = -c_haleft / m_haleft
x_intercept_right = -c_haright / m_haright

x_extrapolate_half = np.linspace(x_hal_intersect*1.2, 100, 101)

x_poleposition = np.linspace(x_intercept_right, x_intercept_left, 101)

y_haleft_line = m_haleft * x_extrapolate_half + c_haleft

y_haright_line = m_haright * x_extrapolate_half + c_haright

plt.figure(figsize=(10, 6))
plt.scatter(axial_dist, left_halfwidths)
plt.scatter(axial_dist, right_halfwidths)
plt.plot(x_extrapolate_half, y_haleft_line, color='gray', linestyle='--', label='Left Halfwidth Extrapolation')
plt.plot(x_extrapolate_half, y_haright_line, color='orange', linestyle='--', label='Right Halfwidth Extrapolation')
plt.axvline(x=x_intercept_left, color='blue', linestyle=':', label=f'Leftmost pole position: x={x_intercept_left:.2f}')
plt.axvline(x=x_intercept_right, color='green', linestyle=':', label=f'Rightmost pole position: x={x_intercept_right:.2f}')
plt.scatter(x_hal_intersect, y_hal_intersect, color='red', label='Estimated Pole Position')
plt.axhline(y=0, color='black', linestyle='-')
plt.title("Halfwidth Extrapolation")
plt.xlabel("Axial Distance")
plt.ylabel("Halfwidth")

plt.legend(loc='upper right', fontsize='small')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(axial_dist, left_core)
plt.scatter(axial_dist, right_core)
plt.plot(x_extrapolate, y_left_line, color='blue', linestyle='--', label='Left Core Extrapolation')
plt.plot(x_extrapolate, y_right_line, color='orange', linestyle='--', label='Right Core Extrapolation')
plt.scatter(x_intersect, y_intersect, color='red', label='Estimated Core Collapse Point')
plt.scatter(axial_dist, left_halfwidths)
plt.scatter(axial_dist, right_halfwidths)
plt.plot(x_extrapolate_half, y_haleft_line, color='blue', linestyle='--', label='Left Halfwidth Extrapolation')
plt.plot(x_extrapolate_half, y_haright_line, color='orange', linestyle='--', label='Right Halfwidth Extrapolation')
plt.axvline(x=x_intercept_left, color='blue', linestyle=':', label=f'Leftmost pole position: x={x_intercept_left:.2f}')
plt.axvline(x=x_intercept_right, color='green', linestyle=':', label=f'Rightmost pole position: x={x_intercept_right:.2f}')
plt.scatter(x_hal_intersect, y_hal_intersect, color='red', label='Estimated Pole Position')
plt.axvline(x=0, color='gray', linestyle='--', label='Jet Outlet')
plt.axvline(x=x_inter_left, color='blue', linestyle=':', label=f'Leftmost pole position: x={x_inter_left:.2f}')
plt.axvline(x=x_inter_right, color='green', linestyle=':', label=f'Rightmost pole position: x={x_inter_right:.2f}')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title("messy combined graph")
plt.xlabel("Axial Distance")
plt.ylabel("Halfwidth")
plt.axvline(x=0, color='gray', linestyle='--', label='Jet Outlet')
plt.legend(loc='upper left', fontsize='small')
plt.show()