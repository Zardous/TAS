from pointcloud import * # Also imports point.py
import matplotlib.pyplot as plt

cloud = PointCloud()
cloud.read_test_data()

p = cloud.points[1][20]
p.spectral_analysis()
p.Kolmogorov()
p.energy_spectrum()

