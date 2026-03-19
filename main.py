from pointcloud import * # Also imports point.py

cloud = PointCloud()
cloud.read_test_data()
cloud.plot_2D('velocity_skewness')
