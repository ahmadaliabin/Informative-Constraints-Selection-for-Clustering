import numpy as np
import random
import matplotlib.pyplot as plt

point_per_dist=20
mean = [0, 0]
cov = [[5, -3], 
       [-3, 5]]  # diagonal covariance
x = np.random.multivariate_normal(mean, cov, point_per_dist, check_valid='raise')
y=np.ones(point_per_dist, dtype=np.int8)
mean = [6, 6]
cov = [[5, -3], 
       [-3, 5]]  # diagonal covariance
x = np.vstack([x, np.random.multivariate_normal(mean, cov, point_per_dist, check_valid='raise')])
y=np.hstack([y, np.ones(point_per_dist,dtype=np.int8) + 1])