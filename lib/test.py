# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:04:54 2019

@author: QuickLine
"""

import numpy as np
import matplotlib.pyplot as plt
#    x1 = np.random.multivariate_normal([3,3],[[2, 1],
#                                         [1, 2]], 200)    
#     x2,= np.random.multivariate_normal([3,3],[[2, 1],
#                                              [1, 2]], 200)    
#    x = np.vstack([x1, np.random.multivariate_normal([5,5],[[2, -1],
#          
#    x = np.vstack([x1, np.random.multivariate_normal([5,5],[[2, -1],
#          
mean = [0, 0]
cov = [[5, 3], 
       [3, 5]]  # diagonal covariance
x = np.random.multivariate_normal(mean, cov, 100, check_valid='raise')
mean = [8, 8]
cov = [[5, -3], 
       [-3, 5]]  # diagonal covariance
x = np.vstack([x, np.random.multivariate_normal(mean, cov, 100, check_valid='raise')])
mean = [-3, 5]
cov = [[2, 2], 
       [2, 4]]  # diagonal covariance
x = np.vstack([x, np.random.multivariate_normal(mean, cov, 100, check_valid='raise')])

mean = [15, -8]
cov = [[20, 0], 
       [0, 20]]  # diagonal covariance
x = np.vstack([x, np.random.multivariate_normal(mean, cov, 100, check_valid='raise')])
plt.plot(x[:,0], x[:,1], 'x')
plt.axis('equal')
plt.show()