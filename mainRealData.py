import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import lib.utilities as util
import logging
from sklearn.cluster import KMeans
import logging

logging.basicConfig()
logger = logging.getLogger("main_logger")
logger.setLevel(logging.INFO)


#config logger
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#logger = logging.getLogger("main_logger")

# load data
#        0               1             2            3                  4                  5              6           7             8
datasets=['flame.txt', 'jain.txt',  'pathbased.txt', 'unbalance.txt', 'Compound.txt', 'Aggregation.txt', 'iris.txt', 'glass.txt', 'diabetes.txt']
datasetIndex = 6 #the index of dataset selected fromthe from above datasets array
X, Y, N, D = util.load_syntethic_data('data\\'+ datasets[datasetIndex])

num_of_constraints = 10
#%% constraints selection started
logger.info("Pre processing started ...")
nn_idxs_of_all_points, points_impurity, points_centrality, points_density = util.pre_process(X)
logger.info("Constraints selection started")
density_chain, points_centrality, mini_groups, mini_groupId = util.track_density(X, 10, nn_idxs_of_all_points, points_density) #find density around each point
constraints, heuristic = util.select_constraints(X, Y, nn_idxs_of_all_points, points_density, points_centrality, points_impurity, density_chain, mini_groupId, num_of_constraints)
util.visualize(X, Y, points_density, points_centrality, points_impurity, density_chain, mini_groups, constraints, heuristic, visualization_on=True)
logger.info("The end, enjoy it")

print("List of constraints (point i, point j, type(1: must-link, -1: cannot-link)")
print(np.array(constraints))

