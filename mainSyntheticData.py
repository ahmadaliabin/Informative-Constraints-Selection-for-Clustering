import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import lib.utilities as util
import logging
from sklearn.cluster import KMeans

#config logger
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("main_logger")

# load synthetic data
X, Y, N, D = util.gaussian(3, 120)
## or load from a file containing 2D data e.g. zelnik.txt
#X, Y, N, D = util.load_syntethic_data('data\\zelnik.txt')
plt.plot(X[:,0], X[:,1],'ok')
plt.show()
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

