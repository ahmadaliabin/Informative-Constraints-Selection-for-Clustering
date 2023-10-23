# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:39:36 2019

@author: Dr.Abin
"""
import numpy as np
import logging
import random
import matplotlib.pyplot as plt


logger = logging.getLogger("main_logger")
#%%
def pdist2X1(X1):
#    logger.info("Calculating pairwise distances X1")
    try:
        return np.sqrt(np.sum((X1[None , :] - X1[:, None])**2, -1))
    except MemoryError:
        N,D=X1.shape
        distMat=np.zeros((N,N),dtype='float32')
        for i in range(N):
            for j in range(N):
                distMat[i,j]=np.sqrt(np.sum((X1[i , :] - X1[j , :])**2, -1))
        return distMat
#%%
def pdist2X1X2(X1,X2):
#    logger.info("Calculating pairwise distances X1, X2")
    try:
        return np.sqrt(np.sum((X1[None , :] - X2[:, None])**2, -1))
    except MemoryError:
        N,D=X1.shape
        distMat=np.zeros((N,N),dtype='float32')
        for i in range(N):
            for j in range(N):
                distMat[i,j]=np.sqrt(np.sum((X1[i , :] - X2[j , :])**2, -1))
        return distMat
#%%
def load_syntethic_data(file):
#    logger.info("Loading synthetic data")
    XY=np.loadtxt(file)
    N,_=XY.shape
    N_sample=2000
    XY = XY[random.sample(range(N),N_sample),:] if N > N_sample else XY
    X=XY[::,0:-1] if XY.shape[1] > 2 else XY
    Y=XY[::,-1] if XY.shape[1] > 2 else None
    N,D=X.shape
    return X,Y,N,D
#%%
def gaussian(n=2, point_per_dist=100):
    if n==2:
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
    elif n==3:
        mean = [0, 0]
        cov = [[5, 3], 
               [3, 5]]  # diagonal covariance
        x = np.random.multivariate_normal(mean, cov, point_per_dist, check_valid='raise')
        y=np.ones(point_per_dist, dtype=np.int8)
        mean = [10, 8]
        cov = [[5, 3], 
               [3, 5]]  # diagonal covariance
        x = np.vstack([x, np.random.multivariate_normal(mean, cov, point_per_dist, check_valid='raise')])
        y=np.hstack([y, np.ones(point_per_dist,dtype=np.int8) + 1])
        mean = [-2, 8]
        cov = [[5, -2], 
               [-2, 2]]  # diagonal covariance
        x = np.vstack([x, np.random.multivariate_normal(mean, cov, point_per_dist, check_valid='raise')])
        y=np.hstack([y, np.ones(point_per_dist,dtype=np.int8) + 2])


    elif n==4:
        mean = [0, 0]
        cov = [[5, 3], 
               [3, 5]]  # diagonal covariance
        x = np.random.multivariate_normal(mean, cov, point_per_dist, check_valid='raise')
        y=np.ones(point_per_dist, dtype=np.int8)
        mean = [8, 8]
        cov = [[5, -3], 
               [-3, 5]]  # diagonal covariance
        x = np.vstack([x, np.random.multivariate_normal(mean, cov, point_per_dist, check_valid='raise')])
        y=np.hstack([y, np.ones(point_per_dist,dtype=np.int8) + 1])
        mean = [-3, 5]
        cov = [[2, 2], 
               [2, 4]]  # diagonal covariance
        x = np.vstack([x, np.random.multivariate_normal(mean, cov, point_per_dist, check_valid='raise')])
        y=np.hstack([y, np.ones(point_per_dist,dtype=np.int8) + 2])
        mean = [15, -8]
        cov = [[20, 0], 
               [0, 20]]  # diagonal covariance
        x = np.vstack([x, np.random.multivariate_normal(mean, cov, point_per_dist, check_valid='raise')])
        y=np.hstack([y, np.ones(point_per_dist,dtype=np.int8) + 3])

    return x, y, x.shape[0], x.shape[1]
#%%
def calc_density(X, knn, nn_idxs_of_all_points):
    N, D = X.shape
    density=np.zeros(N,np.float32)
    for i in range(N):
        xi=X[i,:]
        knn_idxs_of_xi=nn_idxs_of_all_points[i,1:knn+1]# indices of k neareset neighbors of xi
        knn_of_xi=X[knn_idxs_of_xi,:]
        dist_to_xi=[np.sqrt(np.sum((xi - knn_of_xi[j,:])**2, -1)) for j in range(knn)]
        density[i]=1/max(dist_to_xi)
    return density
#%%
def track_density(X, knn, nn_idxs_of_all_points, density):
    N,D=X.shape
    idx_of_nn_with_higher_density=np.zeros(N,np.int32)
    for i in range(N): # get the indx of nn with higher density 
        knn_idxs_of_xi=nn_idxs_of_all_points[i,1:knn+1]
        knn_density_of_xi=density[knn_idxs_of_xi]
        idx_of_nn_with_higher_density[i]=knn_idxs_of_xi[knn_density_of_xi > density[i]][0] if any(knn_density_of_xi > density[i]) else -1
    
    densityChain=[]
    centrality=np.zeros(N,np.int32)+1
    for i in range(N):
        densityChain.append([i])
        nxt=idx_of_nn_with_higher_density[i]
        if nxt == -1:
            continue
        current=i
        density_of_current=density[current]
        density_of_nxt=density[nxt]       
        while density_of_nxt > density_of_current:
            densityChain[i].append(nxt)
            centrality[nxt]+=1
            current=nxt
            nxt=idx_of_nn_with_higher_density[current]
            if nxt == -1:
                break
            density_of_current = density[current]
            density_of_nxt = density[nxt]
            
    miniGroupCenters=np.nonzero(idx_of_nn_with_higher_density==-1)#extract connected component as mini groups
    miniGroupId=np.zeros(N,np.int32)
    miniGroups=[]
    groupId=0
    for miniGroupCenter in list(miniGroupCenters)[0]:
        newGroup=[]
        for chain in densityChain:
            if chain[-1]==miniGroupCenter:
                newGroup.append(chain[0])
        if len(newGroup) > 0:
            miniGroups.append(newGroup)
            miniGroupId[newGroup]=groupId
            groupId+=1
        
    numelOfGroups=np.zeros(N,np.int32)
    for i in range(N):
        numelOfGroups[i]=np.sum(miniGroupId==miniGroupId[i])

    return densityChain, centrality, miniGroups, miniGroupId

#%%
def calc_impurity(X, knn, nn_idxs_of_all_points, density, densityChain, miniGroupId):
    N,D=X.shape    
    pointsInpurity=np.zeros(N,np.float64)
    for i in range(N): # calc the inpurity of points 
        xi_and_knn_idxs_of_xi=nn_idxs_of_all_points[i,0:knn+1]
        knn_groupId_of_xi=miniGroupId[xi_and_knn_idxs_of_xi]
        for j in np.unique(miniGroupId):
            per=np.sum(knn_groupId_of_xi==j)/float(knn+1)
            pointsInpurity[i]+=per**2
        pointsInpurity[i]=1-pointsInpurity[i]
        
        pointsInpurity[i] *= 1 - density[i] / density[densityChain[i][-1]]
    return pointsInpurity
#%%
def entropy(p1,p2):
    t1=p1*np.log2(p1) if p1!=0 else 0
    t2=p2*np.log2(p2) if p2!=0 else 0
    return -(t1+t2)
#%% visualize_variables
def visualize(X, Y, points_density, points_centrality, points_impurity, density_chain, mini_groups, constraints, heuristic, visualization_on=True):
    D = X.shape[1]
    if visualization_on and D == 2:
        logger.info("Visualization started, please wait, it may take time ...")
        nrows, ncols = 9, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=(8*ncols,8*nrows))
        ax[0].set_title("Density tracking")
        ax[0].set_xlabel('$x_1$')
        ax[0].set_ylabel('$x_2$')
        ax[1].set_title("Centrality")
        ax[1].set_xlabel('$x_1$')
        ax[1].set_ylabel('$x_2$')
        ax[2].set_title("Impurity")
        ax[2].set_xlabel('$x_1$')
        ax[2].set_ylabel('$x_2$')
        ax[3].set_title("Constraints selected by heuristic 1")
        ax[3].set_xlabel('$x_1$')
        ax[3].set_ylabel('$x_2$')
        ax[4].set_title("Constraints selected by heuristic 2")
        ax[4].set_xlabel('$x_1$')
        ax[4].set_ylabel('$x_2$')
        ax[5].set_title("Constraints selected by heuristic 3")
        ax[5].set_xlabel('$x_1$')
        ax[5].set_ylabel('$x_2$')
        ax[6].set_title("Constraints selected by heuristics 1, 2 and 3")
        ax[6].set_xlabel('$x_1$')
        ax[6].set_ylabel('$x_2$')
        ax[7].set_title("Constraints")
        ax[7].set_xlabel('$x_1$')
        ax[7].set_ylabel('$x_2$')
        ax[8].set_title("Data points")
        ax[8].set_xlabel('$x_1$')
        ax[8].set_ylabel('$x_2$')       # start plotting
        flag = 1
        if flag == 1:
            for mini_group in mini_groups:
                color = np.random.rand(3)
                for pt_Indx in mini_group:
                    ax[0].plot(X[pt_Indx,0], X[pt_Indx,1], 'o', markerfacecolor=color, markeredgecolor='k', markeredgewidth=0.75, markersize= (points_density/max(points_density))[pt_Indx]*20+1)
#                    ax[1].plot(X[pt_Indx,0], X[pt_Indx,1], 'o', markerfacecolor=color, markeredgecolor='k', markeredgewidth=0.75, markersize=8)
                    ax[1].plot(X[pt_Indx,0], X[pt_Indx,1], 'o', markerfacecolor=color, markeredgecolor='k', markeredgewidth=0.75, markersize=(points_centrality/max(points_centrality))[pt_Indx]*20+1)
                    ax[2].plot(X[pt_Indx,0], X[pt_Indx,1], 'o', markerfacecolor=color, markeredgecolor='k', markeredgewidth=0.75, markersize=(points_impurity/max(points_impurity))[pt_Indx]*20+1)
                    ax[3].plot(X[pt_Indx,0], X[pt_Indx,1], 'o', markerfacecolor=color, markeredgecolor='k', markeredgewidth=0.75, markersize=(points_impurity/max(points_impurity))[pt_Indx]*20+1)
                    ax[4].plot(X[pt_Indx,0], X[pt_Indx,1], 'o', markerfacecolor=color, markeredgecolor='k', markeredgewidth=0.75, markersize=(points_impurity/max(points_impurity))[pt_Indx]*20+1)
                    ax[5].plot(X[pt_Indx,0], X[pt_Indx,1], 'o', markerfacecolor=color, markeredgecolor='k', markeredgewidth=0.75, markersize=(points_centrality/max(points_centrality))[pt_Indx]*20+1)
                    ax[6].plot(X[pt_Indx,0], X[pt_Indx,1], 'o', markerfacecolor='k', markeredgecolor='k', markeredgewidth=0.75, markersize=3)
                    ax[7].plot(X[pt_Indx,0], X[pt_Indx,1], 'o', markerfacecolor='k', markeredgecolor='k', markeredgewidth=0.75, markersize=3)
            for current_chain in density_chain:
                color = [0, 0, 0]
                for i,j in zip(current_chain[:-1],current_chain[1:]):
                    x1 = X[i,:]
                    x2 = X[j,:]
#                    ax[0].arrow(x1[0], x1[1], .7*(x2[0] - x1[0]), .7*(x2[1] - x1[1]), color=color, head_width=0.2, head_length=0.2, linewidth=0.75,)
#                    ax[1].arrow(x1[0], x1[1], .7*(x2[0] - x1[0]), .7*(x2[1] - x1[1]), color=color, head_width=0.2, head_length=0.2, linewidth=0.75,)
                    ax[0].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], head_width=0.02, head_length=0.02, linewidth=0.75)
                    ax[1].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], head_width=0.02, head_length=0.02, linewidth=0.75)
#                    ax[2].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], head_width=0.02, head_length=0.02, linewidth=0.75)
                    ax[3].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], head_width=0.02, head_length=0.02, linewidth=0.75)
                    ax[4].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], head_width=0.02, head_length=0.02, linewidth=0.75)
                    ax[5].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], head_width=0.02, head_length=0.02, linewidth=0.75)
#                    ax[6].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], head_width=0.02, head_length=0.02, linewidth=0.75)
#                    ax[7].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], head_width=0.02, head_length=0.02, linewidth=0.75)
        ctr = 0
        strategy_line_style = ['-.', ':', '-'] # 0: impure_to_pure, 1: impure_to_impure, 2: from_skeleton
        strategy_line_color = ['b', 'r', 'g']
        for i, j, type_ in constraints:
            x1 = X[i,:]
            x2 = X[j,:]
            if heuristic[ctr] ==0:
                ax[3].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], color = strategy_line_color[heuristic[ctr]], ls = strategy_line_style[heuristic[ctr]], head_width=0.02, head_length=0.02, linewidth=2)
            if heuristic[ctr] ==1:
                ax[4].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], color = strategy_line_color[heuristic[ctr]], ls = strategy_line_style[heuristic[ctr]], head_width=0.02, head_length=0.02, linewidth=2)
            if heuristic[ctr] ==2:
                ax[5].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], color = strategy_line_color[heuristic[ctr]], ls = strategy_line_style[heuristic[ctr]], head_width=0.02, head_length=0.02, linewidth=2)
            ax[6].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], color = strategy_line_color[heuristic[ctr]], ls = strategy_line_style[heuristic[ctr]], head_width=0.02, head_length=0.02, linewidth=2)
            ax[7].arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], color = 'g' if type_ == 1 else 'r', ls= '-'  if type_ == 1 else '--', head_width=0.02, head_length=0.02, linewidth=2)
            ctr += 1
        ax[8].plot(X[:,0], X[:,1],'ok')
        #plt.savefig('points.pdf',dpi=300) # enable if you want to export as pdf
        fig.show()
#%% start to select constraints
def select_constraints(X, Y, nn_idxs_of_all_points, points_density, points_centrality, points_impurity, density_chain, mini_groupId, num_of_constraints=10):
    density_fallofff=0.8
    constraints=[]
    heuristic=[] # 1: impure_to_pure, 2: impure_to_impure, 3: from_skeleton
    points_impurity_sorted_index=np.argsort(points_impurity)[::-1]
    for i in range(num_of_constraints):
        #select constraints by heuristic 1
        impure_point = points_impurity_sorted_index[i]
        chain_of_impure_point = density_chain[impure_point]
        mask = points_density[chain_of_impure_point] >= (density_fallofff * points_density[chain_of_impure_point[-1]])
        pure_point_in_same_chain = chain_of_impure_point[np.nonzero(mask)[0][0]]
        constraints.append([impure_point, pure_point_in_same_chain, 1 if Y[impure_point] == Y[pure_point_in_same_chain] else -1])
        heuristic.append(0)
        
        #select constraints by heuristic 2
        nn_point_to_impure_point_from_other_chain = nn_idxs_of_all_points[impure_point, np.nonzero(mini_groupId[nn_idxs_of_all_points[impure_point,:]] != mini_groupId[impure_point])[0][0]]
        constraints.append([impure_point, nn_point_to_impure_point_from_other_chain, 1 if Y[impure_point] == Y[nn_point_to_impure_point_from_other_chain] else -1])
        heuristic.append(1)
    
        #select constraints by heuristic 3
    sampling_count = 2 # >2
    value = points_centrality.copy()
    for i in range(num_of_constraints):
        chain_value = np.array([np.sum(value[chain]) for chain in density_chain])
        chain_value_max_index = np.argmax(chain_value)
        chain = np.array(density_chain[chain_value_max_index], dtype='int32')
        index = chain[np.linspace(0, len(chain)-1, sampling_count , dtype='int32')]
        for from_, to in zip(index[0:-1], index[1:]):
            constraints.append([from_, to, 1 if Y[from_] == Y[to] else -1])
            heuristic.append(2)      
        value[density_chain[chain_value_max_index]] = 0
    return constraints, heuristic
#%% preprocessing
def pre_process(X):
    #logger.info("Pre processing started ...")
    pdist_X = pdist2X1(X) #pairwise distances between X and X
    nn_idxs_of_all_points = np.argsort(pdist_X, 1) #index of nearest neghbours for points in X
    pi, pc, pd = 0, 0, 0
    k_base=5
    knns=[k_base + 2 * i for i in range(10)]
    r=0.9
    for ctr,knn in enumerate(knns):
        logger.info("k (k nearest neighbors)= " + str(knn)+'...')
        points_density = calc_density(X, knn, nn_idxs_of_all_points) #calc density around each point
        density_chain, points_centrality, mini_groups, mini_groupId = track_density(X, knn, nn_idxs_of_all_points, points_density) #track density for each point
        points_impurity = calc_impurity(X, knn, nn_idxs_of_all_points, points_density, density_chain, mini_groupId)#calc impurity for each point
        decayfactor= 1 - r**ctr # apply the effect of different knn
        pi += decayfactor * points_impurity
        pc += decayfactor * points_centrality
        pd += decayfactor * points_density   
    return nn_idxs_of_all_points, pi, pc, pd
