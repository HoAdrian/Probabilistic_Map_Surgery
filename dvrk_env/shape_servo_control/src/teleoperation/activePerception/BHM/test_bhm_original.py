import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../utils")
from utils import get_point_label_nn, get_point_label_radius, plot_2D_points, get_2D_grid_points
from bhm_original import *
import math
from sklearn.datasets import make_classification

if __name__ == "__main__":
    torch.random.manual_seed(2021)

    gamma = 90.0

    xy_lower_bound = np.array([-0.1, -0.5])
    xy_upper_bound = np.array([0.1, -0.3])
    n_hinge_points = 36
    hinge_points = get_2D_grid_points(n_hinge_points, xy_lower_bound, xy_upper_bound)

    balls_xy_sequence = [np.array([[-0.05, -0.35]]), np.array([[0.05, -0.4]]), np.array([[0.0, -0.47]])]

    sbhm = SBHM(gamma=gamma, grid=hinge_points)

    for i in range(len(balls_xy_sequence)):
        ######### set up real data points
        #balls_xy = np.array([[-0.05, -0.35], [0.05, -0.4]])
        #balls_xy = np.array([[-0.017, -0.35]])
        balls_xy = balls_xy_sequence[i]


        n_query_points = 1024
        train_points_2D = get_2D_grid_points(n_query_points, xy_lower_bound, xy_upper_bound) #(n_query_points, 2)
        train_points_labels = get_point_label_nn(balls_xy, train_points_2D, num_positive_points=50) #(n_query_points, )

        # get all the positive points and some negative points
        pos_points_2D = train_points_2D[train_points_labels==1]
        pos_points_labels = train_points_labels[train_points_labels==1]

        num_neg_points = 100
        neg_points = train_points_2D[train_points_labels==0]
        mask = np.random.choice(neg_points.shape[0], size=num_neg_points, replace=False)
        neg_points = neg_points[mask]
        neg_points_labels = np.zeros((len(mask),))
        train_points_2D = np.concatenate((pos_points_2D, neg_points), axis=0)
        train_points_labels = np.concatenate((pos_points_labels, neg_points_labels), axis=0)

        ## randomly sample training points
        #mask = np.random.choice(train_points_2D.shape[0], size=500, replace=False)
        # train_points_2D = train_points_2D[mask]
        # train_points_labels = train_points_labels[mask]

        

        ############ set up toy data points
        # xy_lower_bound = np.array([-2, -2])
        # xy_upper_bound = np.array([2, 2])
        # train_points_2D, train_points_labels = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2)
        # train_points_labels = train_points_labels.ravel()
        # n_hinge_points = 36
        # hinge_points = get_2D_grid_points(n_hinge_points, xy_lower_bound, xy_upper_bound)
        # plt.scatter(train_points_2D[:,0], train_points_2D[:,1], c=train_points_labels, marker='x', cmap='jet')
        # plt.colorbar()

        # M = hinge_points.shape[0]
        # N = train_points_2D.shape[0]
        # print("N", N)
        
        ########## start learning
        
        X =train_points_2D
        
        sbhm_target = train_points_labels
        sbhm.fit(X, sbhm_target)
    

        
        test_points_2D = get_2D_grid_points(1024, xy_lower_bound, xy_upper_bound) #(n_query_points, 2)
        qX = test_points_2D
        sbhm_pred_mean = sbhm.predict_proba(qX)[:,1]


        plot_path = "/home/dvrk/active_data/1ball/online_learning"
        #plot_2D_points(hinge_points, [1 for _ in range(hinge_points.shape[0])], vmin=0, vmax=1, title=f"hinge points", path= f"{plot_path}/debug", name="hinge_points_debug")
        plot_2D_points(X, train_points_labels, vmin=0, vmax=1, title=f"ground truth t={i}", path= f"{plot_path}/debug_original", name=f"ground_truth_prediction_{i}")
        plot_2D_points(qX, sbhm_pred_mean, vmin=0, vmax=1, title=f"sbhm mean prediction t={i}", path= f"{plot_path}/debug_original", name=f"sbhm_prediction_debug_{i}")
        