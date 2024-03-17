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
from bhm_optimized import SBHM_Optimized, rbf_kernel
import math
from sklearn.datasets import make_classification

########################## SBHM ############################
if __name__ == "__main__":
    torch.random.manual_seed(2021)
    np.random.seed(2021)

    gamma = 240#60#15
    num_EM_iters = 10 #100

    ######### set up real data points
    balls_xy = np.array([[-0.05, -0.35], [0.05, -0.4]])
    #balls_xy = np.array([[0.017, -0.4]])

    xy_lower_bound = np.array([-0.1, -0.5])
    xy_upper_bound = np.array([0.1, -0.3])

    n_train_points = 1024
    train_points_2D = get_2D_grid_points(n_train_points, xy_lower_bound, xy_upper_bound) #(n_query_points, 2)
    train_points_labels = get_point_label_nn(balls_xy, train_points_2D, num_positive_points=50) #(n_query_points, )
    #print(np.sum(train_points_labels))
    # train_points_2D = train_points_2D[train_points_labels==1]
    # train_points_labels = train_points_labels[train_points_labels==1]

    # mask = np.random.choice(train_points_2D.shape[0], size=100, replace=False)
    # train_points_2D = train_points_2D[mask]
    # train_points_labels = train_points_labels[mask]


    n_hinge_points = 36
    hinge_points = torch.tensor(get_2D_grid_points(n_hinge_points, xy_lower_bound, xy_upper_bound))

    ############ set up toy data points
    # train_points_2D, train_points_labels = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, shift=3)
    # train_points_labels = train_points_labels.ravel()
    # plt.scatter(train_points_2D[:,0], train_points_2D[:,1], c=train_points_labels, marker='x', cmap='jet')
    # plt.colorbar()

    #hinge_points = torch.tensor(get_2D_grid_points(36, [-3,-3], [3,3]))

    M = hinge_points.shape[0]+1
    N = train_points_2D.shape[0]
    
    ########## start learning
    variances = 1000.*torch.ones((M,))
    S_0 = torch.diag(variances)
    S_0_inv = torch.diag(1./variances)
    m_0 = torch.zeros((M,1))
    xi = torch.ones((N,1))

    sbhm = SBHM_Optimized(S_0, S_0_inv, m_0, xi)
    
    X = torch.tensor(train_points_2D)
    Phi = rbf_kernel(X, hinge_points, gamma=gamma, bias_trick=True)
    sbhm_target = torch.tensor(train_points_labels).reshape(-1,1)#.type(torch.DoubleTensor)
    sbhm.EM_algo(sbhm_target, Phi, num_iters=num_EM_iters)
    W = sbhm.sample_weights()

    test_points_2D = get_2D_grid_points(1024, xy_lower_bound, xy_upper_bound) #(n_query_points, 2)
    qX = torch.tensor(test_points_2D)
    test_Phi = rbf_kernel(qX, hinge_points, gamma=gamma, bias_trick=True)
    _, sbhm_pred_mean, sbhm_pred_std = sbhm.predict(test_Phi, W)

    #laplace_pred = sbhm.laplace_predict(test_Phi)

    plot_path = "/home/dvrk/active_data/1ball/online_learning"
    #plot_2D_points(hinge_points, [1 for _ in range(hinge_points.shape[0])], vmin=0, vmax=1, title=f"hinge points", path= f"{plot_path}/debug", name="hinge_points_debug")
    plot_2D_points(X, train_points_labels, vmin=0, vmax=1, title=f"ground truth debug", path= f"{plot_path}/debug_optimized", name="ground_truth_prediction_debug")
    plot_2D_points(qX, sbhm_pred_mean, vmin=0, vmax=1, title=f"sbhm mean prediction debug", path= f"{plot_path}/debug_optimized", name="sbhm_prediction_debug")
    plot_2D_points(qX, sbhm_pred_std, vmin=0, vmax=math.inf, title=f"sbhm prediction std debug", path= f"{plot_path}/debug_optimized", name="sbhm_std_debug", has_vmin=True, has_vmax=False)
    #plot_2D_points(qX, laplace_pred, vmin=0, vmax=1, title=f"sbhm laplace prediction debug", path= f"{plot_path}/debug_optimized", name="sbhm__laplace_prediction_debug")

