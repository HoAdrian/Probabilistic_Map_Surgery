import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal

import numpy as np
from numpy.linalg import cholesky, inv



'''
Sequential Bayesian Hilbert Map
N = number of examples, M = dimension of features = number of hinge points (+1 if use bias trick)
target value = either 0 or 1
'''

def rbf_kernel(X, Y, gamma, bias_trick=True):
    '''
    X shape: (N, d)
    Y shape: (M, d)

    compute exp(-gamma*L2_norm(x_i - y_j)) for each i, j

    return: N feature vectors of M dimension (kernel matrix). If use bias trick, append 1 to each row
    '''

    X = torch.unsqueeze(X, dim=1) #(N,1,d)
    Y = torch.unsqueeze(Y, dim=0) #(1,M,d)
    sqr_dist = torch.sum((X-Y)**2, dim=-1) #(N,M)
    kernel_mat = torch.exp(-gamma*sqr_dist) #(N,M)
    if bias_trick:
        ones = torch.ones((X.shape[0], 1))
        kernel_mat = torch.cat((ones, kernel_mat), dim=1)
    return kernel_mat

def lambda_helper(xi):
    return 1/2*xi*(torch.sigmoid(xi) - 0.5)

def spd_inverse(A):
    L = torch.linalg.cholesky(A)
    inv_L_t = torch.linalg.inv(L).transpose(0,1)
    inv_L = torch.linalg.inv(L)
    return torch.matmul(inv_L_t, inv_L)

def E_step(xi, S_0_inv, m_0, t, Phi, full_covar=False):
    '''
    N = number of examples, M = dimension of features = number of hinge points (+1 if use bias trick)

    xi: (N,1) variational parameters
    S_0_inv: (M,M) inverse of covariance of old posterior
    m_0: (M,1) mean of old posterior
    t: (N,1) target values
    Phi: (N,M) feature vectors
    '''
    weighted_Phi = lambda_helper(xi)*Phi #(N,M)
    S_N_inv = S_0_inv + 2* torch.matmul(weighted_Phi.t(), Phi)
    m_right = torch.matmul(S_0_inv, m_0) + torch.sum((t-0.5)*Phi, dim=0).reshape(-1,1) #(M,1)
    L = torch.linalg.cholesky(S_N_inv)
    Z = torch.triangular_solve(m_right, L, upper=False).solution
    m_N = torch.triangular_solve(Z, L.T, upper=True).solution

    L_inv = torch.triangular_solve(torch.eye(Phi.shape[1]).type(torch.DoubleTensor), L, upper=True).solution

    if full_covar:
        S_N = torch.matmul(L_inv.t(), L_inv)
        return m_N, S_N
    else:
        return m_N, S_N_inv, L_inv

def M_step(L_inv, m_N, Phi):
    '''
    Update the variational parameter xi = [xi_1, ... , xi_N]^T to maximize variational lower bound
    '''
    XMX = torch.matmul(Phi, m_N)**2
    XSX = torch.sum(torch.matmul(Phi, L_inv.t()) ** 2, dim=1)
    new_xi = np.sqrt(XMX + XSX)
    return new_xi

class SBHM_Optimized:
    '''
    REMEMBER: 
    Use tensor of type double, .type(torch.DoubleTensor)
    '''
    def __init__(self, S_0, S_0_inv, m_0, xi):
        '''
         xi: (N,1) variational parameters
        S_0: (M,M) covariance of old posterior
        m_0: (M,1) mean of old posterior
        '''
        self.covariance_inv = S_0_inv
        self.covariance = S_0
        self.mean = m_0
        self.xi = xi
    
    def EM_algo(self, t, Phi, num_iters):

        for i in range(num_iters):
            print(f"+++EM iter {i}")
            self.mean, self.covariance_inv, self.L_inv = E_step(self.xi, self.covariance_inv, self.mean, t, Phi)
            self.covariance = torch.matmul(self.L_inv.t(), self.L_inv)
            # self.new_xi = M_step(self.covariance, self.mean, Phi)
            self.new_xi = M_step(self.L_inv, self.mean, Phi)

        return self.covariance, self.mean, self.xi

    def sample_weights(self):
        num_samples = 1000
        normal = MultivariateNormal(self.mean.reshape(-1), self.covariance)
        W = normal.sample(torch.Size([num_samples])).type(torch.DoubleTensor)
        return W #(num_samples, M)

    def predict(self, Phi, W):
        pred = torch.sigmoid(torch.matmul(Phi, W.t())) #(N,num_samples)
        pred_mean = torch.sum(pred, dim=-1)/W.shape[0]
        pred_std = torch.std(pred, dim=-1)

        return pred, pred_mean, pred_std




############################ numpy implementation ##############################################
# from scipy.stats import multivariate_normal
# from scipy.linalg import solve_triangular
# from scipy.special import expit

# def rbf_kernel(X, Y, gamma, bias_trick=True):
#     X = np.expand_dims(X, axis=1) #(N,1,d)
#     Y = np.expand_dims(Y, axis=0) #(1,M,d)
#     sqr_dist = np.sum((X - Y)**2, axis=-1) #(N,M)
#     kernel_mat = np.exp(-gamma * sqr_dist) #(N,M)
#     if bias_trick:
#         ones = np.ones((X.shape[0], 1))
#         kernel_mat = np.concatenate((ones, kernel_mat), axis=1)
#     return kernel_mat

# def lambda_helper(xi):
#     return 1/2 * xi * (1 / (1 + np.exp(-xi)) - 0.5)

# def spd_inverse(A):
#     L = cholesky(A)
#     inv_L_t = inv(L).T
#     inv_L = inv(L)
#     return np.dot(inv_L_t, inv_L)

# def E_step(xi, S_0_inv, m_0, t, Phi, full_covar=False):
#     weighted_Phi = lambda_helper(xi) * Phi #(N,M)
#     S_N_inv = S_0_inv + 2 * np.dot(weighted_Phi.T, Phi)
#     m_right = np.dot(S_0_inv, m_0) + np.sum((t - 0.5) * Phi, axis=0).reshape(-1, 1) #(M,1)
#     L = cholesky(S_N_inv)
#     Z = solve_triangular(L, m_right, lower=True)
#     m_N = solve_triangular(L.T, Z, lower=False)

#     L_inv = solve_triangular(L.T, np.eye(Phi.shape[1]), lower=True)

#     if full_covar:
#         S_N = np.dot(L_inv.T, L_inv)
#         return m_N, S_N
#     else:
#         return m_N, S_N_inv, L_inv

# def M_step(S_N_inv, L_inv, m_N, Phi):
#     XMX = np.dot(Phi, m_N)**2
#     XSX = np.sum(np.dot(Phi, L_inv.T) ** 2, axis=1)
#     new_xi = np.sqrt(XMX + XSX)
#     return new_xi

# class SBHM:
#     def __init__(self, S_0, S_0_inv, m_0, xi):
#         self.covariance_inv = S_0_inv
#         self.covariance = S_0
#         self.mean = m_0
#         self.xi = xi
    
#     def EM_algo(self, t, Phi, num_iters):
#         for i in range(num_iters):
#             print(f"+++EM iter {i}")
#             self.mean, self.covariance_inv, self.L_inv = E_step(self.xi, self.covariance_inv, self.mean, t, Phi)
#             self.covariance = np.dot(self.L_inv.T, self.L_inv)
#             self.new_xi = M_step(self.covariance_inv, self.L_inv, self.mean, Phi)
#         return self.covariance, self.mean, self.xi

#     def sample_weights(self):
#         num_samples = 1000
#         normal = multivariate_normal(self.mean.reshape(-1), self.covariance)
#         W = normal.rvs(size=num_samples)
#         return W #(num_samples, M)

#     def predict(self, Phi, W):
#         pred = expit(np.dot(Phi, W.T)) #(N,num_samples)
#         pred_mean = np.mean(pred, axis=-1)
#         pred_std = np.std(pred, axis=-1)
#         return pred, pred_mean, pred_std