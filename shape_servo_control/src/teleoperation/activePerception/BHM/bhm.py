import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal


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

def E_step(xi, S_0, m_0, t, Phi):
    '''
    N = number of examples, M = dimension of features = number of hinge points (+1 if use bias trick)

    xi: (N,1) variational parameters
    S_0: (M,M) covariance of old posterior
    m_0: (M,1) mean of old posterior
    t: (N,1) target values
    Phi: (N,M) feature vectors

    Initial posterior (or prior at time step 0) is q(w|m_0, S_0),
    update the posterior approximator to be q(w|m_N, S_N)
    '''
    weighted_Phi = lambda_helper(xi)*Phi #(N,M)
    outer_product_Phi = torch.matmul(weighted_Phi.transpose(0,1), Phi) #(M,M)
    inv_S_N = spd_inverse(S_0) + 2*outer_product_Phi
    S_N = spd_inverse(inv_S_N) #(M,M)

    inv_S_0 = spd_inverse(S_0)
    temp = torch.matmul(inv_S_0, m_0) + torch.sum((t-0.5)*Phi, dim=0).reshape(-1,1) #(M,1)
    m_N = torch.matmul(S_N, temp)

    return m_N, S_N

def M_step(S_N, m_N, Phi):
    '''
    S_N: (M,M) covariance of new posterior
    m_N: (M,1) mean of new posterior
    Phi: (N,M) feature vectors

    Update the variational parameter xi = [xi_1, ... , xi_N]^T to maximize variational lower bound
    '''
    A = (S_N + torch.matmul(m_N, m_N.transpose(0,1))) #(M,M)
    B = torch.matmul(A, Phi.transpose(0,1)) #(M,M)*(M,N) => (M,N)
    C = torch.matmul(Phi, B) # (N,M)*(M,N) => (N,N)
    new_xi = torch.sqrt(torch.diagonal(C))
    return new_xi.reshape(-1,1) 

def sig_likelihood(w, Phi):
    '''
    w: (M,1)
    Phi: (N,M)

    Compute Prob(t=1 | x_i, w) for i=1,...,N
    return: (N,1) matrix where each element is the likelihood for that data point
    '''
    return torch.sigmoid(torch.matmul(Phi, w))

class SBHM:
    def __init__(self, S_0, m_0, xi):
        self.covariance = S_0
        self.mean = m_0
        self.xi = xi
    
    def EM_algo(self, t, Phi, num_iters):

        S_0 = self.covariance
        m_0 = self.mean
        xi = self.xi
        for i in range(num_iters):
            print(f"+++EM iter {i}")
            m_N, S_N = E_step(xi, S_0, m_0, t, Phi)
            new_xi = M_step(S_N, m_N, Phi)
            xi = new_xi
            S_0 = S_N
            m_0 = m_N
        self.covariance = S_0
        self.mean = m_0
        self.xi = xi
        return S_0, m_0, xi

    def sample_weights(self):
        num_samples = 1000
        normal = MultivariateNormal(self.mean.reshape(-1), self.covariance)
        W = normal.sample(torch.Size([num_samples])).float()
        return W #(num_samples, M)

    def predict(Phi, W):
        pred = torch.sigmoid(torch.matmul(Phi, W.t())) #(N,num_samples)
        pred_mean = torch.sum(pred, dim=-1)/W.shape[0]
        pred_std = torch.std(pred, dim=-1)

        return pred, pred_mean, pred_std

if __name__ == "__main__":
    torch.random.manual_seed(2021)
    gamma = 0.5
    num_EM_iters = 1

    # rbf kernel
    X = torch.tensor([[0,1], [0,2]]).float()
    Y = torch.tensor([[0,2], [0,4], [0,8]]).float()

    Phi = rbf_kernel(X,Y,gamma=gamma)
    N = Phi.shape[0]
    M = Phi.shape[1]
    print(f"N: {N}")
    print(f"M: {M}")
    

    # e step and m step
    variances = 1000*torch.ones((M,))
    S_0 = torch.diag(variances)
    m_0 = torch.zeros((M,1))
    t = torch.tensor([1,0]).reshape(-1,1)
    xi = torch.ones((N,1))

    print(f"Phi: {Phi.shape}")
    print(f"S_0: {S_0.shape}")

    sbhm = SBHM(S_0, m_0, xi)
    sbhm.EM_algo(t, Phi, num_iters=num_EM_iters)
    
    # for i in range(num_EM_iters):
    #     print(f"+++iter {i}")
    #     m_N, S_N = E_step(xi, S_0, m_0, t, Phi)
    #     new_xi = M_step(S_N, m_N, Phi)
    #     xi = new_xi
    #     S_0 = S_N
    #     m_0 = m_N
    #     # print("m_N shape: ",m_N.shape)
    #     # print("S_N shape", S_N.shape)
    #     # print("xi shape: ", xi.shape)

    # xy_upper_bound = np.array([3,3])
    # xy_lower_bound = np.array([-3,-3])
    # width = 3
    # x_ax = np.linspace(start=xy_lower_bound[0], stop=xy_upper_bound[0], num=width)
    # y_ax = np.linspace(start=xy_lower_bound[1], stop=xy_upper_bound[1], num=width)
    # x_grid, y_grid = np.meshgrid(x_ax, y_ax)
    # grid_points = np.concatenate((x_grid[:,:,np.newaxis], y_grid[:,:,np.newaxis]), axis=-1)
    # grid_points = grid_points.reshape((-1,2))

    
    