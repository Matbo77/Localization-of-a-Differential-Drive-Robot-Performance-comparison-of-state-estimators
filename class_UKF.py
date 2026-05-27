import math 
import numpy as np

import localization_simu_main as loc_simu
from scipy import linalg

# Centralized Kalman Filter
class classUKF:
    def __init__(self,  X0_est, Sigma, Qk , Rk,
        dim_x=1, dim_z=1, dim_u=1, eval_fx=None,
        eval_hk=None, h_correction = None, state_correction = None
    ):
        """
        Initializes the Unscented Kalman filter creating the necessary matrices
        """
        self.name = "UKF"
        self.X0_est= X0_est  # initial mean state estimate
        self.Xk_list = [X0_est]

        self.Xk_estim = [X0_est]
        self.Sigma = Sigma  # covariance state estimate
        self.Sigma_list = [Sigma]
        self.Qk = Qk  # process noise
        self.Rk = Rk  # measurement noise

        self.eval_hk = eval_hk
        self.h_correction = h_correction

        self.eval_fx = eval_fx
        self.state_correction = state_correction

        self.Yk_list = []
        self.Uk_list = []

        self.nu = dim_u
        self.n = dim_x
        self.ny = dim_z
        self.nw = self.Qk.shape[0]

        self.K = []
        self.mean_CT = 0 # computation time

    def init(self,Yk_list,Uk_list,Xk_list):
        
        self.Yk_list = Yk_list
        self.Uk_list = Uk_list


    def predict(self,t, Xk, Uk,Te):
        # Predict the next state 

        Sigma_aug = linalg.block_diag(self.Sigma,self.Qk)

        Xk_aug =  np.append(Xk,np.zeros((self.nw,1)))

        #Sigma_points
        #sigma_points = self.compute_sigma_points(Xk_aug,Sigma_aug)
        L = Sigma_aug.shape[0]

        #gaussian process
        alpha_UKF = 1e-1 # 1e-3 1e-2 1e-1  # this tunable parameter control the dispersion (generally between 1e-4 and 1)
        # of the sigma point around the mean
        beta = 2
        kappa = 0
        lambda_a = alpha_UKF**2*(L+kappa)-L

        #Weights
        Wm_0 = lambda_a/(L+lambda_a)
        Wc_0 = Wm_0 + (1-alpha_UKF**2 + beta)
        Wm_i = 1/(2*(L+lambda_a)) # there was an error
        Wc_i = Wm_i

        sqrt_Sigma = math.sqrt(L+lambda_a)*linalg.sqrtm(Sigma_aug)
        #sqrt_Sigma = math.sqrt(L+lambda_a)*linalg.cholesky(Sigma,lower=False)

        sigma_points = np.zeros((L, 2 * L + 1))
        sigma_points[:, 0] = Xk_aug  # Point central

        
        for i in range(L):
            sigma_points[:, 1 + i] = Xk_aug + sqrt_Sigma[:, i]  
            sigma_points[:, 1 + L + i] = Xk_aug - sqrt_Sigma[:, i]  


        
        sigma_pred = np.zeros((self.n, 2 * L + 1))
        sigma_pred[:, 0] = self.eval_fx(t,sigma_points[:self.n,0].T,Uk,sigma_points[self.n:,0].T,Te,self.n)
        sigma_pred[:, 0] = self.state_correction(sigma_pred[:, 0])

        for i in range(0,2*L):
            
            sigma_pred[:, 1 + i] = self.eval_fx(t,sigma_points[:self.n,1+i].T,Uk,sigma_points[self.n:,1+i].T,Te,self.n)
            sigma_pred[:, 1 + i] = self.state_correction(sigma_pred[:, 1 + i])


        Xk_pred = Wm_0*sigma_pred[:, 0] + Wm_i*sigma_pred[:, 1:].sum(axis=1)


        #diff_chi_X_pred = [sigma_pred[:,c] - Xk_pred for c in range(sigma_pred.shape[1])]
        diff_chi_X_pred = sigma_pred - Xk_pred[:, None]

        #for column in range(diff_chi_X_pred.shape[1]):
        #    diff_chi_X_pred[:,column] = self.state_correction(diff_chi_X_pred[:,column])

        diff_chi_X_pred = self.state_correction(diff_chi_X_pred) # vectorize way

        #print(self.Sigma)  
        self.Sigma = self.cov_pond_diff(diff_chi_X_pred,diff_chi_X_pred,Wc_0,Wc_i)
        #print(self.Sigma)
        #print("Xk_pred=",Xk_pred)

        return Xk_pred, self.Sigma

    def update(self,t,Xk_pred,X_anchors,Zmes):
        # Update the state prediction with the current measurement


        #ny = Zmes.shape[0]

        L = self.Sigma.shape[0]

        #gaussian process
        alpha_UKF = 1e-1 # 1e-3 1e-2 1e-1  # this tunable parameter control the dispersion (generally between 1e-4 and 1)
        # of the sigma point around the mean
        beta = 2
        kappa = 0
        lambda_a = alpha_UKF**2*(L+kappa)-L

        #Weights
        Wm_0 = lambda_a/(L+lambda_a)
        Wc_0 = Wm_0 + (1-alpha_UKF**2 + beta)
        Wm_i = 1/(2*(L+lambda_a)) # there was an error
        Wc_i = Wm_i


        sqrt_Sigma = math.sqrt(L+lambda_a)*linalg.sqrtm(self.Sigma)
        #sqrt_Sigma = math.sqrt(L+lambda_a)*linalg.cholesky(Sigma,lower=False)

        sigma_points = np.zeros((L, 2 * L + 1))
        sigma_points[:, 0] = Xk_pred  # Point central

        #Compute sigma points
        for i in range(L):
            sigma_points[:, 1 + i] = Xk_pred + sqrt_Sigma[:, i]  
            sigma_points[:, 1 + L + i] = Xk_pred - sqrt_Sigma[:, i]  

        # Evaluate the expected measurement and compute the residual, then update the state prediction
        sigma_z = np.zeros((self.ny, 2 * L + 1))
        sigma_z[:, 0] = self.eval_hk(t,sigma_points[:,0].T,X_anchors)
        #sigma_z[:, 0] = self.h_correction(sigma_z[:, 0])

        for i in range(0,2*L):
            
            sigma_z[:, 1 + i] = self.eval_hk(t,sigma_points[:,1+i].T,X_anchors)
            #sigma_z[:, 1 + i] = self.h_correction(sigma_z[:, 1 + i])
            # do not normalize before computing a mean

        Z_hat = Wm_0*sigma_z[:, 0] + Wm_i*sigma_z[:, 1:].sum(axis=1)

        diff_chi_Z = sigma_z - Z_hat[:, None]

        diff_chi_Z = self.h_correction(diff_chi_Z)

        diff_chi_X = sigma_points - Xk_pred[:, None]

        diff_chi_X = self.state_correction(diff_chi_X)

        cov_Z = self.cov_pond_diff(diff_chi_Z,diff_chi_Z,Wc_0,Wc_i) + self.Rk

        cov_XZ = self.cov_pond_diff(diff_chi_X,diff_chi_Z,Wc_0,Wc_i)

        #print("cov_Z",cov_Z)
        #print("cov_XZ",cov_XZ)
        #print(cov_XZ.shape)
        #print(cov_Z.shape)


        self.K = cov_XZ @ np.linalg.inv(cov_Z)    

    
        residual = Zmes - Z_hat
        
        #print("res",residual)
        residual = self.h_correction(residual)

        #print("Xk_pred=",Xk_pred)
        #print("K=",self.K)
        Xk = Xk_pred + self.K @ residual
        
        #print("Xk=",Xk)


        Xk = self.state_correction(Xk)
        #print(Xk)

        self.Sigma = self.Sigma - self.K @ cov_Z @ self.K.T
        # self.Sigma = 0.5*(self.Sigma + self.Sigma.T)
        #self.Sigma = self.Sigma - cov_XZ @ np.linalg(cov_Z).T @ cov_XZ.T

        self.Sigma_list.append(self.Sigma)

        return Xk, self.Sigma
    


    def compute_sigma_points(self,X,Sigma):
    
        L = Sigma.shape[0]

        #gaussian process
        alpha_UKF = 1e-1 # 1e-3 1e-2 1e-1  # this tunable parameter control the dispersion (generally between 1e-4 and 1)
        # of the sigma point around the mean
        beta = 2
        kappa = 0
        lambda_a = alpha_UKF^2*(L+kappa)-L

        #Weights
        Wm_0 = lambda_a/(L+lambda_a)
        Wc_0 = Wm_0 + (1-alpha_UKF^2 + beta)
        Wm_i = 1/(2*(L+lambda_a)) # there was an error
        Wc_i = Wm_i

        sqrt_Sigma = math.sqrt(L+lambda_a)*linalg.sqrtm(Sigma)
        #sqrt_Sigma = math.sqrt(L+lambda_a)*linalg.cholesky(Sigma,lower=False)

        sigma_points = np.zeros((L, 2 * L + 1))
        sigma_points[:, 0] = X  # Point central

        # Ajouter les points décalés (+ et - pour chaque colonne)
        for i in range(L):
            sigma_points[:, 1 + i] = X + sqrt_Sigma[:, i]  # X + gamma * sqrt(Sigma)[:, i]
            sigma_points[:, 1 + L + i] = X - sqrt_Sigma[:, i]  # X - gamma * sqrt(Sigma)[:, i]

        return sigma_points
 

    def cov_pond_diff(self,diff_chi_1,diff_chi_2,W_0,W):
        # empirical covariance matrix with
        # UKF ponderation

        _,c = diff_chi_1.shape
        

        #cov_matrix = W_0*diff_chi_1[:,0] @ diff_chi_2[:,0].T
        cov_matrix = W_0*np.outer(diff_chi_1[:,0],diff_chi_2[:,0])

        for i in range(1,c):

            #cov_matrix = cov_matrix + W*diff_chi_1[:,i] @ (diff_chi_2[:,i]).T
            cov_matrix += W*np.outer(diff_chi_1[:,i],diff_chi_2[:,i])

        return cov_matrix

    def cov_pond(self,chi_1,chi_1_mean,chi_2,chi_2_mean,W_0,W):
        """ Compute the UKF weighted covariance """

                # empirical covariance matrix with
        # UKF ponderation

        _,c = chi_1.shape

        cov_matrix = W_0*(chi_1[:,0]-chi_1_mean[:,0]) @ (chi_2[:,0]-chi_2_mean[:,0]).T

        for i in range(1,c):

            cov_matrix = cov_matrix + W*(chi_1[:,i]-chi_1_mean[:,i])  @ (chi_2[:,i]-chi_2_mean[:,i]).T


        return cov_matrix

 # other functions

    # see use of polar transform in UKF