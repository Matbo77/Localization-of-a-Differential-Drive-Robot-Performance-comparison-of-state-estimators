import math 
import numpy as np

import localization_simu_main as loc_simu


# Centralized Kalman Filter
class classEKF:
    def __init__(self,  X0_est, Sigma, Qk , Rk,
        dim_x=1, dim_z=1, dim_u=1, eval_Ak=None, eval_Bk=None, eval_fx=None,
        eval_hk=None, eval_Hk=None, h_correction = None, state_correction = None
    ):
        """
        Initializes the extended Kalman filter creating the necessary matrices
        """
        self.name = "EKF"
        self.X0_est= X0_est  # initial mean state estimate
        self.Xk_list = [X0_est]

        self.Xk_estim = [X0_est]
        self.Sigma = Sigma  # covariance state estimate
        self.Sigma_list = [Sigma]
        self.Qk = Qk  # process noise
        self.Rk = Rk  # measurement noise

        self.eval_Ak = eval_Ak
        self.eval_Bk = eval_Bk

        self.eval_hk = eval_hk
        self.h_correction = h_correction
        self.eval_Hk = eval_Hk

        self.eval_fx = eval_fx
        self.state_correction = state_correction

        self.Yk_list = []
        self.Uk_list = []

        self.K = []
        self.mean_CT = 0 # computation time

        self.nu = dim_u
        self.n = dim_x
        self.ny = dim_z
        self.nw = self.Qk.shape[0]
        
        #self.marker = "--"


    def init(self,Yk_list,Uk_list,Xk_list):
        
        self.Yk_list = Yk_list
        self.Uk_list = Uk_list


    def predict(self,t, Xk, Uk,Te):
        # Predict the next state 

        #nu = Uk.shape[0]
        #self.n = Xk.shape[0] 

        Xk_pred = self.eval_fx(t,Xk,Uk,np.zeros(self.nu),Te,self.n)
        Xk_pred = self.state_correction(Xk_pred)

        # Update the covariance matrix of the state prediction, 
        # you need to evaluate the Jacobians
        Ak = self.eval_Ak(t,Xk_pred,Uk,Te)
        Bk = self.eval_Bk(t,Xk_pred,Uk,Te)
        
        #print(self.Sigma)
        self.Sigma = Ak @ self.Sigma @ Ak.T + Bk @ self.Qk @ Bk.T
        #print(self.Sigma)
        #print("Xk_pred=",Xk_pred)

        return Xk_pred,self.Sigma

    def update(self,t,Xk_pred,X_anchors,Zmes):
        # Update the state prediction with the current measurement
    
        # Compute the Kalman gain, you need to evaluate the Jacobian Ht
        #n = len(Xk_pred) # (n,) = Xk.shape
        Ht = self.eval_Hk(t,Xk_pred,X_anchors)
        #print(Ht.T)

        #print(Ht)
        cov_Z = Ht @ self.Sigma @ Ht.T + self.Rk  # S
        self.K = self.Sigma @ Ht.T @ np.linalg.inv(cov_Z)

        #self.K =  self.Sigma @ Ht.T @ np.linalg.inv(Ht @ self.Sigma @ Ht.T + self.Rk)
        # Evaluate the expected measurement and compute the residual, then update the state prediction
        Z_hat = self.eval_hk(t,Xk_pred,X_anchors)
        #self.y = residual(z, z_hat)
        
        #print("Zmes",Zmes)
        #print("Z_hat",Z_hat)
    
        residual = Zmes - Z_hat
        
        #residual[0:2:] = loc_simu.normalize_angle(residual[0:2:])
        #print("res",residual)
        residual = self.h_correction(residual)

        #print("Xk_pred=",Xk_pred)
        #print("K=",self.K)
        Xk = Xk_pred + self.K @ residual
        
        #print(self.K @ residual)
        #print("Xk=",Xk)

        # Note that I is the identity matrix.
        #I_KH = np.eye(n) - self.K @ Ht
        #self.Sigma = I_KH @ self.Sigma @ I_KH.T + self.K @ self.Rk @ self.K.T

        Xk = self.state_correction(Xk)
        #print(Xk)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature. 
        
        self.Sigma = (np.eye(self.n) - self.K @ Ht) @ self.Sigma 

        #self.Sigma = self.Sigma - self.K @ (Ht @ self.Sigma @ Ht.T + self.Rk) @ self.K.T

        self.Sigma_list.append(self.Sigma)

        return Xk,self.Sigma