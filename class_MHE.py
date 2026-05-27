import math 
import numpy as np
import time
import localization_simu_main as loc_simu

from casadi import *
from casadi.tools import *

import dynamic_meas as dyn_meas

# Centralized Moving Horizon Estimation
class classMHE:
    def __init__(self,  X0_est, Sigma, Qk , Rk, Nhorizon, Te,
        dim_x=1, dim_z=1, dim_u=1, eval_Ak=None, eval_Bk=None, eval_fx=None,
        eval_hk=None, eval_Hk=None, h_correction = None, state_correction = None, 
        eval_fx_casadi=None, eval_hk_casadi=None, h_correction_casadi = None,
        state_correction_casadi = None
    ):
        """
        Initializes the extended Kalman filter creating the necessary matrices
        """
        self.name = "MHE"
        self.X0_est= X0_est  # initial mean state estimate
        self.Xk_list = [X0_est]

        self.Xk_estim = [X0_est]
        self.Sigma = Sigma  # covariance state estimate
        self.Qk = Qk  # process noise
        self.Rk = Rk  # measurement noise

        self.eval_Ak = eval_Ak
        self.eval_Bk = eval_Bk

        self.eval_hk = eval_hk
        self.eval_hk_casadi = eval_hk_casadi
        self.h_correction = h_correction
        self.h_correction_casadi = h_correction_casadi
        self.eval_Hk = eval_Hk

        self.eval_fx = eval_fx
        self.eval_fx_casadi = eval_fx_casadi
        self.state_correction = state_correction
        self.state_correction_casadi = state_correction_casadi

        self.Yk_list = np.array([[]])
        self.Uk_list = []

        self.Nw = 0 # current horizon size
        self.Nhorizon = Nhorizon # horizon size

        self.X0_MHE = X0_est

        self.Yk_horizon = []
        self.Uk_horizon = []

        self.nu = dim_u
        self.n = dim_x
        self.ny = dim_z

        self.Te = Te

        # Option for arrival cost update and warmstart generation
        self.option_UKF = 0 # 0: EKF, 1 : UKF (to do !)

    def init(self,Yk_list,Uk_list,Xk_list):
        
        self.Yk_list = Yk_list
        self.Uk_list = Uk_list

    def predict(self,t, Xk, Uk,Te):
        # Predict the next state 

        #self.Uk_list.append(Uk)

        #nu = Uk.shape[0]
        #self.nu = nu

        Xk_pred = np.array(Xk)
       
        # Init only at time t==0
        if t==0: 
            Xk_pred = self.eval_fx(t,Xk,Uk,np.zeros(self.nu),Te,self.n)
            Xk_pred = self.state_correction(Xk_pred)
            #self.X0_MHE = np.array([Xk_pred]).T


            self.X0_MHE = Xk_pred
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
        

        #self.ny = Zmes.shape[0]
        self.Nw = min(t,self.Nhorizon)

        inv_Q = np.linalg.inv(self.Qk)
        inv_R = np.linalg.inv(self.Rk)

        #self.Yk_list.append([Zmes])
        #self.Yk_list = np.append(self.Yk_list,[Zmes])


        if self.Nw == self.Nhorizon:
            self.Yk_horizon = self.Yk_list[t-self.Nhorizon+1:t+2]
            self.Uk_horizon = self.Uk_list[t-self.Nhorizon+1:t+1]
        else:
            self.Yk_horizon = self.Yk_list[t-self.Nw+1:t+2]
            self.Uk_horizon = self.Uk_list[t-self.Nw+1:t+1]
        
        #print("Zmes:",Zmes)
        #print("Yk_horizon:",self.Yk_horizon)

        X = SX.sym("X", self.n) # [x, y, theta]
        U = SX.sym("U", self.nu) # [v, w]
        W = SX.sym("W", self.nu) # [w_v, w_w]

        #x_next = SX.zeros(3)  # Initialize x_next as a 3D symbolic vector
        #x_next[0] = x[0] + self.Te * cos(x[2]) * (u[0] + w[0])  # Update x
        #x_next[1] = x[1] + self.Te * sin(x[2]) * (u[0] + w[0])  # Update y
        #x_next[2] = x[2] + self.Te * (u[1] + w[1])              # Update theta
    
        #Xk_next = self.RK4_casadi(t, X, U, W, self.Te, self.n)
        Xk_next = self.eval_fx_casadi(t, X, U, W, self.Te, self.n)
        Xk_next = self.state_correction_casadi(Xk_next)
        phi = Function("phi", [X, U, W], [Xk_next], ["x", "u", "w"], ["x_next"])

        # phi = Function("phi", [X,U,W], [vertcat(*Xk_next)],["X", "U", "W"], ["X_next"])
        #["X","U","W"], ["x","y","theta"]

        #print(phi)
        #print(phi(X,U,[0,0]))
        #print(type(phi))

        #
        h_meas = self.eval_hk_casadi(t,X,X_anchors)  # self.h_meas_casadi
        #h_meas = Function("h_meas", [X], Z_meas, ["X"], ["Y"])

        diff_Y = SX.sym("diff_Y", self.ny)
        meas_residual_norm = self.h_correction_casadi(t,diff_Y,X_anchors)

        h_correction_casadi = Function('h_meas_norm', [diff_Y], [meas_residual_norm], ["diff_Y"],["diff_Y_corr"])

        #test_casadi = h_correction_casadi(np.array([6,6,-6,-6]))
        #print("Test h_correction_casadi:",test_casadi)

        self.X0_MHE = np.array(self.X0_MHE) #arrival cost state
        # must be row array !!!
        
        # EKF warmstart
        #x_est_MHE = np.array([self.X0_MHE]) #warmstart
        x_est_MHE = self.run_EKF_horizon(t,self.X0_MHE,self.Sigma,self.Nw,self.Yk_horizon, X_anchors, self.Uk_horizon,1)
        #print(x_est_MHE)

        x_MHE,w_MHE,time_mhe = mhe_non_linear_casadi_solver(phi, h_meas, h_correction_casadi, inv_Q, inv_R,  self.Yk_horizon.T, self.Uk_horizon.T, x_est_MHE.T, self.X0_MHE, self.Sigma, self.Nw)

        #print(x_MHE[-1])
        #print(type(x_MHE[-1]))  # type DM

        #Xk = x_MHE[-1]  # self.X0_MHE.T)[0]
        Xk = np.array(x_MHE[-1].T)[0] # Convert DM to row np.array !
        Xk = self.state_correction(Xk)


        if self.Nw == self.Nhorizon:
                
            ######## Covariance update
            # partial derivation 
            Hk = self.eval_Hk(t,self.X0_MHE,X_anchors)
            Ak = self.eval_Ak(t,self.X0_MHE,self.Uk_horizon[0],self.Te)   #,DM.zeros(Nu, 1)
            Bw = self.eval_Bk(t,self.X0_MHE,self.Uk_horizon[0],self.Te)

            # Cassadi DM type
            #K_MHE = mtimes([self.Sigma,Hk.T,np.linalg.inv(mtimes([Hk,self.Sigma,Hk.T]) + self.Rk)])
            #P_MHE = mtimes((DM.eye(self.n)-mtimes(K_MHE,Hk)),self.Sigma)   # DM type
            #self.Sigma = mtimes(mtimes((Ak,P_MHE)),Ak.T) + mtimes(mtimes(Bw,self.Qk),Bw.T)
            
            K_MHE = self.Sigma @ Hk.T @ np.linalg.inv(Hk @ self.Sigma @ Hk.T + self.Rk)
            Sigma_MHE = (np.eye(self.n) - K_MHE @ Hk) @ self.Sigma
            self.Sigma = Ak @ Sigma_MHE @ Ak.T + Bw @ self.Qk @ Bw.T


                
            self.X0_MHE  = x_MHE[:][1]
                
            #x0_MHE = phi(x_MHE[:][0], u_sim[:,k], DM.zeros(Nu, 1))
        else:
            self.X0_MHE  = x_MHE[0]
            # no cov update

        # Convert from DM to row array
        self.X0_MHE = np.array(self.X0_MHE.T)[0]
        
        #print(Xk)

        # Update cov 
        #self.update_cov()
        
        #self.Sigma = (np.eye(n) - self.K @ Ht) @ self.Sigma 

        #self.Sigma = self.Sigma - self.K @ (Ht @ self.Sigma @ Ht.T + self.Rk) @ self.K.T

        return Xk,self.Sigma

# The arrival cost is updated using the smoothed EKF update.


    def run_EKF_horizon(self,t,Xk,Pk,Nhorizon,Y_horizon,X_anchors,U_horizon,flag_correction):
        
        if flag_correction:  # k = 0
            Xk,Pk = self.EKF_correction(t,Xk,Pk,Y_horizon[0],X_anchors)

        Xk_horizon = [Xk]

        for k in range(0,Nhorizon):

            Xk,Pk = self.EKF_prediction(t+k,Xk,Pk,U_horizon[k])

            Xk,Pk = self.EKF_correction(t+k,Xk,Pk,Y_horizon[k+1],X_anchors)

            Xk_horizon.append(Xk)


        return np.array(Xk_horizon)
        
    def EKF_prediction(self,t,Xk,Pk,Uk):

        nu = Uk.shape[0]
        self.n = Xk.shape[0] 

        Xk_pred = self.eval_fx(t,Xk,Uk,np.zeros(nu),self.Te,self.n)
        Xk_pred = self.state_correction(Xk_pred)

        # Update the covariance matrix of the state prediction, 
        # you need to evaluate the Jacobians
        Ak = self.eval_Ak(t,Xk_pred,Uk,self.Te)
        Bk = self.eval_Bk(t,Xk_pred,Uk,self.Te)
        
        #print(self.Sigma)
        Pk = Ak @ Pk @ Ak.T + Bk @ self.Qk @ Bk.T
        #print(self.Sigma)
        #print("Xk_pred=",Xk_pred)

        return Xk_pred,Pk

    def EKF_correction(self,t,Xk_pred,Pk,Zmes,X_anchors):

        Ht = self.eval_Hk(t,Xk_pred,X_anchors)
        #print(Ht.T)

        S = Ht @ self.Sigma @ Ht.T + self.Rk
        K = self.Sigma @ Ht.T @ np.linalg.inv(S)

        # Evaluate the expected measurement and compute the residual, then update the state prediction
        Z_hat = self.eval_hk(t,Xk_pred,X_anchors)
    
        residual = Zmes - Z_hat
        residual = self.h_correction(residual)


        Xk = Xk_pred + K @ residual

        Xk = self.state_correction(Xk)

        Pk = (np.eye(self.n) - K @ Ht) @ Pk 

        #self.Sigma = self.Sigma - self.K @ (Ht @ self.Sigma @ Ht.T + self.Rk) @ self.K.T

        return Xk,Pk
    
    def update_cov(self,x_MHE,P_MHE,Uk,t,C):
     
        if self.Nhorizon==self.Nw:
            # Cov update
            K_MHE = mtimes([P_MHE ,C.T,np.linalg.inv(mtimes([C,P_MHE ,C.T]) + self.Rk)])
            P_MHE = mtimes((DM.eye(self.n)-mtimes(K_MHE,C)),P_MHE)
            
            # partial derivation 
            #Ak =  PHI(x0_MHE,u_sim[:,k])   #,DM.zeros(Nu, 1)
            #Bw = PSI(x0_MHE)

            Ak  = self.eval_Ak(t,x_MHE,Uk,self.Te)
            Bw = self.eval_Bk(t,x_MHE,Uk,self.Te)

            P_MHE = mtimes(mtimes((Ak,P_MHE)),Ak.T) + mtimes(mtimes(Bw,self.Qk),Bw.T)
                
            x0_MHE  = x_MHE[:][1]
                
            #x0_MHE = phi(x_MHE[:][0], u_sim[:,k], DM.zeros(Nu, 1))
        else:
            x0_MHE  = x_MHE[:][0]
            # no cov update

        return x0_MHE, P_MHE

def mhe_non_linear_casadi_solver(phi, h_meas, h_correction_casadi , inv_Q, inv_R, y_sim, u_sim , x_est, x0_MHE, P0_MHE, Nhorizon):
    # Entries:
    # dynamic f, Ak
    # measure h, Ck
    # inv_Q, inv_R
    # y_sim, u_sim 
    # x0_MHE, P0_MHE,  arrival cost prior state and covariance 
    # N : horizon size
    # x_est : warmstart solution
 
    # Output :
    # x_MHE,w_MHE


    # # Dimensions
    Nx = x0_MHE.shape[0] # Dim state
    Nu = u_sim.shape[0]  # Dim input
    Ny = y_sim.shape[0] # Dim output
    Nw = inv_Q.shape[0] # Dim process noise
    

    # Initialisation MHE
    shooting = struct_symSX([
        entry("X", repeat=Nhorizon+1, shape=(Nx, 1)),
        entry("W", repeat=Nhorizon, shape=(Nw, 1))
    ])
    parameters = struct_symSX([
        entry("U", repeat=Nhorizon, shape=(Nu, 1)),
        entry("Y", repeat=Nhorizon+1, shape=(Ny, 1)),
        entry("Pi", shape=(Nx, Nx)), # inv_P
        entry("x0", shape=(Nx, 1))
    ])
    
    # Building MHE objective / cost function
    obj = 0
    obj += mtimes([(shooting["X", 0] - parameters["x0"]).T, parameters["Pi"], (shooting["X", 0] - parameters["x0"])])
    for k in range(Nhorizon+1):

        v = parameters["Y", k] - h_meas(shooting["X", k])
        v = h_correction_casadi(v)
        obj += mtimes([v.T, inv_R, v])

    #print(v.shape)
    for k in range(Nhorizon):
        #print(shooting["W", k].shape)
        #print(mtimes([shooting["W", k].T, inv_Q, shooting["W", k]]))
        obj += mtimes([shooting["W", k].T, inv_Q, shooting["W", k]])
    
    # Constraints
    g = [] # equality constraint
    for k in range(Nhorizon):
        g.append(shooting["X", k+1] - phi(shooting["X", k], parameters["U", k], shooting["W", k]))
    
    # Optimisation problem
    nlp = {"x": shooting, "p": parameters, "f": obj, "g": vertcat(*g)} #  
    #opts = {"ipopt.print_level": 1, "print_time": True}  # 'ipopt.max_iter':100 #  "ipopt.tol":1e-12
    opts = {"ipopt.print_level": 0, "print_time":False}
    nlpsolver = nlpsol("nlpsol", "ipopt", nlp, opts)
    
    
    # set parameters
    current_params = parameters(0)

    current_params["Y", lambda x: horzcat(*x)] = y_sim[:,0:Nhorizon+1] #goto Nhorizon
    current_params["Pi"] = inv(P0_MHE)
    current_params["x0"] = x0_MHE
    
    #warmstart
    init_guess = shooting(0)
    init_guess["X", lambda x: horzcat(*x)] = x_est[:, 0:Nhorizon+1]

    
    if Nhorizon>0:
        current_params["U", lambda x: horzcat(*x)] = u_sim[:,0:Nhorizon]
        init_guess["W", lambda x: horzcat(*x)] = DM.zeros(Nw, Nhorizon) # centered on zero
        # initialize with EKF implicit solution
        
    start_time = time.time()
    res = nlpsolver(x0=init_guess, p=current_params, lbg=0, ubg=0)
    end_time = time.time()
    #time_mhe.append(end_time - start_time)
    time_mhe = end_time - start_time
    
    sol = shooting(res["x"])
    x_MHE = sol["X"]
    w_MHE = sol["W"]
      

    
    return x_MHE,w_MHE,time_mhe


def mhe_non_linear_casadi_solver_optistack(phi, h_meas, h_correction_casadi , inv_Q, inv_R, y_sim, u_sim , x_est, x0_MHE, P0_MHE, Nhorizon):

   # MHE solver implemented with Casadi Opti stack
   # to FINISH !

    opti = Opti()

    # # Dimensions
    Nx = x0_MHE.shape[0] # Dim state
    Nu = u_sim.shape[0]  # Dim input
    Ny = y_sim.shape[0] # Dim output
    Nw = inv_Q.shape[0] # Dim process noise
    
    X = opti.variable(Nx,Nhorizon+1)
    W = opti.variable(Nw,Nhorizon)

    Ymes = opti.parameter()
    U = opti.parameter()
    X0 = opti.parameter()
    opti.set_value(Ymes, y_sim)
    opti.set_value(U, u_sim)
    opti.set_value(X0, x0_MHE)

    opti.set_initial(X, x_est)
    opti.set_initial(W, np.zeros((Nw,Nhorizon)))




    p_opts = {"expand":True}
    s_opts = {"max_iter": 100}
    opti.solver("ipopt",p_opts,
                    s_opts)
    
    sol = opti.solve()




    X_sol = sol.value(X)
    W_sol = sol.value(W)

    return X_sol,W_sol

    # Phi_k = SX.eye(Nx)
    # Phi_k[0,2] = -sin(x[2])*dt*u[0]
    # Phi_k[1,2] = cos(x[2])*dt*u[0]
    
    # B_k = SX.zeros(Nx,2)
    # B_k[0,0] = cos(x[2])*dt
    # B_k[1,0] = sin(x[2])*dt
    # B_k[2,1] = dt
    
    # PHI = Function('PHI', [x, u], [Phi_k], ["x", "u"], ['Ak']) 
    
    # PSI = Function('PSI', [x], [B_k], ["x"], ['Bk']) 
    
    # C = DM([[1,0,0],[0,1,0]])  
    # y = mtimes(C, x)
 
    
    


