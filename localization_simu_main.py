
import matplotlib.pyplot as plt
import numpy as np
import math
#import control as ct

from scipy import linalg, signal #, matrix
from scipy.stats import chi2

import warnings
import time

from casadi import *
from casadi.tools import *

import dynamic_meas as dyn_meas
import class_EKF
import class_UKF
import class_MHE
import class_PF

class localization_simu:
    """Implementing localization simulations and perform state estimation"""

    def __init__(self, N_MC, Nsim, domain, eval_hk, h_correction, Te):
        self.X0 = []
        self.Nsim = Nsim
        self.N_MC = N_MC
        self.delta = 1
        self.domain = domain
        self.Xk_list  = []
        self.Uk_list  = []
        self.Yk_list  = []
        self.X_anchors = []
        self.observers_list = []
        self.Te = Te
        self.eval_hk = eval_hk
        self.h_correction = h_correction
    
    def run_sim(self, X0, X_anchors, Rk, Qk):
        """ Simulate the system dynamic and take measurements """
        self.X0 = X0
        self.Xk_list = [X0]
        self.X_anchors = X_anchors
        flag_discretization = 4 # for RK4 


        Xk_ref = [[15, 16]]*(self.Nsim+1)

        # bounds Turtlebot3 Burger
        #v_max = 0.22  # 0.22 m/s
        v_max = 2.0
        w_max = 2.84 #2.84  # 1.84 or 2.84 rad/s -> 162 deg/s

        # v_max = 5
        # w_max = 2
        
        #u_bound = np.matrix(([v_max, w_max])).T
        u_bound = np.array([v_max, w_max])

        
        ny = Rk.shape[0]
        Wk = np.random.multivariate_normal([0, 0], Qk, size=self.Nsim)
        Vk = np.random.multivariate_normal([0]*ny, Rk, size=self.Nsim+1)

        #print(Vk) 
        #print(np.cov(Vk).shape)
        #print(np.cov(Vk.T))


        Y0 = self.eval_hk(0,X0,self.X_anchors) + Vk[0]
        self.Yk_list.append(Y0)
        #print(self.Yk_list)

        for k in range(0,self.Nsim):
            
            Xk_current = self.Xk_list[k]

            Uk = self.diff_drive_proportionnal_control(Xk_current, Xk_ref[k+1], u_bound, self.Te)
            #Uk = np.array([0,0]) # free regime (uncontrolled)

            #print("Xk_current",Xk_current)

            Xk_next = self.propagate_dynamic(k,Xk_current,[Uk],[Wk[k]],1,self.Te,flag_discretization)[-1]
            Xk_next  = f_dyn_correction(Xk_next)
            #print(Xk_next)

            self.Uk_list.append(Uk) #command without noise
            self.Xk_list.append(Xk_next)

            Yk_next = self.eval_hk(k,Xk_next,self.X_anchors) + Vk[k+1]
            Yk_next = self.h_correction(Yk_next)
            self.Yk_list.append(Yk_next)

        # Convert to np.array
        self.Xk_list = np.array(self.Xk_list)
        self.Yk_list = np.array(self.Yk_list)
        self.Uk_list = np.array(self.Uk_list)

        return self.Xk_list,self.Yk_list,self.Uk_list

    def run_estim(self,X_anchors, estimator):
        """ Perform state estimation for a given estimator class """

        # X0_est
        Xest_list = [estimator.X0_est]
        Xk = estimator.X0_est

        #print(estimator.X0_est)

        estimator.init(self.Yk_list,self.Uk_list,self.Xk_list)

        for k in range(0,self.Nsim):

            #print("Time:",k) #" ",self.Uk_list[k]
            Xk_pred,Sigma_pred = estimator.predict(k, Xk, self.Uk_list[k], self.Te)


            Xk,Sigma = estimator.update(k,Xk_pred,X_anchors,self.Yk_list[k+1])
        
            Xest_list.append(Xk)

        # Convert to np.array
        Xest_list = np.array(Xest_list)

        estimator.Xk_estim = Xest_list

        return Xest_list

    

    def lqr_control(self,Xk, Xk_ref, u_previous, u_bound, Q, R, Te):
    
        # for diff drive TB3

        theta = Xk[2, 0].item()
        v = u_previous[0,:].item()
        
        A = np.array(([1, 0, -np.sin(theta)*v*Te], [0, 1, np.cos(theta)*v*Te], [0, 0 , 1]))

        B = np.array(([np.cos(theta), 0], [np.sin(theta), 0], [0 , 1]))*Te
        
        #Ctrb = ctrl.ctrb(A, B)
        #print(np.linalg.matrix_rank(Ctrb))
        
        # Calcul du gain LQR
        #P_c = linalg.solve_continuous_are(A, B, Q, R)
        
        P = linalg.solve_discrete_are(A, B, Q, R)
        K_LQ = np.linalg.inv(R) @ B.T @ P

        
        """Calcule la commande LQR"""
        u_k = -K_LQ @ (Xk - Xk_ref)

        #print(u_k)
        return np.clip(u_k, -u_bound, u_bound )  # Limites du TurtleBot3 # saturation
    

    def diff_drive_proportionnal_control(self,Xk, Xk_ref, u_bound, Te):
        # artificial heading theta ref

        diff_x = Xk_ref[0] - Xk[0]
        diff_y = Xk_ref[1] - Xk[1]
        theta_des = math.atan2(diff_y, diff_x)
        dist = math.sqrt(diff_x**2 + diff_y**2) 

        v_max = u_bound[0]
        w_max = u_bound[1]

        # d_theta_max = 0.5*math.pi   # 0.5*math.pi
        # d_xy_max = 0.8  # (m)   # 1  0.5

        # slow to avoid too many oscillations at the end
        d_theta_max = 1.3*math.pi   # 
        d_xy_max = 0.6  # (m)   # 

        K_theta = w_max/d_theta_max 
        K_v = v_max/d_xy_max 

        # K_theta = 1
        # K_v = 0.2  

        diff_theta = theta_des - Xk[2]

        diff_theta = dyn_meas.normalize_angle(diff_theta)
    
        w_k = K_theta * diff_theta 
        v_k = K_v * dist
        
        u_k = np.array([v_k,w_k])
        #u_k = np.concatenate((v_k,w_k),axis=0)
        
        return np.clip(u_k, -u_bound, u_bound )



    def propagate_dynamic(self,t,X0,Uk,Wk,N_h,Te,flag_discretization):
        """ Propagate the dynamic """
        # Uk : list of input vector
        # Nh : horizon length

        Xk_horizon = [X0]
        Xk = X0
        n = len(X0)

        if flag_discretization == 1:
            for k in range(N_h):

                Xk = Euler_explicit(t+k,Xk,Uk[k],Wk[k],Te,n)
                #Xk_horizon = [Xk_horizon, Xk]
                Xk_horizon.append(Xk) 
        elif flag_discretization ==2:
            for k in range(N_h):

                Xk = RK2(t+k,Xk,Uk[k],Wk[k],Te,n)
                #Xk_horizon = [Xk_horizon, Xk]
                Xk_horizon.append(Xk) 
        else: #flag_discretization == 4
            for k in range(N_h):

                Xk = RK4(t+k,Xk,Uk[k],Wk[k],Te,n)
                #Xk_horizon = [Xk_horizon, Xk]
                Xk_horizon.append(Xk) 
        
        return Xk_horizon

    def RMSE(self, Xk_estim, str_name):

        diff_X_estim = self.Xk_list - Xk_estim
        # 

        # # Compute the norms of the errors for each time step
        #norm_X_estim = np.linalg.norm(diff_X_estim[:, :2], axis=1)  # Error norm of the first two elements (pos x and y)

        #norm_X_estim = np.linalg.norm(diff_X_estim[:,:], axis=1) # Error norm of all state vector

        #Correction in the RMSE taking care of angle modulus for instance
        #norm_X_estim = np.linalg.norm(f_dyn_correction(diff_X_estim.T), axis=1)  # Error norm of all state vector with correction
        norm_X_estim = np.linalg.norm(f_dyn_correction(diff_X_estim.T), axis=0)

    
        print(f'{str_name}: RMSE={np.mean(norm_X_estim)}, fRMSE={np.mean(norm_X_estim[-5:])} ')

        # Temps = {metrics['Time']:.4f}s

    def RMSEpos(self, Xk_estim, str_name):

        diff_X_estim = self.Xk_list - Xk_estim
        # 

        # # Compute the norms of the errors for each time step
        norm_X_estim = np.linalg.norm(diff_X_estim[:, :2], axis=1) 
         # Error norm of the first two elements (pos x and y)

    
        print(f'{str_name}: RMSEpos={np.mean(norm_X_estim)}, fRMSEpos={np.mean(norm_X_estim[-5:])} ')


# other functions

def eval_Ak(t,Xk,Uk,Te):
    """ Linearized discretized dynamic matrix / jacobian"""
    # Xk = [xk,yk,thetak], state at k
    # Uk = [v_k, w_k]
    # X_anchor = [xpk,ypk], anchor position matrix at time k
    # t : time
    
    #(n,_) = len(Xk) #.shape
    #A = np.array(([1, 0, -np.sin(theta)*v*Te], [0, 1, np.cos(theta)*v*Te], [0, 0 , 1]))

     
    Ak = np.array([[1, 0, -np.sin(Xk[2])*Uk[0]*Te], [0, 1, np.cos(Xk[2])*Uk[0]*Te], [0, 0 , 1]])

    return Ak

def eval_Bk(t,Xk,Uk,Te):
    """ Linearized discretized input matrix / jacobian"""
    # Xk = [xk,yk,thetak], state at k
    # X_anchor = [xpk,ypk], anchor position matrix at time k
    # t : time

    #B = np.array(([np.cos(theta), 0], [np.sin(theta), 0], [0 , 1]))*Te
    Bk = np.array([[np.cos(Xk[2])*Te, 0], [np.sin(Xk[2])*Te, 0], [0 , Te]])

    return Bk


def f_dyn(t,Xk,Uk,Wk):
    # Xk = [xk,yk,thetak], state at k
    # uk = [vk,wk], command at k
    # t : time (in this example f_dyn is not time-varying)
         
    dXk = np.array([(Uk[0]+Wk[0])*math.cos(Xk[2]), (Uk[0]+Wk[0])*math.sin(Xk[2]), Uk[1] + Wk[1]]) 
    #dXk = [ uk[0]*math.cos(Xk[2]), uk[0]*math.sin(Xk[2]), uk[1] ]

    return dXk

def f_dyn_correction(Xk):
    """Normalize the state heading angle """
    Xk[2] = dyn_meas.normalize_angle(Xk[2])
    return Xk

def f_dyn_correction_casadi(Xk):
    """Normalize the state heading angle with casadi compatible function"""
    Xk[2] = dyn_meas.normalize_angle_casadi(Xk[2])
    return Xk


def Euler_explicit(t,Xk,uk,wk,Te,n):

    dXk = f_dyn(t,Xk,uk,wk)

    #Xk_1 = Xk + Te*dXk
    Xk_1 = [Xk[i] + Te*dXk[i] for i in range(n) ]

    return Xk_1


def RK2(t,Xk,uk,wk,Te,n):

    k_1 = f_dyn(t,Xk,uk,wk)
    Xk_12 = [Xk[i] + Te/2*k_1[i] for i in range(n) ]
    #k_2 = self.f_dyn(0,Xk + Te/2*k_1,uk)
    k_2 = f_dyn(t,Xk_12,uk,wk)

    #Xk_1 = Xk + Te*k_2
    Xk_1 = [Xk[i] + Te*k_2[i] for i in range(n) ]

    return Xk_1

def RK4(t,Xk,uk,wk,Te,n):
    """ Runge-Kutta order 4"""
    # Xk : state at k
    # uk : command at k
    # Te : sampling period
    # Xk_1 : state at k+1 
    k_1 = f_dyn(t,Xk,uk,wk)
    Xk_12 = [Xk[i] + Te/2*k_1[i] for i in range(n) ] # len(Xk)
    k_2 = f_dyn(t,Xk_12,uk,wk)
    Xk_23 = [Xk[i] + Te/2*k_2[i] for i in range(n) ]
    k_3 = f_dyn(t,Xk_23,uk,wk)
    Xk_34 = [Xk[i] + Te*k_3[i] for i in range(n) ]
    k_4 = f_dyn(t,Xk_34,uk,wk)

    #Xk_1 = Xk + Te/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
    Xk_1 = [Xk[i] + Te/6*(k_1[i] + 2*k_2[i] + 2*k_3[i] + k_4[i]) for i in range(n) ]

    return Xk_1


def plot_traj(Xk_list,X_anchors,Xk_estim):
    
    n_anchors = X_anchors.shape[0]

    fig, ax = plt.subplots()
    
    #for anchor in X_anchors: 
        #ax.scatter(anchor[0],anchor[1], marker='x', c='red',  label='anchor')

    for i in range(n_anchors):
        name = 'anchor ' + str(i)
        ax.scatter(X_anchors[i][0],X_anchors[i][1], marker='x',  label=name)
    
    ax.scatter(Xk_list[0,0],Xk_list[0,1], marker='o', c='green', label='X0')
    ax.plot(Xk_list[:,0],Xk_list[:,1], '--.', label='Traj')

    if len(Xk_estim)>0:
        ax.scatter(Xk_estim[0,0],Xk_estim[0,1], marker='o', label='X0 estim')
        ax.plot(Xk_estim[:,0],Xk_estim[:,1], '--.', label='Traj estim')

            
    ax.set_title('Trajectory sketch')
    ax.set(xlabel='Position x [m]', ylabel='Position y [m]')
    ax.legend()
    ax.grid(True)
    plt.ion()
    plt.show()
    #plt.show(block=True)

def plot_state_estimate(Xk_list,estimator):

    Xk_estimate = estimator.Xk_estim.copy()
    name_estim = estimator.name

    fig, axs = plt.subplots(3)

    axs[0].plot(Xk_list[:,0], '-o', label='True')
    axs[0].plot(Xk_estimate[:,0], '-o', label=name_estim)
    axs[0].set_title('Position x')
    axs[0].set(xlabel='time step', ylabel='Position x [m]') #  [s]

    axs[1].plot(Xk_list[:,1], '-o', label='True')
    axs[1].plot(Xk_estimate[:,1], '-o', label=name_estim)
    axs[1].set(xlabel='time step', ylabel='Position y [m]') #  [s]
    axs[1].set_title('Position y')

    axs[2].plot(Xk_list[:,2]*180/math.pi, '-o', label='True')
    axs[2].plot(Xk_estimate[:,2]*180/math.pi, '-o', label=name_estim)
    axs[2].set_title('Heading angle theta')
    axs[2].set(xlabel='time step', ylabel='theta [deg]')  # [rad]  # [s]
    

    for ax in axs.flat:
        ax.legend()
        ax.grid(True)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.ion()
    #plt.ioff()
    plt.show()

def plot_norm_error(Xk_list,estimator):

    Xk_estimate = estimator.Xk_estim.copy() 
    name_estim = estimator.name

    fig, ax = plt.subplots()

    Xk_error = Xk_list - Xk_estimate 
    Xk_error_norm = linalg.norm(Xk_error, axis=1)

    ax.plot(Xk_error_norm, '-o', label=name_estim)
    ax.set_title('Norm error')
    ax.set(xlabel='time step', ylabel='Norm error')
    ax.legend()
    ax.grid(True)

    plt.ion()
    #plt.ioff()
    plt.show()

def main():
    
    # Kinodynamic motion planning
    # (Optimal) Kinodynamic Planning 

    #For repeatability
    np.random.seed(0)
    #np.random.seed(1)
    #np.random.seed(2) # error EKF

    low_level = 1e-5 # low level but avoid zero error

    #X0 = np.array([-2.0,-2.0,0.0])
    X0 = np.array([1.0,1.0,0.0]) #math.pi/2  #start point
    #X0 = np.array([10,12,0]) 
    Sigma_0 = np.diag([2.0**2,1.0**2,(5.0*math.pi/180)**2])
    #Sigma_0 = np.diag([low_level**2,low_level**2,low_level**2]) #nearly no uncertainty, for debugging filter
    X0_est = np.random.multivariate_normal(X0, Sigma_0, size=1)[0]
    # [0]*3

    #Not to sensitive to initial condition (X0,Sigma_0)

    Te = 0.1 # 0.1
    len_map = 20
    Nsim = 40  # 60 30 6
    N_MC = 10 

    Qk = np.diag([0.01**2, 0.01**2])
    #Qk = np.diag([low_level**2, low_level**2]) #nearly no noise, for debugging filter

    X_anchors = np.array([[10, 4], [8, 14]])  # Positions des 2 ancres   # [7,10]
    Rk = np.diag([0.5**2, (2*math.pi/180)**2]*2)
    #Rk = np.diag([low_level**2, low_level**2]*2) # nearly no noise, for debugging filter
    #Rk = np.diag([5**2, (100*math.pi/180)**2]*2) # high noise
    # But very sensitive to noise especially on angular measure

    Rk_lin = np.diag([0.5**2,0.5**2])

    Nhorizon_MHE = 4  
    #Nhorizon_MHE = Nsim
    N_particles = 1500 #1500 #50 # 30
    #print(math.dist(X0[0:2],X_anchors[0]))
    #print(math.dist(X0[0:2],X_anchors[1]))

    EKF = class_EKF.classEKF(X0_est, Sigma_0, Qk , Rk,
        dim_x=3, dim_z=4, dim_u=2, eval_Ak=eval_Ak, eval_Bk=eval_Bk, eval_fx=RK4,
        eval_hk=dyn_meas.h_meas, eval_Hk=dyn_meas.eval_Hk, h_correction = dyn_meas.h_meas_correction, state_correction = f_dyn_correction)
    
    UKF = class_UKF.classUKF(X0_est, Sigma_0, Qk , Rk,
        dim_x=3, dim_z=4, dim_u=2, eval_fx=RK4,
        eval_hk=dyn_meas.h_meas, h_correction = dyn_meas.h_meas_correction, state_correction = f_dyn_correction)
    
    MHE = class_MHE.classMHE(X0_est, Sigma_0, Qk, Rk, Nhorizon_MHE, Te,
        dim_x=3, dim_z=4, dim_u=2, eval_Ak=eval_Ak, eval_Bk=eval_Bk, eval_fx=RK4,
        eval_hk=dyn_meas.h_meas, eval_Hk=dyn_meas.eval_Hk, h_correction = dyn_meas.h_meas_correction, state_correction = f_dyn_correction,
        eval_fx_casadi=dyn_meas.RK4_casadi, eval_hk_casadi=dyn_meas.h_meas_casadi, h_correction_casadi = dyn_meas.h_meas_correction_casadi, state_correction_casadi = f_dyn_correction_casadi)
    
    PF = class_PF.classPF(X0_est, Sigma_0, Qk, Rk, N_particles,
        dim_x=3, dim_z=4, dim_u=2, eval_fx=RK4,
        eval_hk=dyn_meas.h_meas, h_correction = dyn_meas.h_meas_correction, state_correction = f_dyn_correction)

    #verify Rk dimension corresponds with eval_hk

    # list_estimator = ["EKF"] # , MHE, Particle_filter
    #simulator = MonteCarloSimulator(n_simulations=10, n_steps=100, anchors=anchors)
    #results = simulator.compare_estimators()


    simu = localization_simu(N_MC, Nsim, len_map, dyn_meas.h_meas, dyn_meas.h_meas_correction, Te)

    #simu.run_MonteCarlo()

    Xk_list, Yk_list, Uk_list = simu.run_sim(X0, X_anchors, Rk, Qk)

    #print(Yk_list[0:3])
    #plot_traj(Xk_list,X_anchors,np.array([]))


    Xk_EKF = simu.run_estim(X_anchors, EKF)

    Xk_UKF = simu.run_estim(X_anchors, UKF)

    Xk_MHE = simu.run_estim(X_anchors, MHE)

    Xk_PF = simu.run_estim(X_anchors, PF)

    plt.pause(2)

    #print(Xk_list)
    #print(Xk_EKF)
    #print("--------ESTIM------")
    print("Comparaison des estimateurs :")

    #plot_traj(Xk_list,X_anchors,[])

    # Error with plot_traj_meas it chnages the data 
    #dyn_meas.plot_traj_meas(EKF,X_anchors,Xk_list,Yk_list,plot_ellipse=True)

    dyn_meas.plot_traj_meas(UKF,X_anchors,Xk_list,Yk_list,plot_ellipse=True)

    dyn_meas.plot_traj_meas(MHE,X_anchors,Xk_list,Yk_list,plot_ellipse=False)

    dyn_meas.plot_traj_meas(PF,X_anchors,Xk_list,Yk_list,plot_ellipse=True)

    #err_estim = [elt1-elt2 for elt1,elt2 in zip(Xk_list,Xk_EKF)]
    #print(err_estim)

    print("Handle mesurement correction in MHE optim")
    
    # # Calculate the errors

    simu.RMSE(Xk_EKF,"EKF")
    simu.RMSEpos(Xk_EKF,"EKF")

    simu.RMSE(Xk_UKF,"UKF")
    simu.RMSEpos(Xk_UKF,"UKF")

    simu.RMSE(Xk_MHE,"MHE")
    simu.RMSEpos(Xk_MHE,"MHE")

    simu.RMSE(Xk_PF,"PF")
    simu.RMSEpos(Xk_PF,"PF")

    #plot_state_estimate(Xk_list,EKF)

    plot_state_estimate(Xk_list,UKF)

    plot_state_estimate(Xk_list,MHE)

    plot_state_estimate(Xk_list,PF)

    plot_norm_error(Xk_list,UKF)

    plot_norm_error(Xk_list,PF)

    print("For Monte-Carlo simu voir ROS 2 / Test / Monte-Carlo_run_RMSE.py ")

    #Cramer-Rao Lower bound

if __name__ == "__main__":
    main()

