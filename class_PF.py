import math 
import numpy as np
import matplotlib.pyplot as plt

import localization_simu_main as loc_simu
from scipy import linalg

from scipy.stats import multivariate_normal

# Centralized Particle Filter
class classPF:
    def __init__(self,  X0_est, Sigma, Qk , Rk, N_particles,
        dim_x=1, dim_z=1, dim_u=1, eval_fx=None,
        eval_hk=None, h_correction = None, state_correction = None
    ):
        """
        Initializes the Particle filter creating the necessary matrices
        """
        self.name = "PF"
        self.X0_est = X0_est  # initial mean state estimate
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

        self.K = []
        self.mean_CT = 0 # computation time

        self.Yk_list = []
        self.Uk_list = []

        self.weights_normalized = [] # array of normalized weights
        self.weights_inter = [] # array of intermediate weights 
        self.Xk_particles = []
        self.N_particles = N_particles

        self.nu = dim_u
        self.n = dim_x
        self.ny = dim_z
        self.nw = self.Qk.shape[0]

    def init(self, Yk_list,Uk_list,Xk_list):
        
        self.Yk_list = Yk_list
        self.Uk_list = Uk_list
        self.Xk_list = Xk_list

        # self.nu = self.Uk_list.shape[0]
        # self.n = self.Xk_list.shape[0]
        # self.ny = self.Yk_list.shape[0]
        # self.nw = self.Qk.shape[0]

        # Initialize particles
        
        self.Xk_particles =  np.random.multivariate_normal(self.X0_est, self.Sigma, size=self.N_particles)
        self.weights_normalized = 1/self.N_particles * np.ones(self.N_particles)

        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Particle Filter visualization")
        ##ax.legend()

        self.ax = ax


    def predict(self,t, Xk, Uk, Te):
        # Predict the next state 

        #process_noise_distribution = multivariate_normal(mean = np.zeros((self.nw,1)), cov = self.Qk)

        w_process_noise_sampled = np.random.multivariate_normal(mean = np.zeros(self.nw), cov = self.Qk, size=self.N_particles)
        #w_process_noise_sampled

        for i in range(0,self.N_particles):
            #self.Xk_particles[i] = self.state_correction(self.Xk_particles[i])
            self.Xk_particles[i] = self.eval_fx(t,self.Xk_particles[i],Uk,w_process_noise_sampled[i],Te,self.n)
            
            #self.Xk_particles[i] = self.state_correction(self.Xk_particles[i])

        #print(t,self.weights_normalized)
        #self.plot_particles(t,[],[])


        return Xk, self.Sigma

    def update(self,t,Xk_pred,X_anchors,Zmes):
        # Update the state prediction with the current measurement

        self.plot_particles(t,X_anchors,Zmes)

        #meas_noise_distribution = multivariate_normal(mean = np.zeros((self.ny,1)), cov = self.Rk)
        v_meas_noise_sampled = np.random.multivariate_normal(mean = np.zeros(self.ny), cov = self.Rk, size=self.N_particles)

        observed_particles = np.zeros((self.N_particles,self.ny))

        self.weights_inter = np.zeros(self.N_particles)
        #new_weights = np.zeros((1,self.N_particles))

        for i in range(0,self.N_particles):
            observed_particles[i] = self.eval_hk(t,self.Xk_particles[i],X_anchors) #+ v_meas_noise_sampled[i]
            #observed_particles[i] = self.h_correction(observed_particles[i])

            self.weights_inter[i] = self.weights_normalized[i] * self.compute_likelihood(Zmes, observed_particles[i])

        self.weights_normalized = self.weights_inter/(self.weights_inter.sum())


        sum_sq_weights = np.sum([weight**2 for weight in self.weights_normalized])
        Neff = 1/sum_sq_weights


        # Mean state estimation with weights
        # Xk = (self.Xk_particles @ self.weights_normalized).sum(axis=0)
        Xk = np.sum(self.Xk_particles * self.weights_normalized[:, np.newaxis], axis=0)
        Xk = self.state_correction(Xk)

        for i in range(0,self.N_particles):
            self.Xk_particles[i] = self.state_correction(self.Xk_particles[i])

        diff_Xk_particles = self.Xk_particles.T - Xk[:, np.newaxis]
        diff_Xk_particles = self.state_correction(diff_Xk_particles)

        #self.Sigma = np.average(diff_Xk_particles @ diff_Xk_particles.T , weights=self.weights_normalized, axis=0)

        self.Sigma = np.zeros((self.n,self.n))  # Initialize covariance matrix
        for i in range(len(self.weights_normalized)):
            self.Sigma += self.weights_normalized[i] * np.outer(diff_Xk_particles[:,i], diff_Xk_particles[:,i])

        # self.Sigma *= 1.0 / (1.0 - self.weights_normalize @ self.weights_normalize.T)
        self.Sigma_list.append(self.Sigma)

        self.plot_particles(t,X_anchors,Zmes)
        #self.visualize_likelihood(t,Zmes,X_anchors,grid_size=200, grid_radius=1.5)

        # Resample if condition is met
        if Neff < (self.N_particles/3):
            
            #print("Resample: to do","Neff =",Neff)
            #self.multinomialResampling()
            self.systematicResampling()
            a = 1
            
        self.plot_particles(t,X_anchors,Zmes)

        return Xk, self.Sigma
    

    def compute_likelihood(self, Zmes, Z_pred):  
        """
        Compute likelihood p(z|sample) for a specific measurement given sample state and landmarks. 
        """
        R_inv = np.linalg.inv(40*self.Rk)
        residual = Zmes - Z_pred
        residual = self.h_correction(residual) # 
        likelihood = np.exp(-0.5 * residual.T @ R_inv @ residual)
        # (1/np.sqrt(2*np.pi)@R_inv)*

        #rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
        #likelihood = multivariate_normal.pdf(Zmes, mean=Z_pred, cov=self.Rk)
        
        # log_likelihood = -0.5 * residual.T @ R_inv @ residual
        return likelihood

    def plot_particles(self,t,X_anchors,Zmes):

        self.ax.clear()
        #plt.clf()
        self.ax.scatter(self.Xk_particles[:,0],self.Xk_particles[:,1],c="blue", s=8000*self.weights_normalized,alpha=0.3, edgecolors='none')
        self.ax.scatter(self.Xk_list[t,0],self.Xk_list[t,1],c="green", s=500, marker = "o", alpha=0.3,label="True pos")
        
        if len(Zmes)>0:

            for i in range(len(X_anchors)):
                pos_anchor = X_anchors[i]
                name = "Meas pos " + str(i)
                self.ax.scatter(pos_anchor[0] - Zmes[2*i]*math.cos(Zmes[2*i+1]), pos_anchor[1] - Zmes[2*i]*math.sin(Zmes[2*i+1]),c="red", s=500, marker = "o", alpha=0.3,label=name)


        self.ax.grid(True)
        self.ax.legend()
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_title("Particle Filter visualization at time "+str(t))
        #self.ax limits # around the true pos
        plt.draw()
        plt.pause(0.0001)

    def multinomialResampling(self):
        """ Bootstrap resampling particles """
        # aka Multinomial resampling

        bootstrap_particles_indexes = np.random.choice(np.arange(self.N_particles),self.N_particles,p=self.weights_normalized)

        self.Xk_particles = self.Xk_particles[bootstrap_particles_indexes]
        #self.weights_normalized = self.weights_normalized[bootstrap_particles_indexes]
        self.weights_normalized = np.ones(self.N_particles)*1/self.N_particles


        return

    # SIR  (Sequential Importance Resampling) 
    


    def systematicResampling(self):
        """Systematic resampling method"""
        """ aka Stochastic universal sampling """
        """
        Loop over cumulative sum once hence particles should keep same order (however some disappear, other are
        replicated). Variance on number of times a particle will be selected lower than with stratified resampling.

        Computational complexity: O(N)

        :param samples: Samples that must be resampled.
        :param N: Number of samples that must be generated.
        :return: Resampled weighted particles.
        """

        #cumulative_weights
        #cValues = []
        #cValues.append(self.weights_normalized.copy())

        #for i in range(self.N_particles-1):
        #    cValues.append(cValues[i] + self.weights_normalized[i+1])

        cValues = self.weights_normalized.copy()
        cValues = np.cumsum(cValues)

        #Starting random point
        startingPoint = np.random.uniform(low=0.0,high=1/self.N_particles)

        #this list stores indices of resamples states
        resampledIndex = []
        for j in range(0,self.N_particles):
            currentPoint = startingPoint + j/self.N_particles
            s=0
            while (currentPoint>cValues[s]):
                s=s+1

            resampledIndex.append(s)

        self.Xk_particles = self.Xk_particles[resampledIndex]
        #self.weights_normalized = self.weights_normalized[bootstrap_particles_indexes]
        self.weights_normalized = np.ones(self.N_particles)*1/self.N_particles

        return

    def visualize_likelihood(self,t,Zmes,X_anchors,grid_size=200, grid_radius=1.0):
        """ Visualize the measurements likelihood """
        # 2D with magnitude colors for high likelihood

        #self.Xk_list[t,0]
        #self.Xk_list[t,1]
        #px_list,py_list = []
        #self.eval_hk(t,self.Xk_particles[i],X_anchors)

        # Extract reference position
        x_ref, y_ref = self.Xk_list[t, 0:2]

        # Create grid
        x_min, x_max = x_ref - grid_radius, x_ref + grid_radius
        y_min, y_max = y_ref - grid_radius, y_ref + grid_radius
        x_range = np.linspace(x_min, x_max, grid_size)
        y_range = np.linspace(y_min, y_max, grid_size)
        XX, YY = np.meshgrid(x_range, y_range)

        # Compute likelihood for each grid point
        likelihood = np.zeros_like(XX)
        for i in range(grid_size):
            for j in range(grid_size):
                X_test = np.array([XX[i, j], YY[i, j]])
                Z_pred = self.eval_hk(t, X_test, X_anchors)
                likelihood[i, j] = self.compute_likelihood(Zmes, Z_pred)

        max_likelihood = likelihood.max()
        (likelihood_max_index_y,likelihood_max_index_x)  = np.where(likelihood == max_likelihood)

        max_x =  XX[0,likelihood_max_index_x[0]] 
        max_y = YY[likelihood_max_index_y[0],0]

        # Plot
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(XX, YY, likelihood, shading='auto', cmap='viridis')
        plt.colorbar(label='Likelihood')
        plt.scatter(x_ref, y_ref, color='red', marker='x', s=200, label=f'State at t={t}')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title(f'Measurement Likelihood at t={t} around ({x_ref:.2f}, {y_ref:.2f})')
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

        plt.scatter(max_x, max_y, color='green', marker='x', s=200, label=f'Max likelihood')

        plt.legend()
        plt.draw()
        plt.pause(0.0001)
        #plt.show()

        return


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
