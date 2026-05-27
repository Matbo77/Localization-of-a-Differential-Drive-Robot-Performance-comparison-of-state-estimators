import numpy as np
import math
import matplotlib.pyplot as plt

from casadi import *
from casadi.tools import *
#import casadi as ca

from scipy.stats import chi2


from matplotlib.patches import Ellipse

# def f_dyn_casadi():
#     # États : X = [x, y, theta]
#     X = SX.sym('X', 3)
#     x, y, theta = X[0], X[1], X[2]

#     # Commandes + bruits : U = [v, w], W = [w_v, w_w]
#     U = SX.sym('U', 2)
#     W = SX.sym('W', 2)

#     # Modèle dynamique (avec bruits additifs)
#     dx = (U[0] + W[0]) * cos(theta)
#     dy = (U[0] + W[0]) * sin(theta)
#     dtheta = U[1] + W[1]

#     # Retourne le vecteur d'état dérivé
#     return Function('f_dyn', [X, U, W], [vertcat(dx, dy, dtheta)])

def f_dyn_casadi(t, Xk, Uk, Wk):
    # special f_dyn compatible with CasADi
    v = Uk[0] + Wk[0]
    w = Uk[1] + Wk[1]
    theta = Xk[2]

    # Compute the derivatives with CasADi functions
    dx = v * cos(theta)
    dy = v * sin(theta)
    dtheta = w

    return vertcat(dx, dy, dtheta)

def RK4_casadi(t, Xk, Uk, Wk, Te, n):
    # k1
    k1 = f_dyn_casadi(t, Xk, Uk, Wk)
    Xk_12 = Xk + (Te / 2) * k1

    # k2
    k2 = f_dyn_casadi(t, Xk_12, Uk, Wk)
    Xk_23 = Xk + (Te / 2) * k2

    # k3
    k3 = f_dyn_casadi(t, Xk_23, Uk, Wk)
    Xk_34 = Xk + Te * k3

    # k4
    k4 = f_dyn_casadi(t, Xk_34, Uk, Wk)

    # Final result
    Xk_1 = Xk + (Te / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return Xk_1

def h_meas(t,Xk,X_anchors):
    # Xk = [xk,yk,thetak], state at k
    # X_anchor = [xpk,ypk], anchor position matrix at time k
    # t : time
    # Yk = [range_i, bearing angle_i, ...] concatenation measurements of each anchor
    # bearing angle from robot to anchor

    # (n_anchors,_) = X_anchors.shape    
    # Yk = np.array([ math.dist(Xk[0:2],X_anchors[0,:]), math.atan2(X_anchors[0,1]-Xk[1],X_anchors[0,0]-Xk[0]) ])
    # for i in range(1,n_anchors):
    #     #Yk.extend(np.array([math.dist(Xk[0:2],X_anchors[i,:]), math.atan2(X_anchors[i,1]-Xk[1],X_anchors[i,0]-Xk[0]) ]))
    #     Yk = np.append(Yk,np.array([math.dist(Xk[0:2],X_anchors[i,:]), math.atan2(X_anchors[i,1]-Xk[1],X_anchors[i,0]-Xk[0]) ]))
    Yk = []
    for anchor in X_anchors:
        dx = anchor[0] - Xk[0]
        dy = anchor[1] - Xk[1]
        distance = sqrt(dx**2 + dy**2)
        angle = atan2(dy, dx)
        Yk.append(distance)
        Yk.append(angle)

    return Yk

def h_meas_correction(meas_residual):
    """Normalize the heading angle measurement"""
    #meas_residual[0:2:end] = normalize_angle(meas_residual[0:2:end])
    meas_residual[1:len(meas_residual):2] = [normalize_angle(meas_residual[i]) for i in range(1,len(meas_residual),2)]

    return meas_residual

def h_meas_correction_casadi(t,meas_residual,X_anchors):
    """ Return a corrected residual vector with normalized angles (symbolic, Casadi compatible)"""
    #meas_residual_norm = meas_residual 
    meas_residual_norm = []

    #meas_residual_norm[1:meas_residual.shape[0]:2] = [normalize_angle_casadi(meas_residual[i]) for i in range(1,meas_residual.shape[0],2)]
    for i in range(len(X_anchors)):
        #meas_residual_norm[2*i+1] = normalize_angle_casadi(meas_residual_norm[2*i+1])
        
        #meas_residual_norm.append([meas_residual[2*i],normalize_angle_casadi(meas_residual[2*i+1])])

        # Distance residual (unchanged)
        meas_residual_norm.append(meas_residual[2*i])
        # Angle residual (normalized)
        meas_residual_norm.append(normalize_angle_casadi(meas_residual[2*i + 1]))

    #return meas_residual_norm
    return vertcat(*meas_residual_norm)


        
    # Return a symbolic vector
    
    # print(meas_residual)
    # shooting["X", k]
    #return Function('h_meas_norm', [meas_residual], [vertcat(*meas_residual_norm)], ["Ymes"],["Ymes_norm"])

def h_meas_casadi(t,X,X_anchors):
    Y = []
    for anchor in X_anchors:
        dx = anchor[0] - X[0]
        dy = anchor[1] - X[1]
        distance = sqrt(dx**2 + dy**2)
        angle = atan2(dy, dx)
        Y.append(distance)
        Y.append(angle)
    return Function('h_meas', [X], [vertcat(*Y)], ["X"],["Ymes"])

def normalize_angle(angle):
    """ Return angle between [-pi,pi] """
    return  ((angle + math.pi) % (2*math.pi)) - math.pi # numpy vectorization compatible
    #return math.fmod(angle + math.pi,2*math.pi) - math.pi 

def normalize_angle_casadi(angle):
    #return fmod(angle + pi , 2*pi) - pi    # CasADi compatible but issue with sign
    return if_else(angle >= 0,
        fmod(angle + pi, 2 * pi) - pi,
        -(fmod(-angle + pi, 2 * pi) - pi)) # # CasADi compatible with correction for sign
    #return np.remainder(angle + pi , 2*pi) - pi # not casadi compatible


def eval_Hk(t,Xk,X_anchors):
    """ Linearized measurement matrix / jacobian"""
    # Xk = [xk,yk,thetak], state at k
    # X_anchor = [xpk,ypk], anchor position matrix at time k
    # t : time
    
    n_anchors = X_anchors.shape[0]
    #(n,_) = Xk.shape
    n = len(Xk)

    #Hk = np.array([])
    H_k = np.zeros((2 * n_anchors, n))

    for i in range(n_anchors):
        Xpos = X_anchors[i]
        dx = Xpos[0] - Xk[0] 
        dy = Xpos[1] - Xk[1]
        den = dx**2 + dy**2

        if den ==0:
            H_k[2*i, :] = np.zeros(1,n)
            H_k[2*i + 1, :] = np.zeros(1,n)
            print("Sensor : den==0")
        else:
            sqrt_den = np.sqrt(den)
            # Ligne pour la distance (L1)
            H_k[2*i, :] = [-dx / sqrt_den, -dy / sqrt_den, 0]

            # Ligne pour l'angle (L2)
            H_k[2*i + 1, :] = [dy / den, -dx / den, 0]

    return H_k

def h_meas_lin(t,Xk,X_anchors):
    # linear measurement x, y
    Yk = Xk[0:2]
    #Yk = np.array(Xk[0:2])
    return Yk

def eval_Hk_lin(t,Xk,X_anchors):

    H_k = np.array([[1,0,0],[0,1,0]])

    return H_k  


def h_meas_lin_correction(meas_residual):

    return meas_residual

def plot_traj_meas(estimator,X_anchors,Xk_list,Yk_list,plot_ellipse=False):
    
    n_anchors = X_anchors.shape[0]

    Xk_estim = estimator.Xk_estim.copy() 
    #beware not to copy the reference to estimator.Xk_estim and to modify it

    c_list = ["blue","red","purple","brown","cyan","pink"] 

    fig, ax = plt.subplots()

    #for anchor in X_anchors: 
        #ax.scatter(anchor[0],anchor[1], marker='x', c='red',  label='anchor')

    for i in range(n_anchors): # 
        name = 'anchor ' + str(i)
        ax.scatter(X_anchors[i][0],X_anchors[i][1], marker='x',c=c_list[i],  label=name)
    
    ax.scatter(Xk_list[0,0],Xk_list[0,1], marker='o', c='green', label='X0')
    ax.plot(Xk_list[:,0],Xk_list[:,1], '--.',color="green", label='Traj')

    
    ax.scatter(Xk_estim[0,0],Xk_estim[0,1], marker='o', c='orange', label='X0 estim')
    label_estimator = "Traj estim " + estimator.name
    ax.plot(Xk_estim[:,0],Xk_estim[:,1], '--.', color='orange', label=label_estimator)


    # display measurement

    #for k in range(n_anchors):
    for k in range(0,len(Yk_list)): # len(Yk_list)

        if k>0 and plot_ellipse: # avoid displaying first uncertainty ellipse
            #Pk_estim = 
            drawprobellipse(Xk_estim[k], estimator.Sigma_list[k], 0.95, color='b', linewidth=1, ax=ax)         

        for i in range(n_anchors):
            x_meas = X_anchors[i,0] - Yk_list[k][2*i]*np.cos(Yk_list[k][2*i+1])
            y_meas = X_anchors[i,1] -  Yk_list[k][2*i]*np.sin(Yk_list[k][2*i+1])
            ax.scatter(x_meas,y_meas, marker='o', c=c_list[i]) #, label='Xmeas estim' "blue"

        #x_meas = X_anchors[1,0] - Yk_list[k][2]*np.cos(Yk_list[k][3])
        #y_meas = X_anchors[1,1] -  Yk_list[k][2]*np.sin(Yk_list[k][3])
        #ax.scatter(x_meas,y_meas, marker='o', c="red") #, label='Xmeas estim'

        

    title = 'Trajectory sketch and estimate ' + estimator.name
    ax.set_title(title)
    ax.set(xlabel='Position x [m]', ylabel='Position y [m]')
    ax.legend()
    ax.grid(True)
    plt.ion()
    plt.show()


def drawellipse(x, a, b, color='b', linewidth=2, ax=None):
    """
    Draw an ellipse at position x = [x, y, theta] with semi-axes a and b.
    Theta is the inclination angle of the major axis.

    Parameters
    ----------
    x : array-like, shape (3,)
        Center [x, y] and rotation angle theta (in radians).
    a : float
        Semi-major axis.
    b : float
        Semi-minor axis.
    color : str or array-like, shape (3,)
        Color of the ellipse (e.g., 'r', 'g', [0.5, 0.5, 0.5]).
    linewidth : float, optional
        Width of the ellipse line. Default: 2.

    Returns
    -------
    h : matplotlib.lines.Line2D
        Graphic handle of the ellipse.
    """

    if ax is None:
        ax = plt.gca()  # Use current axes if none provided

    
    # Plot the ellipse
    angle = x[2]
    ellipse = Ellipse((x[0], x[1]), 2*a, 2*b, angle=angle*180/math.pi , alpha = 0.25, facecolor = "orange", edgecolor ="red")
    # , label = "Uncertainties ellipse" 
    ax.add_patch(ellipse)

    return ellipse  # Return the line object


def drawprobellipse(x, C, p_alpha, color='b', linewidth=2, ax=None):
    """
    Draw the elliptic iso-probability contour of a 2D Gaussian distribution.

    Parameters
    ----------
    x : array-like, shape (2,) or (3,)
        Mean [x, y] (and optionally z, ignored for 2D).
    C : array-like, shape (2, 2) or (3, 3)
        Covariance matrix (only the top-left 2x2 block is used for 2D).
    p_alpha : float
        Significance level (e.g., 0.95 for 95% confidence).
    color : str or array-like, shape (3,)
        Color of the ellipse.
    linewidth : float, optional
        Width of the ellipse line. Default: 2.

    Returns
    -------
    h : matplotlib.lines.Line2D
        Graphic handle of the ellipse.
    """

    if ax is None:
        ax = plt.gca()  # Use current axes if none provided


    # Extract the 2x2 covariance submatrix (for 2D)
    if C.shape == (3, 3):
        C = C[:2, :2]
    elif C.shape != (2, 2):
        raise ValueError("Covariance matrix must be 2x2 or 3x3.")

    sxx, syy, sxy = C[0, 0], C[1, 1], C[0, 1]

    # Calculate unscaled semi-axes
    discriminant = (sxx - syy)**2 + 4 * sxy**2

    if discriminant < 0:
        print("Issue cov matrix",C)
    #else:
    a = np.sqrt(0.5 * (sxx + syy + np.sqrt(discriminant)))
    b = np.sqrt(0.5 * (sxx + syy - np.sqrt(discriminant)))

    # Remove imaginary parts (if covariance is negative definite)
    a = np.real(a)
    b = np.real(b)

    # Scale to reflect the specified probability
    scale = np.sqrt(chi2.ppf(p_alpha, 2))
    a *= scale
    b *= scale

    # Determine the inclination angle
    if sxx != syy:
        angle = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    elif sxy == 0:
        angle = 0
    elif sxy > 0:
        angle = np.pi / 4
    else:
        angle = -np.pi / 4

    # Ensure x is 3D (x, y, theta)
    if len(x) == 2:
        x = np.append(x, angle)
    else:
        x[2] = angle

    # Draw the ellipse
    h = drawellipse(x, a, b, color=color, linewidth=linewidth, ax=ax)
    return h