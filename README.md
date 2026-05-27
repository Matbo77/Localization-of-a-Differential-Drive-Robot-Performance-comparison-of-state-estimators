# Localization-of-a-Differential-Drive-Robot-Performance-comparison-of-state-estimators

 <a href= "https://img.shields.io/badge/github-repo-blue?logo=github"> <img src="https://img.shields.io/badge/github-repo-blue?logo=github" alt="GitHub Badge"/></a>
 ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
 <a href= "https://img.shields.io/badge/CasADi-orange"> <img src="https://img.shields.io/badge/CasADi-orange" alt="CasADi Badge"/></a>

📝 Description
This project implements and compares four state estimation algorithms for localizing a 2D differential drive robot using range and bearing angle measurements to known landmarks. The goal is to evaluate the accuracy, robustness, and computational efficiency of each estimator in a noisy environment.

📌 Estimators Implemented:
- Extended Kalman Filter (EKF): Linearizes the nonlinear system.
- Unscented Kalman Filter (UKF): Uses sigma points for better nonlinear handling.
- Moving Horizon Estimation (MHE): Optimizes over a sliding window of past measurements.
- Particle Filter (PF): Uses weighted samples to approximate the posterior distribution.


📈 Visualize Results

<img alt="Traj" src="pictures/traj_estimate_meas_UKF.png" width="70%" height="70%"> </img>
<img alt="Traj_zoom" src="pictures/traj_estimate_meas_UKF_zoom.png" width="70%" height="70%"> </img>
<img alt="Estimate" src="pictures/estimate_UKF.png" width="70%" height="70%"> </img>
<img alt="Traj_MHE" src="pictures/traj_estimate_meas_MHE_zoom.png" width="70%" height="70%"> </img>


## 🤝 Contributing

Contributions are welcome!

Future improvements could include:
- Monte Carlo simulation
- Sliding mode observer


---

## 📚 References




