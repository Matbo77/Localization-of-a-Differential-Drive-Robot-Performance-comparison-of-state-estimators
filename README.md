# Localization-of-a-Differential-Drive-Robot-Performance-comparison-of-state-estimators

 <a href= "https://img.shields.io/badge/github-repo-blue?logo=github"> <img src="https://img.shields.io/badge/github-repo-blue?logo=github" alt="GitHub Badge"/></a>
 ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
 <a href= "https://img.shields.io/badge/CasADi-R"> <img src="https://img.shields.io/badge/CasADi-R" alt="CasADi Badge"/></a>

📝 Description
This project implements and compares four state estimation algorithms for localizing a 2D differential drive robot using range and bearing angle measurements to known landmarks. The goal is to evaluate the accuracy, robustness, and computational efficiency of each estimator in a noisy environment.

Estimators Implemented:

Extended Kalman Filter (EKF): Linearizes the nonlinear system.
Unscented Kalman Filter (UKF): Uses sigma points for better nonlinear handling.
Moving Horizon Estimation (MHE): Optimizes over a sliding window of past measurements.
Particle Filter (PF): Uses weighted samples to approximate the posterior distribution.


📈 Visualize Results


