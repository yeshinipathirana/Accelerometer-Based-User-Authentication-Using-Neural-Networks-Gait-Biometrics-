Project Overview

This project implements an accelerometer-based user authentication system using gait (walking) behavioral biometrics. Sensor readings from accelerometer and gyroscope are processed into statistical feature vectors and classified using a Feedforward Multi-Layer Perceptron (FFMLP) neural network trained in MATLAB.

Dataset

10 users, 2 walking sessions per user

~12 minutes of natural walking per user

Sampling rate: 31 Hz

Total sensor records: 220,000+

Columns: event_flag, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

Method

Preprocessing

Load all CSVs and extract user_id and session_id

Validate 7-column format and merge into a unified dataset

Windowing

Sliding windows: 2.5 seconds (~78 samples)

Step: 39 samples (50% overlap)

Feature Extraction (90+ features)

Time-domain statistics per axis: mean, std, min, max, median, IQR, skewness, kurtosis, energy, zero-crossing rate

Magnitude features for accelerometer & gyroscope

Model

FFMLP Neural Network (MATLAB Neural Network Toolbox)

Training: Levenberg–Marquardt (trainlm)

Input: ~90 features, Output: 10 user classes (one-hot)

Evaluation

Accuracy, Precision, Recall, F1-score, Confusion Matrix

Security metrics: FAR, FRR, EER

Results

Baseline model: 99.31% test accuracy (8 errors / 1156 samples)

Optimized model: 99.50% test accuracy (6 errors / 1156 samples)

Security performance: Very low FAR/FRR, average EER < 1%

Hyperparameter Optimization

Performed grid search across:

Hidden neurons: 50, 100, 150, 200

Learning rates: 0.001, 0.01, 0.1
Best configuration: 200 neurons, LR=0.001

Technologies Used

MATLAB

Neural Network Toolbox (FFNN / trainlm)

Feature Engineering (Time-series statistics)

Security 

Evaluation (FAR / FRR / EER)
<img width="2489" height="2970" alt="Neural_Network_Architecture_Diagram" src="https://github.com/user-attachments/assets/00b4e3cb-1f8d-476f-b2fa-4cebd6b8dba6" />
<img width="4768" height="1849" alt="User_Variance_Comparison_Chart" src="https://github.com/user-attachments/assets/c6b80fbb-ca18-4f0e-a6dd-d5f600084641" />
<img width="3000" height="2400" alt="confusion_matrix" src="https://github.com/user-attachments/assets/0e0e9961-801b-4157-a62b-d8b59ba0f843" />
<img width="4768" height="3543" alt="Pre_Post_Optimization_Comparison_Chart" src="https://github.com/user-attachments/assets/8c5ca50c-f836-4f34-b8db-79ca0752ef89" />
<img width="3570" height="2370" alt="Hyperparameter_Optimization_Results_Table_Visualization" src="https://github.com/user-attachments/assets/ef9c0caf-d6df-494e-8aef-eb2d93e00ad0" />
<img width="4768" height="1849" alt="User_Variance_Comparison_Chart" src="https://github.com/user-attachments/assets/a2f21af7-0be8-42d0-9e34-ce9941419c0e" />
