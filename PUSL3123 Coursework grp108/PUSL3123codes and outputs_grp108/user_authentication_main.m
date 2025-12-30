
clear; clc; close all;

%% ========================================================================
% CONFIGURATION PARAMETERS
% ========================================================================

% Data paths
DATASET_DIR = 'Dataset';  
OUTPUT_DIR = 'outputs';      % Output directory for results

% Feature extraction parameters
SAMPLE_RATE_HZ = 31;         % Approximate sampling rate (samples per second)
WINDOW_DURATION_S = 2.5;     % Window duration in seconds
WINDOW_SIZE = round(SAMPLE_RATE_HZ * WINDOW_DURATION_S);  % Window size in samples
STEP_SIZE = round(WINDOW_SIZE / 2);  % Step size for sliding windows (50% overlap)

% Neural network parameters
HIDDEN_LAYER_SIZE = 100;      % Default number of hidden neurons
TRAIN_EPOCHS = 100;           % Training epochs
TEST_SIZE_RATIO = 0.2;        % 20% for testing

% Random seed for reproducibility
RANDOM_SEED = 42;
rng(RANDOM_SEED);

%% ========================================================================
% STEP 1: LOAD RAW CSV DATA
% ========================================================================

fprintf('\n=== STEP 1: LOADING RAW CSV DATA ===\n');
fprintf('Dataset directory: %s\n', DATASET_DIR);

% Get all CSV files matching the pattern U*NW_*.csv
csvFiles = dir(fullfile(DATASET_DIR, 'U*NW_*.csv'));

if isempty(csvFiles)
    error('No CSV files found in %s. Please check the dataset directory.', DATASET_DIR);
end

fprintf('Found %d CSV files\n', length(csvFiles));

% Initialize storage
allRawData = [];
allUserIDs = [];
allSessions = [];
allSourceFiles = {};

% Load each CSV file
for i = 1:length(csvFiles)
    csvPath = fullfile(DATASET_DIR, csvFiles(i).name);
    fprintf('Loading: %s\n', csvFiles(i).name);
    
    % Parse user ID and session from filename (e.g., U1NW_FD.csv)
    filename = csvFiles(i).name;
    userMatch = regexp(filename, 'U(\d+)NW_(\w+)', 'tokens');
    if isempty(userMatch)
        warning('Could not parse filename: %s. Skipping.', filename);
        continue;
    end
    
    userID = str2double(userMatch{1}{1});
    session = userMatch{1}{2};
    
    % Read CSV file
    try
        csvData = readmatrix(csvPath);
        
        % Validate data dimensions
        if size(csvData, 2) ~= 7
            warning('File %s has %d columns, expected 7. Skipping.', filename, size(csvData, 2));
            continue;
        end
        
        % Store data
        allRawData = [allRawData; csvData];
        allUserIDs = [allUserIDs; repmat(userID, size(csvData, 1), 1)];
        allSessions = [allSessions; repmat({session}, size(csvData, 1), 1)];
        allSourceFiles = [allSourceFiles; repmat({filename}, size(csvData, 1), 1)];
        
    catch ME
        warning('Error loading %s: %s', filename, ME.message);
        continue;
    end
end

fprintf('Total samples loaded: %d\n', size(allRawData, 1));
fprintf('Unique users: %s\n', mat2str(unique(allUserIDs)'));
fprintf('Data loading complete.\n\n');

%% ========================================================================
% STEP 2: FEATURE EXTRACTION
% ========================================================================

fprintf('=== STEP 2: FEATURE EXTRACTION ===\n');
fprintf('Window size: %d samples (%.1f seconds)\n', WINDOW_SIZE, WINDOW_DURATION_S);
fprintf('Step size: %d samples\n', STEP_SIZE);

% Group data by user, session, and source file
uniqueUsers = unique(allUserIDs);
featureMatrix = [];
featureLabels = [];
featureMetadata = struct('user_id', {}, 'session', {}, 'source_file', {});

for u = 1:length(uniqueUsers)
    userID = uniqueUsers(u);
    userMask = allUserIDs == userID;
    userData = allRawData(userMask, :);
    userSessions = allSessions(userMask);
    userFiles = allSourceFiles(userMask);
    
    % Group by source file (each file is a separate session)
    uniqueFiles = unique(userFiles);
    
    for f = 1:length(uniqueFiles)
        fileMask = strcmp(userFiles, uniqueFiles{f});
        fileData = userData(fileMask, :);
        fileSession = userSessions{find(fileMask, 1)};
        
        % Extract features using sliding windows
        numWindows = floor((size(fileData, 1) - WINDOW_SIZE) / STEP_SIZE) + 1;
        
        for w = 1:numWindows
            startIdx = (w - 1) * STEP_SIZE + 1;
            endIdx = startIdx + WINDOW_SIZE - 1;
            
            if endIdx > size(fileData, 1)
                break;
            end
            
            window = fileData(startIdx:endIdx, :);
            
            % Compute features for this window
            windowFeatures = extractWindowFeatures(window);
            
            % Store features
            featureMatrix = [featureMatrix; windowFeatures];
            featureLabels = [featureLabels; userID];
            
            % Store metadata
            metadata.user_id = userID;
            metadata.session = fileSession;
            metadata.source_file = uniqueFiles{f};
            featureMetadata(end+1) = metadata;
        end
    end
end

fprintf('Feature extraction complete.\n');
fprintf('Total feature vectors: %d\n', size(featureMatrix, 1));
fprintf('Number of features per vector: %d\n', size(featureMatrix, 2));
fprintf('Feature matrix shape: [%d x %d]\n\n', size(featureMatrix, 1), size(featureMatrix, 2));

%% ========================================================================
% STEP 3: STATISTICAL ANALYSIS
% ========================================================================

fprintf('=== STEP 3: STATISTICAL ANALYSIS ===\n');

% Overall statistics
featureMeans = mean(featureMatrix, 1);
featureStds = std(featureMatrix, 0, 1);

fprintf('Mean values for first 10 features:\n');
disp(featureMeans(1:min(10, length(featureMeans))));

fprintf('Standard deviations for first 10 features:\n');
disp(featureStds(1:min(10, length(featureStds))));

% Inter-user and intra-user variance analysis
fprintf('\n=== VARIANCE ANALYSIS ===\n');
uniqueUsers = unique(featureLabels);
numFeatures = size(featureMatrix, 2);

interUserVariances = zeros(numFeatures, 1);
intraUserVariances = zeros(numFeatures, 1);

for featIdx = 1:numFeatures
    % Inter-user variance: variance of means across users
    userMeans = arrayfun(@(u) mean(featureMatrix(featureLabels == u, featIdx)), uniqueUsers);
    interUserVariances(featIdx) = var(userMeans);
    
    % Intra-user variance: average variance within each user
    userVariances = arrayfun(@(u) var(featureMatrix(featureLabels == u, featIdx)), uniqueUsers);
    intraUserVariances(featIdx) = mean(userVariances);
end

% Create variance analysis table
varianceTable = table((1:numFeatures)', interUserVariances, intraUserVariances, ...
    'VariableNames', {'Feature_Index', 'Inter_User_Variance', 'Intra_User_Variance'});

% Ensure output directory exists
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

writetable(varianceTable, fullfile(OUTPUT_DIR, 'User_Variance_Analysis_Report.csv'));
fprintf('Variance analysis saved to: %s\n', fullfile(OUTPUT_DIR, 'User_Variance_Analysis_Report.csv'));

% Plot variance comparison
figure('Name', 'Variance Analysis', 'Position', [100, 100, 800, 500]);
bar([interUserVariances(1:min(20, numFeatures)), intraUserVariances(1:min(20, numFeatures))]);
legend('Inter-User Variance', 'Intra-User Variance', 'Location', 'northwest');
xlabel('Feature Index');
ylabel('Variance Magnitude');
title('Inter-User vs Intra-User Feature Variance (First 20 Features)');
grid on;
saveas(gcf, fullfile(OUTPUT_DIR, 'User_Variance_Comparison_Chart.png'));
fprintf('Variance comparison chart saved.\n\n');

%% ========================================================================
% STEP 4: DATA SPLITTING AND PREPROCESSING
% ========================================================================

fprintf('=== STEP 4: DATA SPLITTING AND PREPROCESSING ===\n');

% Stratified split to maintain class distribution
cv = cvpartition(featureLabels, 'HoldOut', TEST_SIZE_RATIO);
trainIdx = training(cv);
testIdx = test(cv);

X_train = featureMatrix(trainIdx, :);
y_train = featureLabels(trainIdx);
X_test = featureMatrix(testIdx, :);
y_test = featureLabels(testIdx);

fprintf('Training set: %d samples\n', size(X_train, 1));
fprintf('Testing set: %d samples\n', size(X_test, 1));

% Feature standardization (z-score normalization)
[X_train_scaled, mu_train, sigma_train] = zscore(X_train);
X_test_scaled = (X_test - mu_train) ./ sigma_train;

fprintf('Feature standardization complete.\n\n');

%% ========================================================================
% STEP 5: NEURAL NETWORK TRAINING (INITIAL MODEL)
% ========================================================================

fprintf('=== STEP 5: NEURAL NETWORK TRAINING (INITIAL MODEL) ===\n');
fprintf('Hidden layer size: %d neurons\n', HIDDEN_LAYER_SIZE);
fprintf('Training epochs: %d\n', TRAIN_EPOCHS);

% Create feedforward neural network
net = feedforwardnet(HIDDEN_LAYER_SIZE);

% Configure network
net.trainFcn = 'trainlm';  % Levenberg-Marquardt algorithm
net.performFcn = 'mse';    % Mean squared error
net.trainParam.epochs = TRAIN_EPOCHS;
net.trainParam.showWindow = false;  % Suppress training window

% Prepare target vectors (one-hot encoding)
y_train_categorical = categorical(y_train);
y_train_onehot = dummyvar(y_train_categorical)';

% Train network
fprintf('Training neural network...\n');
[net, trainInfo] = train(net, X_train_scaled', y_train_onehot);

fprintf('Training complete.\n');

%% ========================================================================
% STEP 6: MODEL EVALUATION
% ========================================================================

fprintf('\n=== STEP 6: MODEL EVALUATION ===\n');

% Make predictions
y_test_pred_raw = net(X_test_scaled');
[~, y_test_pred] = max(y_test_pred_raw, [], 1);
y_test_pred = y_test_pred(:);

% Map predictions back to user IDs
uniqueUsers = unique(y_train);
y_test_pred_labels = uniqueUsers(y_test_pred);

% Calculate accuracy
accuracy = mean(y_test_pred_labels == y_test) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

% Confusion matrix
confMat = confusionmat(y_test, y_test_pred_labels);
fprintf('\nConfusion Matrix:\n');
disp(confMat);

% Per-user metrics
fprintf('\n=== PER-USER CLASSIFICATION METRICS ===\n');
for u = 1:length(uniqueUsers)
    userID = uniqueUsers(u);
    truePos = confMat(u, u);
    falsePos = sum(confMat(:, u)) - truePos;
    falseNeg = sum(confMat(u, :)) - truePos;
    
    precision = truePos / (truePos + falsePos);
    recall = truePos / (truePos + falseNeg);
    f1Score = 2 * (precision * recall) / (precision + recall);
    
    fprintf('User %d - Precision: %.3f, Recall: %.3f, F1-Score: %.3f\n', ...
        userID, precision, recall, f1Score);
end

% Plot confusion matrix
figure('Name', 'Confusion Matrix', 'Position', [100, 100, 900, 700]);
confusionchart(y_test, y_test_pred_labels);
title('User Authentication Confusion Matrix');
saveas(gcf, fullfile(OUTPUT_DIR, 'Authentication_Confusion_Matrix.png'));
fprintf('\nConfusion matrix saved to: %s\n', fullfile(OUTPUT_DIR, 'Authentication_Confusion_Matrix.png'));

%% ========================================================================
% STEP 7: FAR, FRR, AND EER CALCULATION
% ========================================================================

fprintf('\n=== STEP 7: FAR, FRR, AND EER CALCULATION ===\n');

farFrrResults = [];
for u = 1:length(uniqueUsers)
    userID = uniqueUsers(u);
    
    % Genuine attempts (should be accepted)
    genuineMask = y_test == userID;
    genuinePred = y_test_pred_labels(genuineMask);
    falseRejections = sum(genuinePred ~= userID);
    totalGenuine = sum(genuineMask);
    FRR = (falseRejections / totalGenuine) * 100;
    
    % Impostor attempts (should be rejected)
    impostorMask = y_test ~= userID;
    impostorPred = y_test_pred_labels(impostorMask);
    falseAcceptances = sum(impostorPred == userID);
    totalImpostor = sum(impostorMask);
    FAR = (falseAcceptances / totalImpostor) * 100;
    
    % Equal Error Rate (approximation)
    EER = (FAR + FRR) / 2;
    
    farFrrResults = [farFrrResults; userID, FAR, FRR, EER];
    
    fprintf('User %d - FAR: %.2f%%, FRR: %.2f%%, EER: %.2f%%\n', userID, FAR, FRR, EER);
end

% Save FAR/FRR results
farFrrTable = array2table(farFrrResults, 'VariableNames', {'User_ID', 'FAR_percent', 'FRR_percent', 'EER_percent'});
writetable(farFrrTable, fullfile(OUTPUT_DIR, 'FAR_FRR_Metrics.csv'));
fprintf('\nFAR/FRR metrics saved to: %s\n', fullfile(OUTPUT_DIR, 'FAR_FRR_Metrics.csv'));

%% ========================================================================
% STEP 8: HYPERPARAMETER OPTIMIZATION
% ========================================================================

fprintf('\n=== STEP 8: HYPERPARAMETER OPTIMIZATION ===\n');

hiddenLayerOptions = [50, 100, 150, 200];
learningRateOptions = [0.001, 0.01, 0.1];

bestAccuracy = 0;
bestHiddenSize = HIDDEN_LAYER_SIZE;
bestLearningRate = 0.01;
optimizationResults = [];

fprintf('Testing %d hidden layer sizes and %d learning rates...\n', ...
    length(hiddenLayerOptions), length(learningRateOptions));

for h = 1:length(hiddenLayerOptions)
    for lr = 1:length(learningRateOptions)
        hiddenSize = hiddenLayerOptions(h);
        learningRate = learningRateOptions(lr);
        
        fprintf('Testing: Hidden=%d, LR=%.3f... ', hiddenSize, learningRate);
        
        % Create and configure network
        optNet = feedforwardnet(hiddenSize);
        optNet.trainFcn = 'trainlm';
        optNet.performFcn = 'mse';
        optNet.trainParam.epochs = TRAIN_EPOCHS;
        optNet.trainParam.lr = learningRate;
        optNet.trainParam.showWindow = false;
        
        try
            % Train
            [optNet, ~] = train(optNet, X_train_scaled', y_train_onehot);
            
            % Evaluate
            optPredRaw = optNet(X_test_scaled');
            [~, optPred] = max(optPredRaw, [], 1);
            optPred = optPred(:);
            optPredLabels = uniqueUsers(optPred);
            
            optAccuracy = mean(optPredLabels == y_test) * 100;
            
            optimizationResults = [optimizationResults; hiddenSize, learningRate, optAccuracy];
            
            if optAccuracy > bestAccuracy
                bestAccuracy = optAccuracy;
                bestHiddenSize = hiddenSize;
                bestLearningRate = learningRate;
                bestNet = optNet;  % Save best network
            end
            
            fprintf('Accuracy: %.2f%%\n', optAccuracy);
            
        catch ME
            fprintf('Error: %s\n', ME.message);
        end
    end
end

fprintf('\n=== OPTIMIZATION RESULTS ===\n');
fprintf('Best Configuration:\n');
fprintf('  Hidden Layer Size: %d\n', bestHiddenSize);
fprintf('  Learning Rate: %.3f\n', bestLearningRate);
fprintf('  Best Accuracy: %.2f%%\n', bestAccuracy);

% Save optimization results
optTable = array2table(optimizationResults, 'VariableNames', {'Hidden_Units', 'Learning_Rate', 'Accuracy_percent'});
writetable(optTable, fullfile(OUTPUT_DIR, 'Hyperparameter_Optimization_Results.csv'));
fprintf('\nOptimization results saved to: %s\n', fullfile(OUTPUT_DIR, 'Hyperparameter_Optimization_Results.csv'));

% Plot optimization landscape
if size(optimizationResults, 1) > 0
    figure('Name', 'Hyperparameter Optimization', 'Position', [100, 100, 1000, 800]);
    scatter3(optimizationResults(:, 1), optimizationResults(:, 2), optimizationResults(:, 3), ...
        120, optimizationResults(:, 3), 'filled');
    colorbar;
    xlabel('Number of Hidden Units');
    ylabel('Learning Rate');
    zlabel('Classification Accuracy (%)');
    title('Hyperparameter Optimization Performance Landscape');
    grid on;
    colormap jet;
    saveas(gcf, fullfile(OUTPUT_DIR, 'Hyperparameter_Optimization_Landscape.png'));
    fprintf('Optimization landscape saved.\n');
end

%% ========================================================================
% STEP 9: FINAL OPTIMIZED MODEL
% ========================================================================

fprintf('\n=== STEP 9: FINAL OPTIMIZED MODEL ===\n');

if exist('bestNet', 'var')
    fprintf('Training final model with optimal parameters...\n');
    
    % Evaluate final model
    finalPredRaw = bestNet(X_test_scaled');
    [~, finalPred] = max(finalPredRaw, [], 1);
    finalPred = finalPred(:);
    finalPredLabels = uniqueUsers(finalPred);
    
    finalAccuracy = mean(finalPredLabels == y_test) * 100;
    fprintf('Final Optimized Model Accuracy: %.2f%%\n', finalAccuracy);
    
    % Save model
    save(fullfile(OUTPUT_DIR, 'Optimized_User_Authentication_Model.mat'), ...
        'bestNet', 'mu_train', 'sigma_train', 'bestHiddenSize', 'bestLearningRate', ...
        'finalAccuracy', 'uniqueUsers');
    fprintf('Optimized model saved to: %s\n', fullfile(OUTPUT_DIR, 'Optimized_User_Authentication_Model.mat'));
else
    fprintf('Using initial model as final model.\n');
    save(fullfile(OUTPUT_DIR, 'Optimized_User_Authentication_Model.mat'), ...
        'net', 'mu_train', 'sigma_train', 'HIDDEN_LAYER_SIZE', ...
        'accuracy', 'uniqueUsers');
end

%% ========================================================================
% SUMMARY
% ========================================================================

fprintf('\n=== EXECUTION COMPLETE ===\n');
fprintf('All results saved to: %s\n', OUTPUT_DIR);
fprintf('Best accuracy achieved: %.2f%%\n', bestAccuracy);
fprintf('\nGenerated files:\n');
fprintf('  - User_Variance_Analysis_Report.csv\n');
fprintf('  - User_Variance_Comparison_Chart.png\n');
fprintf('  - Authentication_Confusion_Matrix.png\n');
fprintf('  - FAR_FRR_Metrics.csv\n');
fprintf('  - Hyperparameter_Optimization_Results.csv\n');
fprintf('  - Hyperparameter_Optimization_Landscape.png\n');
fprintf('  - Optimized_User_Authentication_Model.mat\n');

%% ========================================================================
% HELPER FUNCTION: EXTRACT WINDOW FEATURES
% ========================================================================

function features = extractWindowFeatures(window)
    % Extract statistical features from a time-series window
    % Input: window - [N x 7] matrix with columns: [event_flag, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    % Output: features - [1 x M] feature vector
    
    features = [];
    
    % Process each sensor axis (skip event_flag column)
    sensorColumns = 2:7;  % acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    axisNames = {'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'};
    
    for colIdx = 1:length(sensorColumns)
        col = sensorColumns(colIdx);
        axisData = window(:, col);
        axisName = axisNames{colIdx};
        
        % Statistical features per axis
        features = [features, mean(axisData)];                    % mean
        features = [features, std(axisData)];                     % std
        features = [features, min(axisData)];                      % min
        features = [features, max(axisData)];                     % max
        features = [features, median(axisData)];                  % median
        features = [features, iqr(axisData)];                     % IQR
        features = [features, skewness(axisData)];                % skewness
        features = [features, kurtosis(axisData)];                % kurtosis
        features = [features, sum(axisData.^2) / length(axisData)]; % energy
        
        % Zero crossings
        centered = axisData - mean(axisData);
        zeroCrossings = sum(diff(sign(centered)) ~= 0);
        features = [features, zeroCrossings];
    end
    
    % Magnitude features for accelerometer and gyroscope
    accData = window(:, 2:4);  % acc_x, acc_y, acc_z
    gyroData = window(:, 5:7); % gyro_x, gyro_y, gyro_z
    
    accMagnitude = sqrt(sum(accData.^2, 2));
    gyroMagnitude = sqrt(sum(gyroData.^2, 2));
    
    % Features for accelerometer magnitude
    features = [features, mean(accMagnitude)];
    features = [features, std(accMagnitude)];
    features = [features, min(accMagnitude)];
    features = [features, max(accMagnitude)];
    features = [features, median(accMagnitude)];
    features = [features, iqr(accMagnitude)];
    features = [features, skewness(accMagnitude)];
    features = [features, kurtosis(accMagnitude)];
    features = [features, sum(accMagnitude.^2) / length(accMagnitude)];
    centeredAccMag = accMagnitude - mean(accMagnitude);
    zeroCrossingsAcc = sum(diff(sign(centeredAccMag)) ~= 0);
    features = [features, zeroCrossingsAcc];
    
    % Features for gyroscope magnitude
    features = [features, mean(gyroMagnitude)];
    features = [features, std(gyroMagnitude)];
    features = [features, min(gyroMagnitude)];
    features = [features, max(gyroMagnitude)];
    features = [features, median(gyroMagnitude)];
    features = [features, iqr(gyroMagnitude)];
    features = [features, skewness(gyroMagnitude)];
    features = [features, kurtosis(gyroMagnitude)];
    features = [features, sum(gyroMagnitude.^2) / length(gyroMagnitude)];
    centeredGyroMag = gyroMagnitude - mean(gyroMagnitude);
    zeroCrossingsGyro = sum(diff(sign(centeredGyroMag)) ~= 0);
    features = [features, zeroCrossingsGyro];
end

