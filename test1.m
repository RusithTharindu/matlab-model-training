% Load data for all 10 users
users = ["U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08", "U09", "U10"];
target_user = "U01"; % Set the target user
allData = [];
allLabels = [];

% Combine datasets and assign labels
for i = 1:length(users)
    data = load(['dataset/', char(users(i)), '_Acc_FreqD_FDay']); % Load dataset
    loadedData = data.Acc_FD_Feat_Vec;
    if users(i) == target_user
        labels = ones(size(loadedData, 1), 1); % Label as 1 (authentic)
    else
        labels = zeros(size(loadedData, 1), 1); % Label as 0 (imposter)
    end
    allData = [allData; loadedData]; % Combine data
    allLabels = [allLabels; labels]; % Combine labels
end

% Split features and labels
X = allData(:, 1:end-1); % Features
Y = allLabels;           % Labels

% Normalize features
X = normalize(X, 'zscore');

% Split data into training and testing sets
[trainInd, testInd] = dividerand(size(X, 1), 0.7, 0.3);

X_train = X(trainInd, :); Y_train = Y(trainInd);
X_test = X(testInd, :); Y_test = Y(testInd);

% Define and train a feedforward neural network
hiddenLayerSizes = [10, 10]; % Two hidden layers with 10 neurons each
net = feedforwardnet(hiddenLayerSizes);
net.trainParam.epochs = 1000; % Max epochs
net.trainParam.goal = 1e-4;   % Error goal
net = train(net, X_train', Y_train');

% Test the model
Y_pred = net(X_test');
Y_pred_class = round(Y_pred); % Convert to binary classes (0 or 1)

% Evaluate performance
accuracy = sum(Y_pred_class' == Y_test) / length(Y_test) * 100;
disp(['Overall Accuracy: ', num2str(accuracy), '%']);

% Confusion Matrix
figure; % Display the confusion chart
confusionchart(Y_test, Y_pred_class');

% Load test user data to check if they are imposter or target
test_user = "U04"; % Set the test user for evaluation
testData = load(['dataset/', char(test_user), '_Acc_FreqD_MDay']);
testFeatures = testData.Acc_FD_Feat_Vec(:, 1:end-1); % Features
testFeatures = normalize(testFeatures, 'zscore');    % Normalize test data

% Predict for the test user
testPredictions = net(testFeatures');  % Get prediction probabilities
testPredictedLabels = round(testPredictions'); % Convert to binary classes (0 or 1)

% Imposter Detection
threshold = 0.9; % Threshold for classification
imposterFlag = mean(testPredictions) < threshold;

% Optional: Evaluate test user accuracy (if true labels are available)
if exist('testData.Acc_FD_Feat_Vec', 'var')
    testLabels = testData.Acc_FD_Feat_Vec(:, end); % Actual labels
    testAccuracy = sum(testPredictedLabels' == testLabels) / length(testLabels) * 100;
    disp(['Test User Accuracy: ', num2str(testAccuracy), '%']);
end

% Calculate FAR, FRR, and EER
thresholds = 0:0.01:1; % Threshold range
FAR = zeros(size(thresholds));
FRR = zeros(size(thresholds));

for i = 1:length(thresholds)
    threshold = thresholds(i);
    Y_pred_class_threshold = Y_pred > threshold; % Classify based on threshold
    
    % Calculate False Positives (FP), False Negatives (FN), etc.
    FP = sum(Y_pred_class_threshold' == 1 & Y_test == 0);
    FN = sum(Y_pred_class_threshold' == 0 & Y_test == 1);
    TN = sum(Y_pred_class_threshold' == 0 & Y_test == 0);
    TP = sum(Y_pred_class_threshold' == 1 & Y_test == 1);
    
    % Calculate FAR and FRR
    FAR(i) = FP / (FP + TN); % Imposter classified as target
    FRR(i) = FN / (FN + TP); % Target classified as imposter
end

% Find EER
[~, eerIndex] = min(abs(FAR - FRR)); % Find index where FAR ≈ FRR
EER = (FAR(eerIndex) + FRR(eerIndex)) / 2; % Compute EER
thresholdEER = thresholds(eerIndex); % Corresponding threshold

disp(['EER: ', num2str(EER)]);
disp(['Threshold at EER: ', num2str(thresholdEER)]);

% Plot FAR and FRR
figure;
plot(thresholds, FAR, 'r', 'DisplayName', 'FAR');
hold on;
plot(thresholds, FRR, 'b', 'DisplayName', 'FRR');
plot(thresholdEER, EER, 'ko', 'MarkerSize', 8, 'DisplayName', 'EER');
xlabel('Threshold');
ylabel('Error Rate');
legend show;
title('FAR and FRR with EER');
grid on;
