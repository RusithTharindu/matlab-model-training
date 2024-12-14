clear all;
clc;

% Define users
users = ["U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08", "U09", "U10"];

% Loop through each target user
for target_user = 1:10

    % Ratios for splitting data
    target_user_training_ratio = 0.8;
    target_user_testing_ratio = 0.2;
    imposter_training_ratio = 0.2;
    imposter_testing_ratio = 0.2;

    allTrainData = [];
    allTrainLabels = [];
    allTestData = [];
    allTestLabels = [];

    % Collect data for all users
    for u = 1:10
        % Load and combine feature vectors
        trainData_Acc = load(['dataset/', char(users(u)), '_Acc_FreqD_FDay']);
        Acc_FD_Feat_Vec = trainData_Acc.Acc_FD_Feat_Vec;

        trainData_Time_FD = load(['dataset/', char(users(u)), '_Acc_TimeD_FDay']);
        Time_FD_Feat_Vec = trainData_Time_FD.Acc_TD_Feat_Vec;

        testData_Acc = load(['dataset/', char(users(u)), '_Acc_FreqD_MDay']);
        Acc_MD_Feat_Vec = testData_Acc.Acc_FD_Feat_Vec;

        testData_Time_FD = load(['dataset/', char(users(u)), '_Acc_TimeD_MDay']);
        Time_MD_Feat_Vec = testData_Time_FD.Acc_TD_Feat_Vec;

        % Combine features
        combinedTrainData = [Acc_FD_Feat_Vec, Time_FD_Feat_Vec];
        combinedTestData = [Acc_MD_Feat_Vec, Time_MD_Feat_Vec];

        % Normalize training data (compute mean and std on training data only)
        if u == target_user
            meanTrain = mean(combinedTrainData, 1);
            stdTrain = std(combinedTrainData, 0, 1);
        end

        combinedTrainData = (combinedTrainData - meanTrain) ./ stdTrain;
        combinedTestData = (combinedTestData - meanTrain) ./ stdTrain;

        if u == target_user
            % Target user: Split into training and testing
            numSamples = size(combinedTrainData, 1);
            indices = randperm(numSamples);
            numTrain = round(target_user_training_ratio * numSamples);

            targetTrainData = combinedTrainData(indices(1:numTrain), :);
            targetTrainLabels = ones(size(targetTrainData, 1), 1);

            targetTestData = combinedTrainData(indices(numTrain+1:end), :);
            targetTestLabels = ones(size(targetTestData, 1), 1);

            allTrainData = [allTrainData; targetTrainData];
            allTrainLabels = [allTrainLabels; targetTrainLabels];

            allTestData = [allTestData; targetTestData];
            allTestLabels = [allTestLabels; targetTestLabels];
        else
            % Imposter user: Add as imposter data
            numSamples = size(combinedTrainData, 1);
            indices = randperm(numSamples);
            numTrain = round(imposter_training_ratio * numSamples);
            numTest = round(imposter_testing_ratio * numSamples);

            imposterTrainData = combinedTrainData(indices(1:numTrain), :);
            imposterTrainLabels = zeros(size(imposterTrainData, 1), 1);

            imposterTestData = combinedTrainData(indices(numTrain+1:numTrain+numTest), :);
            imposterTestLabels = zeros(size(imposterTestData, 1), 1);

            allTrainData = [allTrainData; imposterTrainData];
            allTrainLabels = [allTrainLabels; imposterTrainLabels];

            allTestData = [allTestData; imposterTestData];
            allTestLabels = [allTestLabels; imposterTestLabels];
        end
    end

    % Define inputs (features) and targets (labels)
    X = allTrainData;
    Y = allTrainLabels;

    % Create a feedforward neural network
    hiddenLayerSize = 10; % Increased number of neurons for better learning
    net = feedforwardnet(hiddenLayerSize);

    % Train the network
    [net, tr] = train(net, X', Y');

    % Predict on training data
    trainPredictions = net(X')';
    trainPredictions = trainPredictions >= 0.5;

    % Calculate training accuracy
    trainingAccuracy = sum(trainPredictions == Y) / length(Y) * 100;

    % Predict on testing data
    testPredictions = net(allTestData')';
    testPredictions = testPredictions >= 0.5;

    % Calculate testing accuracy
    testingAccuracy = sum(testPredictions == allTestLabels) / length(allTestLabels) * 100;

    % Display results
    disp(['User ', num2str(target_user), ' Training complete!']);
    disp(['Training Accuracy: ', num2str(trainingAccuracy), '%']);
    disp(['Testing Accuracy: ', num2str(testingAccuracy), '%']);
    disp(' ');
    view(net);
end
