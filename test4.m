clear all;
clc


% Define users
users = ["U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08", "U09", "U10"];

for target_user = 1:10


target_user_testing_ratio = 0.8 ;
target_user_traning_ratio = 0.8 ;
imposter_user_testing_ratio = 0.2 ;
imposter_user_traning_ratio = 0.2 ;


allTrainData = [];
allTrainLabels = [];
allTestData = [];
allTestLabels = [];

 %  clearvars -except users target_user; 

% Loop through each user and train the neural network for each user
for u = 1:10
    userData = [];  % Store user-specific data
    userTestData = []; 
    
    % Load and combine training data for the specific user
    % Load Acc_FreqD_FDay data
    trainData_Acc = load(['dataset/', char(users(u)), '_Acc_FreqD_FDay']);
    Acc_FD_Feat_Vec = trainData_Acc.Acc_FD_Feat_Vec;
    
    % Load Time_FreqD_FDay data
    trainData_Time_FD = load(['dataset/', char(users(u)), '_Acc_TimeD_FDay']);
    Time_FD_Feat_Vec = trainData_Time_FD.Acc_TD_Feat_Vec;

    % Load _Acc_FreqD_MDay data
    testData_Acc = load(['dataset/', char(users(u)), '_Acc_FreqD_MDay']);
    Acc_MD_Feat_Vec = testData_Acc.Acc_FD_Feat_Vec;

    % Load Time_FreqD_MDay data
    testData_Time_FD = load(['dataset/', char(users(u)), '_Acc_TimeD_MDay']);
    Time_MD_Feat_Vec = testData_Time_FD.Acc_TD_Feat_Vec;

    % Combine all data for this user (training)
    combinedTrainData = [Acc_FD_Feat_Vec, Time_FD_Feat_Vec];  % Concatenate features along columns
    userData = [userData; combinedTrainData];  % Add to user-specific data

    % Combine all data for this user (testing)
    combinedTestData = [Acc_MD_Feat_Vec, Time_MD_Feat_Vec];  % Concatenate features along columns
    userTestData = [userTestData; combinedTestData];

    if u == target_user
        % Split 70% of the target user's data
        numTargetSamples = size(userData, 1);

        targetIndices = randperm(numTargetSamples); % Randomize indices
        numTargetTrain = round(target_user_traning_ratio * numTargetSamples); % Calculate 70%

        % 70% of target user data as authentic
       
        targetTrainData = userData(targetIndices(1:numTargetTrain), :);
        targetTrainLabels = ones(size(targetTrainData, 1), 1); % Label as 1 (authentic)

        % Append 70% to allData and allLabels
        allTrainData = [allTrainData; targetTrainData];
        allTrainLabels = [allTrainLabels; targetTrainLabels];

        %........................................................................

        % Testing data for the target user
        targetIndicesTesting = randperm(numTargetSamples);
        numTargetTest = round(target_user_testing_ratio * numTargetSamples);
        
        targetTestData = userData(targetIndicesTesting(1:numTargetTest),:);
        targetTestLabels = ones(size(targetTestData, 1), 1);

        allTestData = [allTestData; targetTestData];
        allTestLabels = [allTestLabels; targetTestLabels];
    else
        % Use 30% of the imposter data
        numImposterSamples = size(userData, 1);
        imposterIndices = randperm(numImposterSamples);
        numImposterSelect = round(imposter_user_traning_ratio * numImposterSamples);

        % 30% of imposter user data as imposter
        imposterData = userData(imposterIndices(1:numImposterSelect), :);
        imposterLabels = zeros(size(imposterData, 1), 1);

        allTrainData = [allTrainData; imposterData];
        allTrainLabels = [allTrainLabels; imposterLabels];

        %...................................................................................
        % Testing data for imposters
        imposterIndicesTesting = randperm(numImposterSamples);
        numImposterTest = round(imposter_user_testing_ratio * numImposterSamples);
        imposterTestData = userData(imposterIndicesTesting(1: numImposterTest), :);
        imposterTestLabels = zeros(size(imposterTestData, 1), 1);

        allTestData = [allTestData; imposterTestData];
        allTestLabels = [allTestLabels; imposterTestLabels];
    end
end
    % Define inputs (features) and targets (labels)
    X = allTrainData;
    Y = allTrainLabels;

    % Create a feedforward neural network
    hiddenLayerSize = 10;
    net = feedforwardnet(hiddenLayerSize);

    % Train the network
    [net, tr] = train(net, X', Y');

    % Predict on training data
    trainPredictions = net(X')';
    trainPredictions = trainPredictions >= 0.5;  % Convert predictions to binary (0 or 1)

    % Calculate training accuracy
    trainingAccuracy = sum(trainPredictions == Y) / length(Y) * 100;

    % Predict on testing data
    testPredictions = net(allTestData')';
    testPredictions = testPredictions >= 0.5;  % Convert predictions to binary (0 or 1)

    % Calculate testing accuracy
    testingAccuracy = sum(testPredictions == allTestLabels) / length(allTestLabels) * 100;

    % Display results
    disp(['User ', num2str(target_user), ' Training complete!']);
    disp(['Training Accuracy: ', num2str(trainingAccuracy), '%']);
    disp(['Testing Accuracy: ', num2str(testingAccuracy), '%']);
    disp(' ');
    view(net);  % Visualize the network architecture

end;