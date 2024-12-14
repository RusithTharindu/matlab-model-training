% Define users
users = ["U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08", "U09", "U10"];
target_user = "U05";

allTrainData = [];
allTrainLabels = [];
allTestData = [];
allTestLabels = [];

% Loop through each user to prepare data
for u = 1:length(users)
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
    combinedTrainData = [Acc_FD_Feat_Vec, Time_FD_Feat_Vec];
    userData = [userData; combinedTrainData];

    % Combine all data for this user (testing)
    combinedTestData = [Acc_MD_Feat_Vec, Time_MD_Feat_Vec];
    userTestData = [userTestData; combinedTestData];

    if users(u) == target_user
        % Target user data: 70% for training, 30% for testing
        numSamples = size(userData, 1);
        indices = randperm(numSamples);
        numTrain = round(0.7 * numSamples);

        % 70% authentic data for training
        targetTrainData = userData(indices(1:numTrain), :);
        targetTrainLabels = ones(size(targetTrainData, 1), 1);  % Label as 1

        % 30% authentic data for testing
        targetTestData = userData(indices(numTrain+1:end), :);
        targetTestLabels = ones(size(targetTestData, 1), 1);  % Label as 1

        % Append to global data
        allTrainData = [allTrainData; targetTrainData];
        allTrainLabels = [allTrainLabels; targetTrainLabels];

        allTestData = [allTestData; targetTestData];
        allTestLabels = [allTestLabels; targetTestLabels];
    else
        % Imposter data: Use 30% of other users' data for testing
        numSamples = size(userData, 1);
        indices = randperm(numSamples);
        numSelect = round(0.3 * numSamples);

        % 30% imposter data
        imposterData = userData(indices(1:numSelect), :);
        imposterLabels = zeros(size(imposterData, 1), 1);  % Label as 0

        % Append to global data
        allTestData = [allTestData; imposterData];
        allTestLabels = [allTestLabels; imposterLabels];
    end
end

% Define inputs (features) and targets (labels) for training
X = allTrainData;  % Training data
Y = allTrainLabels;  % Training labels

% Create and train the neural network
hiddenLayerSize = 10; % Number of neurons in the hidden layer
net = feedforwardnet(hiddenLayerSize);

% Train the network
[net, tr] = train(net, X', Y');  % Transpose X and Y for training

disp('Training complete!');
view(net);  % Visualize the network architecture

% Test the trained network for each user
for u = 1:length(users)
    % Load test data for the current user
    testData_Acc = load(['dataset/', char(users(u)), '_Acc_FreqD_MDay']);
    Acc_MD_Feat_Vec = testData_Acc.Acc_FD_Feat_Vec;

    testData_Time_FD = load(['dataset/', char(users(u)), '_Acc_TimeD_MDay']);
    Time_MD_Feat_Vec = testData_Time_FD.Acc_TD_Feat_Vec;

    combinedTestData = [Acc_MD_Feat_Vec, Time_MD_Feat_Vec];  % Concatenate features

    % Predict the labels for the current user's test data
    predictedLabels = net(combinedTestData') > 0.5;  % Threshold at 0.5

    % Display results
    if users(u) == target_user
        trueLabel = 1;  % Authentic
    else
        trueLabel = 0;  % Imposter
    end

    accuracy = sum(predictedLabels' == trueLabel) / length(predictedLabels) * 100;
    disp(['User ', char(users(u)), ' Accuracy: ', num2str(accuracy), '%']);
end
