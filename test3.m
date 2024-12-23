clear
clc

% Define users
users = ["U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08", "U09", "U10"];
target_user = "U01";

allTrainData = [];
allTrainLabels = [];
allTestData = [];
allTestLabels = [];



% Loop through each user and train the neural network for each user
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

    % .....................................................................

    % Load _Acc_FreqD_MDay data
    testData_Acc = load(['dataset/', char(users(u)), '_Acc_FreqD_MDay']);
    Acc_MD_Feat_Vec = trainData_Acc.Acc_FD_Feat_Vec;

    % Load Time_FreqD_MDay data
    testData_Time_FD = load(['dataset/', char(users(u)), '_Acc_TimeD_MDay']);
    Time_MD_Feat_Vec = testData_Time_FD.Acc_TD_Feat_Vec;

    %......................................................................

    % Combine all data for this user (training)
    combinedTrainData = [Acc_FD_Feat_Vec, Time_FD_Feat_Vec];  % Concatenate features along columns
    userData = [userData; combinedTrainData];  % Add to user-specific data

    % Combine all data for this user (testing)
    combinedTestData = [Acc_MD_Feat_Vec, Time_MD_Feat_Vec];  % Concatenate features along columns
    userTestData = [userTestData; combinedTestData];

% Inside the loop for each user
if users(u) == target_user
    % Split 70% of the target user's data
    numTargetSamples = size(userData, 1);
    targetIndices = randperm(numTargetSamples); % Randomize indices
    numTargetTrain = round(0.7 * numTargetSamples); % Calculate 70%

    % 70% of target user data as authentic
    targetTrainData = userData(targetIndices(1:numTargetTrain), :);
    targetTrainLabels = ones(size(targetTrainData, 1), 1); % Label as 1 (authentic)


    % Append 70% to allData and allLabels
    allTrainData = [allTrainData; targetTrainData];
    allTrainLabels = [allTrainLabels; targetTrainLabels];


    % ..............................Testting ..............................

    numTargetSamplesTest = size(userData, 1);
    targetIndicesTest = randperm(numTargetSamplesTest); % Randomize indices
    numTargetTest = round(0.3 * numTargetSamplesTest); % Calculate 70%

    % 70% of target user data as authentic
    targetTestData = userData(targetIndicesTest(1:numTargetTest), :);
    targetTestLabels = ones(size(targetTestData, 1), 1); % Label as 1 (authentic)


    % Append 70% to allData and allLabels
    allTestData = [allTestData; targetTestData];
    allTestLabels = [allTestLabels; targetTestLabels];

else
    % Use 30% of the imposter data
    numImposterSamples = size(userData, 1);
    imposterIndices = randperm(numImposterSamples); % Randomize indices
    numImposterSelect = round(0.3 * numImposterSamples); % Calculate 30%

    % 30% of imposter user data as imposter
    imposterData = userData(imposterIndices(1:numImposterSelect), :);
    imposterLabels = zeros(size(imposterData, 1), 1); % Label as 0 (imposter)

    % Append 30% to allData and allLabels
    allTrainData = [allTrainData; imposterData];
    allTrainLabels = [allTrainLabels; imposterLabels];

        % ..............................Testting ..............................

    numTargetSamplesTest = size(userData, 1);
    targetIndicesTest = randperm(numTargetSamplesTest); % Randomize indices
    numTargetTest = round(0.3 * numTargetSamplesTest); % Calculate 70%

    % 70% of target user data as authentic
    targetTestData = userData(targetIndicesTest(1:numTargetTest), :);
    targetTestLabels = zeros(size(targetTestData, 1), 1); % Label as 1 (authentic)


    % Append 70% to allData and allLabels
    allTestData = [allTestData; targetTestData];
    allTestLabels = [allTestLabels; targetTestLabels];
end


% Define inputs (features) and targets (labels)
X = allTrainData;  % Training data
Y = allTrainLabels;  % Training labels

% Create a feedforward neural network
hiddenLayerSize = 10; % Number of neurons in the hidden layer
net = feedforwardnet(hiddenLayerSize);

% Train the network
[net, tr] = train(net, X', Y');  % Transpose X and Y for training

% Display training results
disp(['User ',char(users(u)),' Training complete!']);
view(net);  % Visualize the network architecture



end


