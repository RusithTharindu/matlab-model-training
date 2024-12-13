% Define users
users = ["U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08", "U09", "U10"];
allData = [];
allLabels = [];

% Loop through each user and train the neural network for each user
for u = 1:length(users)
    target_user = users(u);  % Set the target user
    userData = [];  % Store user-specific data
    
    % Load training data for the specific user
    trainData = load(['dataset/', char(target_user), '_Acc_FreqD_FDay']);
    trainLoadedData = trainData.Acc_FD_Feat_Vec;
    
    

  
end
