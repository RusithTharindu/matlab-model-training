% Initialize global variables
num_users = 10; % Number of users in the dataset
epochs = 100; % Number of epochs for NN training
target_ratio = 0.7; % Ratio for target user data split
imposter_ratio = 0.3; % Ratio for imposter data split

% Define placeholders for EER, FAR, FRR, Accuracy
EER_results = zeros(num_users, 1);
FAR_results = zeros(num_users, 1);
FRR_results = zeros(num_users, 1);
accuracy_results = zeros(num_users, 1);

% Loop through each target user
for target_user = 1:num_users
    % Create empty containers for training and testing datasets
    training_data = [];
    training_labels = [];
    testing_data = [];
    testing_labels = [];
    
    % 2nd loop: Iterate through all users to create data
    for imposter_user = 1:num_users
        % Load Frequency and Time domain features for both FDay and MDay
        FDay_data = load(sprintf('U0%d_Acc_FreqD_FDay.mat', imposter_user));
        TDay_data = load(sprintf('U0%d_Acc_TimeD_FDay.mat', imposter_user));
        FDay_combined = [FDay_data, TDay_data]; % Combine FDay frequency and time domain
        
        MDay_data = []; % Replace with loading logic for MDay
        MDay_combined = [FDay_data, TDay_data]; % Combine MDay frequency and time domain

        % Label data: 1 for target user, 0 for imposters
        if target_user == imposter_user
            target_label = 1;
        else
            target_label = 0;
        end
        
        % Divide into training and testing datasets
        num_samples = size(FDay_combined, 1);
        train_size = floor(num_samples * target_ratio);
        test_size = num_samples - train_size;
        
        % Append FDay data to training and testing datasets
        training_data = [training_data; FDay_combined(1:train_size, :)];
        training_labels = [training_labels; repmat(target_label, train_size, 1)];
        testing_data = [testing_data; FDay_combined(train_size+1:end, :)];
        testing_labels = [testing_labels; repmat(target_label, test_size, 1)];
        
        % Do the same for MDay data
        num_samples = size(MDay_combined, 1);
        train_size = floor(num_samples * target_ratio);
        test_size = num_samples - train_size;
        
        training_data = [training_data; MDay_combined(1:train_size, :)];
        training_labels = [training_labels; repmat(target_label, train_size, 1)];
        testing_data = [testing_data; MDay_combined(train_size+1:end, :)];
        testing_labels = [testing_labels; repmat(target_label, test_size, 1)];
    end
    
    % After the 2nd loop: Train and Test the NN
    % Create and train the neural network
    net = feedforwardnet([10, 5]); % Two hidden layers with 10 and 5 neurons
    net.trainParam.epochs = epochs;
    net = train(net, training_data', training_labels');
    
    % Test the trained NN
    predictions = net(testing_data');
    predictions = round(predictions'); % Threshold at 0.5 for binary classification
    
    % Calculate metrics
    TP = sum((predictions == 1) & (testing_labels == 1));
    TN = sum((predictions == 0) & (testing_labels == 0));
    FP = sum((predictions == 1) & (testing_labels == 0));
    FN = sum((predictions == 0) & (testing_labels == 1));
    
    accuracy = (TP + TN) / length(testing_labels);
    FAR = FP / (FP + TN);
    FRR = FN / (FN + TP);
    EER = (FAR + FRR) / 2;
    
    % Store results
    accuracy_results(target_user) = accuracy;
    FAR_results(target_user) = FAR;
    FRR_results(target_user) = FRR;
    EER_results(target_user) = EER;
    
    fprintf('User %d: Accuracy = %.2f, FAR = %.2f, FRR = %.2f, EER = %.2f\n', ...
        target_user, accuracy, FAR,FRR,EER);
end