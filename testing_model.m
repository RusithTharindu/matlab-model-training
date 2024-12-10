data = load('dataset/U01_Acc_FreqD_FDay'); % Example for importing data
loadedData = data.Acc_FD_Feat_Vec;
X = loadedData(:, 1:end-1); % Features
Y = loadedData(:, end); % Labels

% disp(Y);

summary(X);
test10 = grpstats(X, Y); % Group-wise statistics

% Standardize the features using z-score normalization for better comparison and model performance.
X = normalize(X, 'zscore'); 

disp(test10);

hiddenLayerSizes = [10, 10]; % Two hidden layers with 10 neurons each
net = feedforwardnet(hiddenLayerSizes);

% Split Data
[trainInd, valInd, testInd] = dividerand(size(X, 1), 0.7, 0.15, 0.15);
X_train = X(trainInd, :); Y_train = Y(trainInd, :);
X_test = X(testInd, :); Y_test = Y(testInd, :);

net.trainParam.epochs = 8000; % Maximum epochs
net.trainParam.goal = 1e-4;  % Error goal
net = train(net, X_train', Y_train');

Y_pred = net(X_test');
performance = perform(net, Y_test', Y_pred);
disp(['Performance: ', num2str(performance)]);


