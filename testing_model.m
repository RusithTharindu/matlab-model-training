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