data = load('U01_Acc_FreqD_FDay'); % Example for importing data
loadedData = data.Acc_FD_Feat_Vec;
X = loadedData(:, 1:end-1); % Features
Y = loadedData(:, end); % Labels

% disp(Y);

summary(X);
test10 = grpstats(X, Y); % Group-wise statistics

X = normalize(X, 'zscore'); %Standardize the features using z-score normalization for better comparison and model performance.

disp(test10);