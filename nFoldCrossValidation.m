% Note: Ignoring 'Subject' column because the label of the participant
% should not play a role in the model. Test set has a different set of
% participants than the training set, so the participant labels are irrelevant.

% User inputs
filename_train = "./train_onehot.csv";
filename_test = "./test_onehot.csv";
nAct = 6;   % Number of activity labels in CSV file
lambdas = [0, 1, 5, 10, 50, 100, 500];   % Constants for regularization
nFolds = length(lambdas);   % Number of folds for cross-validation

% Set up training data
X = csvread(filename_train, 1, 0);  % Skip header of CSV file
[nRow, nCol] = size(X);
X = X(randperm(nRow), :);

% Seperate input parameters from outputs (activity one-hots)
Y = X(:, nCol-nAct+1:end);  % activity one-hots
X = X(:, 1:nCol-nAct-1);

% Standardize training set
%   For each col, subtract by col mean and then divide by col std
[X, mu, sigma] = zscore(X);

% Apply cross-validation and find training error
s = floor(nRow/nFolds);   % Number of samples per fold
bestW = zeros(nCol, nAct);
mostMatches = 0;
for i=1:nFolds
  % Validation set
  start_val = (i-1)*s + 1;
  end_val = i*s;
  
  X_val = X(start_val:end_val, :);
  Y_val = Y(start_val:end_val, :);
  
  % Get training set
  indices = [1:start_val-1, end_val+1:(nFolds*s)];
  X_train = X(indices, :);
  Y_train = Y(indices, :);
  
  % Determine weights
  W = (X_train'*X_train)\(X_train'*Y_train - lambdas(i));
  
  % Compare prediction with validation labels
  Y_pred = X_val*W;
  [_, label_pred] = max(Y_pred, [], 2);
  [_, label_val] = max(Y_val, [], 2);
  
  % Determine matches and "best" weights
  nMatches = sum(label_pred == label_val);
  if (nMatches > mostMatches)
    bestW = W;
    mostMatches = nMatches;
  end
  
  % Display values
  disp(["Lambda: ", num2str(lambdas(i))]);
  disp(["Number of correct predictions: ", num2str(nMatches)]);
  disp(["Percentage of correct predictions: ", num2str(nMatches/s * 100.), "\n"]); 
end

% RUN WITH TEST SET
% Read in test set
X_test = csvread(filename_test, 1, 0);    % Skip header of CSV file
[nTestSamples, _] = size(X_test);
Y_test = X_test(:, nCol-nAct+1:end);
X_test = X_test(:, 1:nCol-nAct-1);

% Apply same standardization to test set
X_test = (X_test - mu)./sigma;

% Make test set label predictions
[_, prediction] = max(X_test*bestW, [], 2);
[_, label_test] = max(Y_test, [], 2);
nMatches = sum(prediction == label_test);
disp("Test set");
disp(["Percentage of correct predictions: ", num2str(nMatches/nTestSamples * 100.)]);