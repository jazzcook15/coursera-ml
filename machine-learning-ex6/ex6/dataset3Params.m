function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

try_vec = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
%try_vec = [0.01, 0.1, 1.0, 10.0];
C_vec = try_vec;
sig_vec = try_vec;

err_mat = zeros(length(C_vec), length(sig_vec));

for i=1:length(C_vec)
  C = C_vec(i);
  for j=1:length(sig_vec)
    sigma = sig_vec(j);
    % train a model with these test C and sigma values
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    % evaulate the model's performance
    pred = svmPredict(model, Xval);
    err_mat(i, j) = mean(double(pred ~= yval));
  end
end

% determine row in each column has the minimum error for that column
[min_r,idx_r] = min(err_mat);
% determine which column of the minimum rows has the minimum error
[~,idx_c] = min(min_r);
% select C from the row that minimized the error
C = C_vec(idx_r(idx_c));
% select sigma as the column that minimized the error
sigma = sig_vec(idx_c);

% =========================================================================

end
