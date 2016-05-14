function [L] = mlca_train(X,Y,print_training_time,verify_Y,K)
% This function is the training algorithm of the MLCA method described in:
% Closed-Form Training of Mahalanobis Distance for Supervised Clustering
% (CVPR 2016)
%
% Version 1.0.0 (March 2016)
% copyright by Marc T. Law (UPMC Univ Paris 06)
% 
%
% Input:
%
% X = input matrix (each row is an input vector) 
% Y = assignment_matrix as defined in our CVPR paper:
%        - its elements are 0 or 1
%        - it contains only one nonzero element per row
%        - it contains at least one nonzero element per column
% (* optional *) print_training_time (default false) = Boolean parameter that indicates 
%        whether the function prints the training time of the method or not
% (* optional *) verify_Y (default false) = Boolean parameter that indicates whether 
%        the function verifies if Y is an assignment matrix or not
% (* optional *) K (default inf) = number of desired clusters. You don't need to use
%        this option except if you want K to be smaller than the number of clusters in Y.
%        Otherwise, it is better to use the option K = inf since it returns the
%        result in Theorem 3.2.
%
% Output:
%
% L = linear transformation matrix such that M = LL' where M is the symmetric
%         positive semidefinite matrix of the learned Mahalanobis distance
if nargin < 5
    K = inf;
end
if nargin < 4
    verify_Y = false;
end
if nargin < 3
    print_training_time = false;
end

if verify_Y
    if ~issparse(Y)
        Y = sparse(Y);
    end
    if ((size(Y,1) * size(Y,2)) - nnz(Y) + sum(sum(Y == 1)) ~= size(Y,1) * size(Y,2))
        error('Error: Y should contain only 0 or 1')
    end
    if sum(sum(Y,2) ~= 1)
        error('Error: Y should contain only 1 nonzero element per row')
    end
end

% We do not consider that computing J is part of the training since
% it can be given as input of the algorithm

% Work on sparse assignment matrix
if ~issparse(Y)
    Y = sparse(Y);
end
% Remove empty clusters
sum_Y = ~sum(Y,1);
if sum(sum_Y)
    Y = Y(:,~sum_Y);
end
% Compute sparse scaled assignment matrix J
J = bsxfun(@rdivide,Y,sqrt(sum(Y)));

% Training
if K >= size(J,2)
    start_time = tic;
    L = pinv(X) * J;
    training_time = toc(start_time);
else
    start_time = tic;
    pinvX = pinv(X);
    [U,D,~] = svd(X * (pinvX * J),'econ');
    d = diag(D);
    L = pinvX * bsxfun(@times,U(:,1:K),sqrt(d(1:K))');
    training_time = toc(start_time);
end
% End of training

if print_training_time
    fprintf('MLCA metric learned in %f seconds\n',training_time);
end
end
