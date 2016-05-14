function [predicted_clusters, frobenius_loss, rand_loss, predicted_clusters_svd, frobenius_loss_svd, rand_loss_svd] = clustering_test(X_test,Y_test,L, apply_SVD,K,nb_reinitializations)
% This function is the test algorithm of the MLCA method described in:
% Closed-Form Training of Mahalanobis Distance for Supervised Clustering
% (CVPR 2016)
%
% Version 1.0.0 (March 2016)
% copyright by Marc T. Law (UPMC Univ Paris 06)
%
%
% Input:
%
% X_test = test input matrix (each row is an input vector) 
% Y_test = test ground truth assignment_matrix as defined in our CVPR paper:
%        - its elements are 0 or 1
%        - it contains only one nonzero element per row
%        - it contains at least one nonzero element per column
% L = linear transformation matrix such that M = LL'
% (* optional *) apply_SVD (default true) = apply Singular Value
%        Decomposition on the projected test set before applying K-means
% (* optional *) K (default size(Y_test,2)) = Number of desired test clusters
%        
%
%
% Output:
%
% predicted_clusters = numeric column vector that contains cluster indices
% frobenius loss = Frobenius loss obtained on the test set
% rand_loss = Rand loss obtained on the test set
% predicted_clusters_svd = numeric column vector that contains cluster indices when performing SVD on the projected test set before applying K-means
% frobenius loss_svd = Frobenius loss obtained on the test set when performing SVD on the projected test set before applying K-means
% rand_loss_svd = Rand loss obtained on the test set when performing SVD on the projected test set before applying K-means

if nargin < 6
    nb_reinitializations = 20;
end
if nargin < 5
    K = size(Y_test,2);
end
if nargin < 4
    apply_SVD = true;
end

X = X_test * L;

if ~issparse(Y_test)
    Y_test = sparse(Y_test);
end

% applying K-means
predicted_clusters = kmeans(X, K, 'Replicates', nb_reinitializations);

% creating predicted assignment matrix
predicted_Y = sparse(1:length(predicted_clusters), predicted_clusters, 1, size(predicted_clusters,1),K);

J_test = bsxfun(@rdivide,Y_test,sqrt(sum(Y_test)));
predicted_J = bsxfun(@rdivide,predicted_Y,sqrt(sum(predicted_Y)));

% compute the Frobenius loss
frobenius_loss = norm((predicted_J * predicted_J') - (J_test * J_test'),'fro').^2;
% compute the rand loss
%rand_loss = full(sum(sum((predicted_Y * predicted_Y'- Y_test * Y_test').^2))) / (size(Y_test,1)*(size(Y_test,1)-1))
rand_loss = full(sum(sum(abs(predicted_Y * predicted_Y'- Y_test * Y_test')))) / (size(Y_test,1)*(size(Y_test,1)-1));

predicted_clusters_svd = -inf;
frobenius_loss_svd = -inf;
rand_loss_svd = -inf;

if apply_SVD
    [U,~,~] = svd(X,'econ');
    if K < size(U,2)
        X = U(:,1:K);
    else
        X = U;
    end
    % applying K-means
    predicted_clusters_svd = kmeans(X, K, 'Replicates', nb_reinitializations);
    
    % creating predicted assignment matrix
    predicted_Y = sparse(1:length(predicted_clusters_svd), predicted_clusters_svd, 1, size(predicted_clusters_svd,1),K);
    
    J_test = bsxfun(@rdivide,Y_test,sqrt(sum(Y_test)));
    predicted_J = bsxfun(@rdivide,predicted_Y,sqrt(sum(predicted_Y)));
    % compute the Frobenius loss
    frobenius_loss_svd = norm((predicted_J * predicted_J') - (J_test * J_test'),'fro').^2;
    % compute the rand loss
    %rand_loss = full(sum(sum((predicted_Y * predicted_Y'- Y_test * Y_test').^2))) / (size(Y_test,1)*(size(Y_test,1)-1))
    rand_loss_svd = full(sum(sum(abs(predicted_Y * predicted_Y'- Y_test * Y_test')))) / (size(Y_test,1)*(size(Y_test,1)-1));
end


end
