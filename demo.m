close all;

%%% Metric Learning for Cluster Analysis (MLCA) demo
%%% The current demo corresponds to the toy experiment reported in Table 2
%%% (Table 3 if noisy = false, and Table 1 if same_cluster_size = true)
%%% of our CVPR 2016 paper: Closed-Form Training of Mahalanobis Distance for Supervised Clustering.
%%% by M.T. Law, Y. Yu, M. Cord, E.P. Xing 

d = 3; % input space dimensionality
T = 500; % minimum number of examples per subcluster

nb_classes = 3; % number of ground truth categories
K = nb_classes; % number of desired clusters
noisy = true; % the experiment corresponds to Table 2 if noisy = true, and to Table 3 otherwise

same_cluster_size = false; % if same_cluster_size = true and noisy = true, the experiment corresponds to Table 1
if same_cluster_size
    defined_size_1 = T;
    defined_size_2 = T;
    defined_size_3 = T;
else
    defined_size_1 = T;
    defined_size_2 = 2*T;
    defined_size_3 = 4*T;
end

%%%
% Creation of the training dataset
%%%

u = zeros(2*T*K,K); 
space = 6;
double_space = 10*space;

C1_1 = repmat([-space,0,0],defined_size_1,1); % Creation of the first subcluster of the first category with T observations
C1_2 = repmat([0, double_space,0],defined_size_1,1); % Creation of the second subcluster of the first category with T observations
C2_1 = repmat([space,0,0],defined_size_2,1); % Creation of the first subcluster of the second category with 2T observations
C2_2 = repmat([space,double_space,0],defined_size_2,1); % Creation of the second subcluster of the second category with 2T observations
C3_1 = repmat([space*0.5,double_space,space],defined_size_3,1); % Creation of the first subcluster of the third category with 4T observations
C3_2 = repmat([-space,0,space],defined_size_3,1); % Creation of the second subcluster of the third category with 4T observations


if ~noisy
    noisy = 0.5;
end

X1_1 = noisy * randn(defined_size_1,d)+ C1_1; % Inclusion of normally distributed noise
X1_2 = noisy * randn(defined_size_1,d)+ C1_2; % Inclusion of normally distributed noise
X2_1 = noisy * randn(defined_size_2,d)+ C2_1; % Inclusion of normally distributed noise
X2_2 = noisy * randn(defined_size_2,d)+ C2_2; % Inclusion of normally distributed noise
X3_1 = noisy * randn(defined_size_3,d)+ C3_1; % Inclusion of normally distributed noise
X3_2 = noisy * randn(defined_size_3,d)+ C3_2; % Inclusion of normally distributed noise
figure1 = figure;
plot3(X1_1(:,1), X1_1(:,2), X1_1(:,3), 'bx',X1_2(:,1), X1_2(:,2), X1_2(:,3), 'bx',X2_1(:,1), X2_1(:,2), X2_1(:,3), 'ro',X2_2(:,1), X2_2(:,2), X2_2(:,3), 'ro',X3_1(:,1), X3_1(:,2), X3_1(:,3), 'gs',X3_2(:,1), X3_2(:,2), X3_2(:,3), 'gs');
axis equal
set(gca,'FontSize',15)  
title('Training data (one color per cluster)')


size_1 = size(C1_1,1) + size(C1_2,1);
size_2 = size(C2_1,1) + size(C2_2,1);
size_3 = size(C3_1,1) + size(C3_2,1);


X = [X1_1;X1_2;X2_1;X2_2;X3_1;X3_2];

Y = sparse((1:size(X,1)),[ones(size_1,1);repmat(2,size_2,1);repmat(3,size_3,1)],1,size(X,1),K);



print_training_time = true;
% call training function
[L] = mlca_train(X,Y,print_training_time,false);



nb_test = 1000;


noisy_test = noisy;

% Creation of the test dataset with the same settings 
% (2 * nb_test) observations per cluster

Q1_1 = noisy_test * randn(nb_test,d)+ repmat(C1_1(1,:),nb_test,1);
Q1_2 = noisy_test * randn(nb_test,d)+ repmat(C1_2(1,:),nb_test,1);
Q2_1 = noisy_test * randn(nb_test,d)+ repmat(C2_1(1,:),nb_test,1);
Q2_2 = noisy_test * randn(nb_test,d)+ repmat(C2_2(1,:),nb_test,1);
Q3_1 = noisy_test * randn(nb_test,d)+ repmat(C3_1(1,:),nb_test,1);
Q3_2 = noisy_test * randn(nb_test,d)+ repmat(C3_2(1,:),nb_test,1);


X_test = [Q1_1;Q1_2;Q2_1;Q2_2;Q3_1;Q3_2];

% Creation of the test assignment matrix with the same settings
Y_test = sparse((1:size(X_test,1)),[ones(2*nb_test,1);repmat(2,2*nb_test,1);repmat(3,2*nb_test,1)],1,size(X_test,1),K);

% apply SVD to the transformed test set
apply_SVD = true;
[predicted_clusters, frobenius_loss, rand_loss, predicted_clusters_svd, frobenius_loss_svd, rand_loss_svd] = clustering_test(X_test,Y_test,L, apply_SVD);

predicted_X1 = X_test(predicted_clusters == 1,:);
predicted_X2 = X_test(predicted_clusters == 2,:);
predicted_X3 = X_test(predicted_clusters == 3,:);

figure2 = figure;
plot3(predicted_X1(:,1), predicted_X1(:,2), predicted_X1(:,3), 'mx',predicted_X2(:,1), predicted_X2(:,2), predicted_X2(:,3), 'co',predicted_X3(:,1), predicted_X3(:,2), predicted_X3(:,3), 'ks');
axis equal
set(gca,'FontSize',15)  
title('Test data (one color per cluster) without SVD')

Projected_test = X_test * L;

projected_X1 = Projected_test(predicted_clusters == 1,:);
projected_X2 = Projected_test(predicted_clusters == 2,:);
projected_X3 = Projected_test(predicted_clusters == 3,:);

figure3 = figure;
plot3(projected_X1(:,1), projected_X1(:,2), projected_X1(:,3), 'mx',projected_X2(:,1), projected_X2(:,2), projected_X2(:,3), 'co',projected_X3(:,1), projected_X3(:,2), projected_X3(:,3), 'ks');
axis equal
set(gca,'FontSize',15)  
title('Projected 3-class test data without SVD')


fprintf('Test Frobenius loss without SVD: %f\n',frobenius_loss);
fprintf('Test rand loss without SVD: %f\n',rand_loss);



predicted_X1 = X_test(predicted_clusters_svd == 1,:);
predicted_X2 = X_test(predicted_clusters_svd == 2,:);
predicted_X3 = X_test(predicted_clusters_svd == 3,:);

figure4 = figure;
plot3(predicted_X1(:,1), predicted_X1(:,2), predicted_X1(:,3), 'yx',predicted_X2(:,1), predicted_X2(:,2), predicted_X2(:,3), 'bo',predicted_X3(:,1), predicted_X3(:,2), predicted_X3(:,3), 'ks');
axis equal
title('Test data (one color per cluster) with SVD')


[U,~,~] = svd(Projected_test,'econ');
if K < size(U,2)
    Projected_test = U(:,1:K);
else
    Projected_test = U;
end

projected_X1 = Projected_test(predicted_clusters_svd == 1,:);
projected_X2 = Projected_test(predicted_clusters_svd == 2,:);
projected_X3 = Projected_test(predicted_clusters_svd == 3,:);

figure5 = figure;
plot3(projected_X1(:,1), projected_X1(:,2), projected_X1(:,3), 'yx',projected_X2(:,1), projected_X2(:,2), projected_X2(:,3), 'bo',projected_X3(:,1), projected_X3(:,2), projected_X3(:,3), 'ks');
axis equal
set(gca,'FontSize',15)  
title('Projected 3-class test data with SVD')


fprintf('Test Frobenius loss with SVD: %f\n',frobenius_loss_svd);
fprintf('Test rand loss with SVD: %f\n',rand_loss_svd);



