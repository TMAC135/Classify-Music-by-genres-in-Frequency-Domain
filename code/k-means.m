%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%   Here is the package to prepprocessing function to partition the 
% total data into two parts -- training data and test data. 
% 
%   Input: ds.mat
% 
%   Output: X_training.mat and X_test.mat
% 
%   Parameters: training_percentage: the percentage of training data(if 
%             it is 0.1,means training data=100,test data=900)
%               dimentionAfterPCA: new dimention after PCA,if we want 
%             visualize it in plot,choose it as 3.
%                
% 
%   Notation:  1: training_percentage is in range(0,1). 
%              2: The size of X_traing and X_test is (d+1)*n.
%              where d is the dimension of the data after PCA, 
%              n is the number of data, the last row of the data is the 
%              true lable of the corresponding data.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all;
clear all;

%% Load  the feature vectors for each types
load ds;
label = ds.output;
fileId = ds.fileId;
fileClassId = ds.fileClassId;
X = ds.input;
label_name = ds.outputName;
file = ds.file;

%% Set the Parameters
training_percentage = 0.7;
dimentionAfterPCA = 20;

%% Set the label for each classifier
label_X1 = (label == 1);
label_X2 = (label == 2);
label_X3 = (label == 3);
label_X4 = (label == 4);
label_X5 = (label == 5);
label_X6 = (label == 6);
label_X7 = (label == 7);
label_X8 = (label == 8);
label_X9 = (label == 9);
label_X10 = (label == 10);

X1 = X(:,label_X1);
X2 = X(:,label_X2);
X3 = X(:,label_X3);
X4 = X(:,label_X4);
X5 = X(:,label_X5);
X6 = X(:,label_X6);
X7 = X(:,label_X7);
X8 = X(:,label_X8);
X9 = X(:,label_X9);
X10 = X(:,label_X10);


%% PCA for model reduction if necessary
% covariance: 156*156 matrix and each column is the coefficients for one
%             component.
% eigvalue_vector: 156*1,vector of eigen values.
% variance_proportion: 156*1 vector of percentage of total variance.
[covariance, eigvalue_vector, variance_proportion] = pcacov(cov(X'));

% reconstruction matrix (156*dimentionAfterPCA)
rec_matrix = covariance(:,1:dimentionAfterPCA);

% data after dimention reduction 
X1_afterPCA = rec_matrix'*X1;
X2_afterPCA = rec_matrix'*X2;
X3_afterPCA = rec_matrix'*X3;
X4_afterPCA = rec_matrix'*X4;
X5_afterPCA = rec_matrix'*X5;
X6_afterPCA = rec_matrix'*X6;
X7_afterPCA = rec_matrix'*X7;
X8_afterPCA = rec_matrix'*X8;
X9_afterPCA = rec_matrix'*X9;
X10_afterPCA = rec_matrix'*X10;

% plot the variance proportion figure
tmp = 0;
figure(1);
hold on;
for i = 1:1:length(variance_proportion)
    tmp = tmp +variance_proportion(i);
    plot(i,tmp,'*');
end
hold off;

% Project the data into 3D,need to set the dimentionAfterPCA to 3
% figure(2);
% hold on;
% scatter3(X1_afterPCA(1,:,:),X1_afterPCA(2,:,:),X1_afterPCA(3,:,:),'g','filled');
% scatter3(X2_afterPCA(1,:,:),X2_afterPCA(2,:,:),X2_afterPCA(3,:,:),'r');
% scatter3(X3_afterPCA(1,:,:),X3_afterPCA(2,:,:),X3_afterPCA(3,:,:),'b');
% scatter3(X4_afterPCA(1,:,:),X4_afterPCA(2,:,:),X4_afterPCA(3,:,:),'y');
% scatter3(X5_afterPCA(1,:,:),X5_afterPCA(2,:,:),X5_afterPCA(3,:,:),'m');
% scatter3(X6_afterPCA(1,:,:),X6_afterPCA(2,:,:),X6_afterPCA(3,:,:),'c');
% scatter3(X7_afterPCA(1,:,:),X7_afterPCA(2,:,:),X7_afterPCA(3,:,:),'k');
% scatter3(X8_afterPCA(1,:,:),X8_afterPCA(2,:,:),X8_afterPCA(3,:,:),'d');
% scatter3(X9_afterPCA(1,:,:),X9_afterPCA(2,:,:),X9_afterPCA(3,:,:),'d','filled');
% scatter3(X10_afterPCA(1,:,:),X10_afterPCA(2,:,:),X10_afterPCA(3,:,:),'*');
% hold off;

%% Construct the training data and test data
% add the label to each data
X1_afterPCA=[X1_afterPCA;ones(1,100)];
X2_afterPCA=[X2_afterPCA;2*ones(1,100)];
X3_afterPCA=[X3_afterPCA;3*ones(1,100)];
X4_afterPCA=[X4_afterPCA;4*ones(1,100)];
X5_afterPCA=[X5_afterPCA;5*ones(1,100)];
X6_afterPCA=[X6_afterPCA;6*ones(1,100)];
X7_afterPCA=[X7_afterPCA;7*ones(1,100)];
X8_afterPCA=[X8_afterPCA;8*ones(1,100)];
X9_afterPCA=[X9_afterPCA;9*ones(1,100)];
X10_afterPCA=[X10_afterPCA;10*ones(1,100)];


% add the test data for each label

split_point = round(100*training_percentage);
seq = randperm(100);

X1_training = X1_afterPCA(:,seq(1:split_point));
X1_test = X1_afterPCA(:,seq(split_point+1:end));

X2_training = X2_afterPCA(:,seq(1:split_point));
X2_test = X2_afterPCA(:,seq(split_point+1:end));

X3_training = X3_afterPCA(:,seq(1:split_point));
X3_test = X3_afterPCA(:,seq(split_point+1:end));

X4_training = X4_afterPCA(:,seq(1:split_point));
X4_test = X4_afterPCA(:,seq(split_point+1:end));

X5_training = X5_afterPCA(:,seq(1:split_point));
X5_test = X5_afterPCA(:,seq(split_point+1:end));

X6_training = X6_afterPCA(:,seq(1:split_point));
X6_test = X6_afterPCA(:,seq(split_point+1:end));

X7_training = X7_afterPCA(:,seq(1:split_point));
X7_test = X7_afterPCA(:,seq(split_point+1:end));

X8_training = X8_afterPCA(:,seq(1:split_point));
X8_test = X8_afterPCA(:,seq(split_point+1:end));

X9_training = X9_afterPCA(:,seq(1:split_point));
X9_test = X9_afterPCA(:,seq(split_point+1:end));

X10_training = X10_afterPCA(:,seq(1:split_point));
X10_test = X10_afterPCA(:,seq(split_point+1:end));

% stack all the test data and training data
X_training=[X1_training,X2_training,X3_training,X4_training,X5_training ...
           X6_training,X7_training,X8_training,X9_training,X10_training];
       
X_test=[X1_test,X2_test,X3_test,X4_test,X5_test ...
           X6_test,X7_test,X8_test,X9_test,X10_test];
       
% pack X_training and X_test
save('X_training.mat','X_training');
save('X_test.mat','X_test');     

%% k-means
rng(1)
% first with 10 labels
k = 10;
idx = kmeans(transpose(X),k);
% this result is terrible and since it's unsurprised, it's very hard to get
% the result

% Then try with 4 selected labels
newdata = transpose([X2_test X2_training X5_test X5_training X7_test X7_training X9_test X9_training]);
X_kmeans = transpose([X_test X_training]);
idxnew = kmeans(newdata, 4);

% assign true labels using conditional probability
km_labels = repmat(1:4,[100 1]);
km_labels = km_labels(:)';

mode(idxnew(1:100));
mode(idxnew(101:200));
mode(idxnew(201:300));
mode(idxnew(301:400));

tabulate(idxnew(101:200));
tabulate(idxnew(301:400));

kmestlabel=zeros(400, 1);
for i = 1:400
    if idxnew(i) == 1
        kmestlabel(i) = 4;
    end
    if idxnew(i) == 2
        kmestlabel(i) = 3;
    end
    if idxnew(i) == 3
        kmestlabel(i) = 1;
    end
    if idxnew(i) == 4
        kmestlabel(i) = 2;
    end
end

% error rate
error=0;
for i = 1:400
    if kmestlabel(i) ~= km_labels(i)
            error = error + 1;
    end   
end
erate = error/400

tabulate(kmestlabel(1:100));
tabulate(kmestlabel(101:200));
tabulate(kmestlabel(201:300));
tabulate(kmestlabel(301:400));
