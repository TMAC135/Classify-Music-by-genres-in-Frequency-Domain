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




% load PCA.mat

clc;    % Clear command window.
clear;  % Delete all variables.
close all;  % Close all figure windows except those created by imtool.


%% Load  the feature vectors for each types
load ds;
label = ds.output;
fileId = ds.fileId;
fileClassId = ds.fileClassId;
X = ds.input;
label_name = ds.outputName;
file = ds.file;

%% Set the Parameters
training_percentage = 0.8;
dimentionAfterPCA = 20;

num_train = 80;
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
X_train =[];
X_test = [];
X = [X;label];
X1 = X(:,label_X1);
X_train = [X_train , X1(:,1:num_train)];
X_test = [X_test, X1(:,num_train+1:end)];
X2 = X(:,label_X2);
X_train = [X_train , X2(:,1:num_train)];
X_test = [X_test, X2(:,num_train+1:end)];
X3 = X(:,label_X3);
X_train = [X_train , X3(:,1:num_train)];
X_test = [X_test, X3(:,num_train+1:end)];
X4 = X(:,label_X4);
X_train = [X_train , X4(:,1:num_train)];
X_test = [X_test, X4(:,num_train+1:end)];

X5 = X(:,label_X5);
X_train = [X_train , X5(:,1:num_train)];
X_test = [X_test, X5(:,num_train+1:end)];
% 
X6 = X(:,label_X6);
X_train = [X_train , X6(:,1:num_train)];
X_test = [X_test, X6(:,num_train+1:end)];

X7 = X(:,label_X7);
X_train = [X_train , X7(:,1:num_train)];
X_test = [X_test, X7(:,num_train+1:end)];

X8 = X(:,label_X8);
X_train = [X_train , X8(:,1:num_train)];
X_test = [X_test, X8(:,num_train+1:end)];

X9 = X(:,label_X9);
X_train = [X_train , X9(:,1:num_train)];
X_test = [X_test, X9(:,num_train+1:end)];

X10 = X(:,label_X10);
X_train = [X_train , X10(:,1:num_train)];
X_test = [X_test, X10(:,num_train+1:end)];

X_train_1 = X_train(1:end-1,:);
X_test_1 = X_test(1:end-1,:);

label_train = X_train(end,:);
label_test =X_test(end,:);

labels = label_train';

num_ouput = 10;
labels_vector = 0.*ones(num_ouput, size(labels, 1));
for n = 1: size(labels, 1)
    labels_vector(labels(n), n) = 1;
end

learningRate = 0.01;
% normalization 
X_train_mean =mean(X_train_1,2);
X_train_std = std(X_train_1')';
X_train_std = repmat(X_train_std,[1 800]);
X_train_1 = X_train_1-repmat(X_train_mean,[1 800]);
X_train_1 = X_train_1./X_train_std;

hidden_vector2 =[]; train_error2 =[]; test_error2 =[];
for i = 1:25
    hidden_num = 5 + (i-1)*5;
    
    [w, v] = Train_Neural_Network(X_train_1,labels_vector,hidden_num,learningRate);
    
    X_train_mean =mean(X_test_1,2);
    X_train_std = std(X_test_1')';
    X_train_std = repmat(X_train_std,[1 200]);
    X_test_1 = X_test_1-repmat(X_train_mean,[1 200]);
    X_test_1 = X_test_1./X_train_std;
    
    test_e = compute_training_error(X_test_1,label_test',w,v);
% training_error
    
%     X_train_mean =mean(X_test_1,2);
%     X_train_std = std(X_test_1')';
%     X_train_std = repmat(X_train_std,[1 200]);
%     X_test_1 = X_test_1-repmat(X_train_mean,[1 200]);
%     X_test_1 = X_test_1./X_train_std;

    train_e = compute_training_error(X_train_1,label_train',w,v);

    
    fprintf('hidden units number: %d  train_error: %f  test_e error: %f\n',hidden_num,train_e, test_e);
    hidden_vector2 = [hidden_vector2;hidden_num];
    train_error2 = [train_error2;train_e];
    test_error2 = [test_error2;test_e];

end



















