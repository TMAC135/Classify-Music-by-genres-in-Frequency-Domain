clc;    % Clear command window.
clear;  % Delete all variables.
close all;  % Close all figure windows except those created by imtool.

% addpath ./nnet/
% addpath ./nnet/nnet/nninitnetwork/

% net = patternnet2(10);

A = importdata('optdigits.tra');
X = double(A(:,1:64)); 
labels = A(:,65);

% hidden_num = 15;
% learningRate = 0.01;
rng(15);

% mean_X = mean(X(:,1:64),1);
% X_var = std(X(:,1:64));
% 
% X = [(X(:,1:64) - repmat(mean_X,[size(X,1),1]))./(repmat(X_var,[size(X,1),1])) X(:,65)];
% 
% X(isnan(X))=0;

%% INITIALIZATION
labels_vector = 0.*ones(10, size(labels, 1));
for n = 1: size(labels, 1)
    labels_vector(labels(n) + 1, n) = 1;
end;

hidden_num = 5;
learningRate = 0.01;
[w, v] = Train_Neural_Network(X',labels_vector,hidden_num,learningRate);
training_error = compute_training_error(X',labels,w,v);
fprintf('hidden units number: %d    trainning error: %f\n',hidden_num,training_error);
% training_error

 hidden_num = 10;
learningRate = 0.01;
[w, v] = Train_Neural_Network(X',labels_vector,hidden_num,learningRate);
training_error = compute_training_error(X',labels,w,v);
fprintf('hidden units number: %d    trainning error: %f\n',hidden_num,training_error);

% training_error

hidden_num = 15;
learningRate = 0.01;
[w, v] = Train_Neural_Network(X',labels_vector,hidden_num,learningRate);
training_error = compute_training_error(X',labels,w,v);
% training_error
fprintf('hidden units number: %d    trainning error: %f\n',hidden_num,training_error);





