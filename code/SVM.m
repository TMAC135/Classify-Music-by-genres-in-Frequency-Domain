%% SVM implementations of the muisc classifiers.


% if we classify the 10 music types ,SVM will not converges.

% clear all;
% close all;
% 
% % load the training and test data
% load X_test.mat;
% load X_training.mat;
% 
% row=size(X_test,1);
% column=size(X_test,2);
% 
% 
% % extract the data and label for all 10 types
% training = X_training(1:row-1,:); 
% training_label = X_training(row,:);
% 
% test=X_test(1:row-1,:);
% test_label=X_test(row,:)';
% 
% number_test=size(test,1);


%% Support Vector Machine implementations of the muisc classifiers.
clear all;
close all;

% load the training and test data
load X_test.mat;
load X_training.mat;

row=size(X_test,1);
column_test=size(X_test,2);
column_training=size(X_training,2);



% % extract the data and label for 4 types(label=2,6,7,10)
% number_training_each=column_training/10;
% training = [X_training(1:row-1,(number_training_each+1):(number_training_each*2))';X_training(1:row-1,(number_training_each*5+1):(number_training_each*6))';...
%     X_training(1:row-1,(number_training_each*6+1):(number_training_each*7))';X_training(1:row-1,(number_training_each*9+1):(number_training_each*10))']; 
% 
% training_label = [X_training(row,(number_training_each+1):(number_training_each*2))';X_training(row,(number_training_each*5+1):(number_training_each*6))';...
%     X_training(row,(number_training_each*6+1):(number_training_each*7))';X_training(row,(number_training_each*9+1):(number_training_each*10))'];
% 
% 
% number_test_each = 100-number_training_each;
% test = [X_test(1:row-1,(number_test_each+1):(number_test_each*2))';X_test(1:row-1,(number_test_each*5+1):(number_test_each*6))';...
%     X_test(1:row-1,(number_test_each*6+1):(number_test_each*7))';X_test(1:row-1,(number_test_each*9+1):(number_test_each*10))']; 
% 
% test_label = [X_test(row,(number_test_each+1):(number_test_each*2))';X_test(row,(number_test_each*5+1):(number_test_each*6))';...
%     X_test(row,(number_test_each*6+1):(number_test_each*7))';X_test(row,(number_test_each*9+1):(number_test_each*10))'];
% 
% number_test=number_test_each*4;

% extract the data and label for 4 types(label=2,5,7,9)
number_training_each=column_training/10;
training = [X_training(1:row-1,(number_training_each+1):(number_training_each*2))';X_training(1:row-1,(number_training_each*4+1):(number_training_each*5))';...
    X_training(1:row-1,(number_training_each*6+1):(number_training_each*7))';X_training(1:row-1,(number_training_each*8+1):(number_training_each*9))']; 

training_label = [X_training(row,(number_training_each+1):(number_training_each*2))';X_training(row,(number_training_each*4+1):(number_training_each*5))';...
    X_training(row,(number_training_each*6+1):(number_training_each*7))';X_training(row,(number_training_each*8+1):(number_training_each*9))'];


number_test_each = 100-number_training_each;
test = [X_test(1:row-1,(number_test_each+1):(number_test_each*2))';X_test(1:row-1,(number_test_each*4+1):(number_test_each*5))';...
    X_test(1:row-1,(number_test_each*6+1):(number_test_each*7))';X_test(1:row-1,(number_test_each*8+1):(number_test_each*9))']; 

test_label = [X_test(row,(number_test_each+1):(number_test_each*2))';X_test(row,(number_test_each*4+1):(number_test_each*5))';...
    X_test(row,(number_test_each*6+1):(number_test_each*7))';X_test(row,(number_test_each*8+1):(number_test_each*9))'];

% make the test label 2,5,7,9 to 1,2,3,4
for i=1:size(test_label)
    switch test_label(i)
        case 2
           test_label(i)=1;
        case 5
            test_label(i)=2;
        case 7
            test_label(i)=3;
        case 9
            test_label(i)=4;       
    end
end


number_test=number_test_each*4;



%% training and test of the data
prediction_label=multisvm(training,training_label,test);

% get the accuracy matrix
Accuracy = zeros(4,4);

for j=1:size(test_label)
    Accuracy(prediction_label(j),test_label(j))=...
        Accuracy(prediction_label(j),test_label(j))+1;
end





