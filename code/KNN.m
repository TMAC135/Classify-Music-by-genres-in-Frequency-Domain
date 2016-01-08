%% K Nearest Neighbor implementations of the muisc classifiers.
clear all;
close all;

% load the training and test data
load X_test.mat;
load X_training.mat;

row=size(X_test,1);
column=size(X_test,2);


% extract the data and label for all 10 types
training = X_training(1:row-1,:)'; 
training_label = X_training(row,:)';

test=X_test(1:row-1,:)';
test_label=X_test(row,:)';

number_test=size(test,1);


%% classify all 10 types


% K=5,Euclidean Distance

test_predict_euclidean_nearest = knnclassify(test, training,...
                                training_label,5,'euclidean','nearest');
tmp = (test_predict_euclidean_nearest==test_label);
error_5_euclidean_nearest = (number_test-sum(tmp))/number_test

% K=10,Euclidean Distance

test_predict_euclidean_nearest = knnclassify(test, training,...
                                training_label,10,'euclidean','nearest');
tmp = (test_predict_euclidean_nearest==test_label);
error_10_euclidean_nearest = (number_test-sum(tmp))/number_test

% K=20,Euclidean Distance

test_predict_euclidean_nearest = knnclassify(test, training,...
                                training_label,20,'euclidean','nearest');
tmp = (test_predict_euclidean_nearest==test_label);
error_20_euclidean_nearest = (number_test-sum(tmp))/number_test

% K=5,city block Distance
test_predict_cityblock_nearest = knnclassify(test, training,...
                                training_label,5,'cityblock','nearest');
tmp = (test_predict_cityblock_nearest==test_label);
error_5_cityblock_nearest = (number_test-sum(tmp))/number_test

% K=10,city block Distance
test_predict_cityblock_nearest = knnclassify(test, training,...
                                training_label,10,'cityblock','nearest');
tmp = (test_predict_cityblock_nearest==test_label);
error_10_cityblock_nearest = (number_test-sum(tmp))/number_test

% K=20,city block Distance
test_predict_cityblock_nearest = knnclassify(test, training,...
                                training_label,20,'cityblock','nearest');
tmp = (test_predict_cityblock_nearest==test_label);
error_20_cityblock_nearest = (number_test-sum(tmp))/number_test

%% Save the corrsponding error
% S={};
% S.error=[error_5_euclidean_nearest,error_10_euclidean_nearest,error_20_euclidean_nearest...
%     error_5_cityblock_nearest,error_10_cityblock_nearest,error_20_cityblock_nearest];
% S.numberOfMusicType=10;
% filename=['error_',num2str(number_test),'(test number)_',num2str(10),'(music type)_',...
%     num2str(row-1),'(dimention)','.mat'];
% save(filename,'-struct','S');


%% get the prediction-action matrix,choose k=10,euclidean distance
Accuracy = zeros(10,10);
% prediction_laebl and test_label
prediction_label = knnclassify(test, training,...
                                training_label,10,'euclidean','nearest');

for j=1:size(test_label)
    Accuracy(prediction_label(j),test_label(j))=...
        Accuracy(prediction_label(j),test_label(j))+1;
end                            











