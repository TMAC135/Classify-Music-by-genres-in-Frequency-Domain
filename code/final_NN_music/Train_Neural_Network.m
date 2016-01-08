function [w, v] = Train_Neural_Network( X,labels,hidden_num,learningRate)
%TRAIN_NEURAL_NETWORK Summary of this function goes here
%   Detailed explanation goes here

    training_size = size(X, 2);
    intput_d = size(X, 1);
    output_d = size(labels, 1);
    
    %% w is from input to hidden 
    w = -0.01+rand(hidden_num, intput_d)*0.02;
    %% v is from hidden to ouput 
    v = -0.01+rand(output_d, hidden_num)*0.02;
    
    random_index = zeros(training_size,1);
    old_train_error = 100;
    error_change = 100;
    for iteration = 1:500
%     iteration =0;
    
%     while error_change > 10^-4 
%          iteration
         iteration = iteration+1;
         
         for k = 1:500
            random_index(k) = floor(rand(1)*training_size + 1);
            
            % Feed forward network
            input = X(:, random_index(k));
            a = w*input;
            z = sigmoid(a);
            o = v*z;
            y = softmax(o);
            
            labels_batch = labels(:, random_index(k));
            
            % Backpropagation
            delta_v = dsoftmax(o).*(y - labels_batch);
            
            delta_w = dsigmoid(a).*(v'*delta_v);
            
            v = v - learningRate.*delta_v*z';
            w = w - learningRate.*delta_w*input';
         end
        
         
        train_error =0;
        for k = 1:500
            y = X(:, random_index(k));
            r = labels(:, random_index(k));
            
            train_error = train_error + norm(softmax(v*sigmoid(w*y)) - r, 2);
        end;
        
        train_error = train_error/500;
        error_change = abs(train_error- old_train_error);
        
        
        old_train_error = train_error;
%         error_change
        
    end;
end

