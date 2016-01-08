function error = compute_training_error( X, labels,w,v )
%COMPUTE_TRAINING_ERROR Summary of this function goes here
%   Detailed explanation goes here
    error = 0;
    for i = 1:size(X,2)
        
        input = X(:,i);
        y = softmax(v*sigmoid(w*input));
        
        max_idx = find(y==max(y));
        
        if max_idx ~= labels(i) 
            error = error +1;
        end
        
    end
    
    error = error/(size(X,2)*2);
    
end

