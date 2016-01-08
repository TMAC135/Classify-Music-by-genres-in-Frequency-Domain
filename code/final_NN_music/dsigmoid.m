function y = dsigmoid( x )
%DSIGMOID Summary of this function goes here
%   Detailed explanation goes here
        y = sigmoid(x).*(1-sigmoid(x));
        
end

