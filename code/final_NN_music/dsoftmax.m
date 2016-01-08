function y = dsoftmax( x )
%DSOFTMAX Summary of this function goes here
%   Detailed explanation goes here
    y = softmax(x).*(1 - softmax(x));
    
end

