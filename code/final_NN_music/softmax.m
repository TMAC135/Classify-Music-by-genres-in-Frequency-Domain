function y = softmax( x )
%SOFTMAX Summary of this function goes here
%   Detailed explanation goes here

%     y = 1./(1 + exp(-x));
      y = exp(x)./sum(exp(x));
        



end

