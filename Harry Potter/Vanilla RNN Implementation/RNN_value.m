classdef RNN_value < handle
    % Class object for storing the gradients of the RNN
    properties
        b       % Bias vector
        c       % Bias vector
        U       % Weight vector
        W       % Weight vector
        V       % Weight vector
    end
    
    methods
        % Constructor 
        function obj = RNN_value(m, K)
           if nargin > 0
               % Initialize gradients with 0 
               obj.b = zeros(m, 1);
               obj.c = zeros(K, 1); 
               obj.U = zeros(m, K);
               obj.W = zeros(m, m);
               obj.V = zeros(K, m);
           end
        end
    end
    
end

