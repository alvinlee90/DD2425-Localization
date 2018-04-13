classdef RNN_model 
    % Class object for the RNN (recurrent neural network)
    properties
        K               % Number of unique outputs
        m               % Dimension of hidden state
        seq_length      % Sequence length
        b               % Bias vector
        c               % Bias vector
        U               % Weight vector
        W               % Weight vector
        V               % Weight vector
    end
    
    methods
        % Constructor 
        function obj = RNN_model(m, K, seq_length, sig)
           if nargin > 0
               % Network parameters
               obj.K = K;
               obj.m = m;
               obj.seq_length = seq_length;

               % Bias vectors set to 0 
               obj.b = zeros(m, 1);
               obj.c = zeros(K, 1);
               
               % Weights randomly initalized 
               obj.U = randn(m, K) * sig;
               obj.W = rand(m, m) * sig;
               obj.V = rand(K, m) * sig;
%                obj.U = eye(m, K);
%                obj.W = eye(m, m);
%                obj.V = eye(K, m);
           end
        end
        
        % Function to calculate the hidden state and probabilities 
        function [h, p] = Evaluate(obj, h0, x)
            % Calculate activation
            a = [obj.W] * h0 + [obj.U] * x + [obj.b]; 
            
            % Calculate hidden state (Tanh function)
            h = tanh(a);
%             h = max(a, 0); 
            
            % Calculate probabilities 
            o = [obj.V] * h + [obj.c];
            p = softmax(o);
        end
    end
end

