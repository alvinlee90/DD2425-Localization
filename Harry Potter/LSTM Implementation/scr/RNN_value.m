classdef RNN_value < handle
    % Class object for storing the gradients of the RNN
     properties
        b_f             % Bias vector (forget gate)
        b_i             % Bias vector (input gate)
        b_o             % Bias vector (output gate)
        b_c             % Bias vector (cell state)
        U_f             % Weight vector (forget gate)
        U_i             % Weight vector (input gate)
        U_o             % Weight vector (output gate)
        U_c             % Weight vector (cell gate)
        W_f             % Weight vector (forget gate)
        W_i             % Weight vector (input gate)
        W_o             % Weight vector (output gate)
        W_c             % Weight vector (cell gate)
        
        V               % Weight vector       
        c               % Bias vector 
    end
    
    methods
        % Constructor 
        function obj = RNN_value(m, K)
           if nargin > 0
               % Bias vectors set to 0 
               obj.b_f = zeros(m, 1);
               obj.b_i = zeros(m, 1);
               obj.b_o = zeros(m, 1);
               obj.b_c = zeros(m, 1);
               obj.c = zeros(K, 1);
               
               % Weights initalized 
               obj.U_f = zeros(m, K);
               obj.U_i = zeros(m, K);
               obj.U_o = zeros(m, K);
               obj.U_c = zeros(m, K);
               obj.W_f = zeros(m, m);
               obj.W_i = zeros(m, m);
               obj.W_o = zeros(m, m);
               obj.W_c = zeros(m, m);
               obj.V = zeros(K, m);
           end
        end
    end
    
end

