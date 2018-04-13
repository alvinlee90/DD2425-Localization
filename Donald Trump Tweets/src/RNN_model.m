classdef RNN_model 
    % Class object for the RNN (recurrent neural network)
    properties
        K               % Number of unique outputs
        m               % Dimension of hidden state
        seq_length      % Sequence length
        
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
        function obj = RNN_model(m, K, seq_length, sig)
           if nargin > 0
               % Network parameters
               obj.K = K;
               obj.m = m;
               obj.seq_length = seq_length;

               % Bias vectors set to 0 
               obj.b_f = zeros(m, 1);
               obj.b_i = zeros(m, 1);
               obj.b_o = zeros(m, 1);
               obj.b_c = zeros(m, 1);
               obj.c = zeros(K, 1);
               
               % Weights randomly initalized 
               obj.U_f = randn(m, K) * sig;
               obj.U_i = randn(m, K) * sig;
               obj.U_o = randn(m, K) * sig;
               obj.U_c = randn(m, K) * sig;
               obj.W_f = rand(m, m) * sig;
               obj.W_i = rand(m, m) * sig;
               obj.W_o = rand(m, m) * sig;
               obj.W_c = rand(m, m) * sig;
               obj.V = rand(K, m) * sig;
           end
        end
        
        % Function to calculate the hidden state and probabilities 
        function [h, c, p, c_hat, o, i, f] = Evaluate(obj, h0, x, c0)
            % Forget gate
            f = logsig([obj.W_f] * h0 + [obj.U_f] * x + [obj.b_f]); 
            
            % Input gate
            i = logsig([obj.W_i] * h0 + [obj.U_i] * x + [obj.b_i]); 
            
            % Output gate
            o = logsig([obj.W_o] * h0 + [obj.U_o] * x + [obj.b_o]); 
            
            % Cell state vector
            c_hat = tanh([obj.W_c] * h0 + [obj.U_c] * x + [obj.b_c]);
            
            c = f .* c0 + i .* c_hat;
            
            % Hidden state
            h = o .* tanh(c);
                        
            % Probabilities  
            y = [obj.V] * h + [obj.c];
            
            p = softmax(y);
        end
    end
end

