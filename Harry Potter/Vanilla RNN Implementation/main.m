clc
clear

%% Main
% Training data file
train_file = 'goblet_book.txt';

% RNN Parameters
m = 128;
seq_length = 25;
sig = 0.001;

% Training Parameters
epoch = 10;
epi = 1e-8;
eta = 0.001;
gamma = 0.9;

% Load training data from file
[x_train, chars, n_output] = LoadData(train_file);

% Container maps for one-hot encoding
key = num2cell(chars);
value = num2cell(1:numel(chars));

char_to_ind = containers.Map(key, value);
int_to_char = containers.Map(value, key);

% Construct RNN and  gradients class
RNN_train = RNN_model(m, n_output, seq_length, sig);
% load('C:\Users\Alvin\Documents\MATLAB\DD2424\Assignment 4\save\rnn_harry_10.mat')
RNN_best = RNN_model(m, n_output, seq_length, 0); 

grad = RNN_value(m, n_output);
m_grad = RNN_value(m, n_output);

% Training parameters initialize
itr = 0;
loss_avg = [];
loss_op = []; 

fileID = fopen('result.txt','w');

% First synthesized text (no training)
h_prev = zeros(RNN_best.m, 1);
x_0 = ConvertOneHot('.', RNN_best.K, char_to_ind);
text = SynthesizeText(RNN_best, int_to_char, h_prev, x_0, 200);
fprintf(fileID, 'Itr: 0\n'); 
fprintf(fileID, text);

% Training loop
for ii = 1:epoch
    tic
    % Reset previous hidden state (h_0) to zeros
    h_prev = zeros(RNN_train.m, 1);
    
    for e = 1:seq_length:(length(x_train) - seq_length - 1)
        % Sample labelled training data
        x_seq = x_train(e:(e + seq_length - 1));
        y_seq = x_train((e + 1):(e + seq_length));
        
        % Convert to one-hot encoding
        x = ConvertOneHot(x_seq, RNN_train.K, char_to_ind);
        y = ConvertOneHot(y_seq, RNN_train.K, char_to_ind);     
        
        % Compute gradients
        [loss, h_prev] = ComputeGradient(RNN_train, grad, x, y, h_prev);
        
        % Store best weights/bias base on loss value
        if isempty(loss_op) || loss_op > loss
            loss_op = loss; 
            RNN_best = RNN_train;
        end 
        
        % Optimizer 
        for f = fieldnames(grad)'
            m_grad.(f{1}) = gamma * m_grad.(f{1}) + (1 - gamma) * (grad.(f{1})).^2;
            RNN_train.(f{1}) = RNN_train.(f{1}) - eta * grad.(f{1}) ...
                ./ sqrt(m_grad.(f{1}) + eta);
        end
        
        % Save exponential moving average of the loss 
        if isempty(loss_avg)
            loss_avg = loss;
        else
            loss_avg = 0.999 * loss_avg + 0.001 * loss; 
            
        end

        itr = itr + 1;
        
        % Print status of training
        if ~mod(itr,500) || itr == 1
            fprintf('Epoch: %d\tItr: %d\tLoss: %f\n', ii, itr, loss_avg);
        end
    
        if ~mod(itr, 10000) 
            text = SynthesizeText(RNN_train, int_to_char, h_prev, x(:,1), 200);
            text = strcat(text, '\n');
            fprintf(text)
            
            fprintf(fileID, '\nItr: %d\n', itr); 
            fprintf(fileID, text);
        end
    end
    toc
end

% Create 1,000 character passage from best model
h_prev = zeros(RNN_best.m, 1);
text = SynthesizeText(RNN_best, int_to_char, h_prev, x_0, 1000);
fprintf(fileID, '\nFinal\n'); 
fprintf(fileID, text);

fclose(fileID);

%% Compute Gradient Function
function [L, h_out] = ComputeGradient(RNN, grad, x, y, h)
% Compute the gradients of the bias and weight matrix with respect to the
% loss

% Threshold for the max/min value of the gradients
threshold = 5;

grad_h_mat = zeros(RNN.m, RNN.seq_length);
grad_a_mat = zeros(RNN.m, RNN.seq_length);

[L, h_mat, p_mat] = ComputeLoss(x, y, RNN, h);

% Gradient of output (o) with respect to loss (dL/do)
g = -(y - p_mat)';

% Gradient of bias c
grad.c = transpose(sum(g, 1)); 

% Gradient of weight V (dL/dV)
temp_g = reshape(g', RNN.K, 1, RNN.seq_length);
temp_h = reshape(h_mat(:,2:end), 1, RNN.m, RNN.seq_length);
grad.V = sum(bsxfun(@times, temp_g, temp_h), 3);

% Calculate calculate gradients of a_t and h_t at t = seq_length
grad_o = g(end,:); 
grad_h = transpose(grad_o * RNN.V);
grad_a = grad_h .* (1 - (h_mat(:,end).^2));

grad_h_mat(:, end) = grad_h;
grad_a_mat(:, end) = grad_a; 

% Recursively calculate gradients of a_t and h_t from (seq_length - 1) to 1
for ii = (RNN.seq_length - 1):-1:1
    grad_o = g(ii,:); 
    grad_h = transpose(grad_o * RNN.V + grad_a' * RNN.W);
    grad_a = grad_h .* (1 - (h_mat(:,ii + 1).^2));

    grad_h_mat(:, ii) = grad_h;
    grad_a_mat(:, ii) = grad_a; 
end

% Gradient of bias b
grad.b = sum(grad_a_mat, 2); 

% Gradient of weight W
temp_a = reshape(grad_a_mat, RNN.m, 1, RNN.seq_length); 
temp_h = reshape(h_mat(:,1:end-1), 1, RNN.m, RNN.seq_length); 
grad.W = sum(bsxfun(@times, temp_a, temp_h), 3);

% Gradient of weight U
temp_x = reshape(x, 1, RNN.K, RNN.seq_length); 
grad.U = sum(bsxfun(@times, temp_a, temp_x), 3);

% Clip gradients
for f = fieldnames(grad)'
    grad.(f{1}) = max(min(grad.(f{1}), threshold), -threshold); 
end

h_out = h_mat(:, end); 

end

%% Compute Loss Function
function [L, h_mat, p_mat] = ComputeLoss(x, y, RNN, h)
% Function to calculate the forward-pass of the back-prob algorithm 

% Matrix to store all hidden states (h) from t = 0:seq_length
h_mat = zeros(RNN.m, RNN.seq_length + 1);
h_mat(:, 1) = h; 

% Matrix to store all probabilities from t = 1:seq_length
p_mat = zeros(RNN.K, RNN.seq_length);

% Compute loss, probabilities and hidden states for gradient calculations 
% (forward-pass)
for ii = 1:RNN.seq_length
    % Forward-pass 
    [h, p] = RNN.Evaluate(h, x(:, ii));
    
    % Store variables 
    p_mat(:,ii) = p;
    h_mat(:,ii + 1) = h; 
end

% Calculate loss
L = sum(y .* p_mat, 1);
L = -sum(log(L)); 

end

%% Convert to One-Hot Encoding Function
function [one_hot] = ConvertOneHot(chars, K, char_to_ind)
% Function to create one-hot representation

seq_length = numel(chars); 
one_hot = zeros(K, seq_length); 

for ii = 1:seq_length
    int = char_to_ind(chars(ii));
    
    % Create onehot representation
    one_hot(:,ii) = full(ind2vec(int, K)); 
end

end

%% Load Training Data Function
function [data, chars, n] = LoadData(filename)
% Function to read in the training data from the text file. 

fid = fopen(filename, 'r');
data = fscanf(fid, '%c');
fclose(fid);

chars = unique(data);
n = numel(chars);

end

%% Synthesize Text Function
function output = SynthesizeText(RNN, int_to_char, h, x, n)
% First character of the output text
int = find(x == 1, 1); 
output = int_to_char(int);

for j = 1:n
    % Calculate hidden state and probabilities     
    [h, p] = RNN.Evaluate(h, x);   

    % Randomly select a character based on output probabities 
    int = randsample(1:RNN.K, 1, true, p);
    char = int_to_char(int);
    
    % Append random character to end of output text
    output = [output, char];
    
    % Update input vector
    x = full(ind2vec(int, RNN.K)); 
end

end