clc
clear

%% Main
% Training data file
train_file = {'condensed_2017.json', 'condensed_2016.json'};

% RNN Parameters
m = 256;
seq_length = 139;
sig = 0.001;

% Training Parameters
epoch = 10;
epi = 1e-8;
eta = 0.001;
gamma = 0.9;

% Load training data from file
[x_train, chars, n_output] = LoadTweetData(train_file);

% Container maps for one-hot encoding
key = num2cell(chars);
value = num2cell(1:numel(chars));

char_to_ind = containers.Map(key, value);
int_to_char = containers.Map(value, key);

% Construct RNN and  gradients class
RNN_train = RNN_model(m, n_output, seq_length, sig);
RNN_best = RNN_model(m, n_output, seq_length, 0); 

grad = RNN_value(m, n_output);
m_grad = RNN_value(m, n_output);

% Training parameters initialize
itr = 0;
loss_avg = [];
loss_op = []; 

% First synthesized text (no training)
h_prev = zeros(RNN_best.m, 1);
c_prev = zeros(RNN_train.m, 1);

x_0 = ConvertOneHot('.', RNN_best.K, char_to_ind);
text = SynthesizeText(RNN_best, int_to_char, h_prev, c_prev, x_0, 140);
disp(text);

% Training loop
for ii = 1:epoch
    tic
    
    % Randomly shuffle tweets 
    x_train = x_train(randperm(numel(x_train)));
    
    for k = 1:numel(x_train)
        % Reset previous hidden state (h_0) to zeros
        h_prev = zeros(RNN_train.m, 1);
        c_prev = zeros(RNN_train.m, 1);

        for e = 1:seq_length:(length(x_train{k}) - seq_length - 1)
            % Sample labelled training data
            x_seq = x_train{k}(e:(e + seq_length - 1));
            y_seq = x_train{k}((e + 1):(e + seq_length));

            % Convert to one-hot encoding
            x = ConvertOneHot(x_seq, RNN_train.K, char_to_ind);
            y = ConvertOneHot(y_seq, RNN_train.K, char_to_ind);     

            % Compute gradients
            [loss, h_prev, c_prev] = ComputeGradient(RNN_train, grad, x, y, h_prev, c_prev);

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

            if ~mod(itr, 5000) 
                text = SynthesizeText(RNN_train, int_to_char, h_prev, c_prev, x(:,1), 140);
                disp(text);
            end
        end
    end
    toc
end

% Create tweet passage from best model
h_prev = zeros(RNN_best.m, 1);
c_prev = zeros(RNN_train.m, 1);
x_0 = ConvertOneHot('.', RNN_best.K, char_to_ind);

text = SynthesizeText(RNN_best, int_to_char, h_prev, c_prev, x_0, 140);
disp(text);

%% Compute Gradient Function
function [L, h_out, c_out] = ComputeGradient(RNN, grad, x, y, h, c)
% Compute the gradients of the bias and weight matrix with respect to the
% loss

[L, h_mat, p_mat, c_mat, c_hat_mat, o_mat, i_mat, f_mat] = ...
    ComputeLoss(x, y, RNN, h, c);

% Threshold for the max/min value of the gradients
threshold = 5;

grad_h_mat = zeros(RNN.m, RNN.seq_length);
grad_c_mat = zeros(RNN.m, RNN.seq_length);
grad_o_mat = zeros(RNN.m, RNN.seq_length);
grad_f_mat = zeros(RNN.m, RNN.seq_length);
grad_i_mat = zeros(RNN.m, RNN.seq_length);
grad_c_hat_mat = zeros(RNN.m, RNN.seq_length);

% Gradient of output (o) with respect to loss (dL/do)
g = -(y - p_mat)';

% Gradient of bias c
grad.c = transpose(sum(g, 1)); 

% Gradient of weight V (dL/dV)
temp_g = reshape(g', RNN.K, 1, RNN.seq_length);
temp_h = reshape(h_mat(:,2:end), 1, RNN.m, RNN.seq_length);
grad.V = sum(bsxfun(@times, temp_g, temp_h), 3);

% Calculate dJ/dy
grad_y = g(end,:); 

% Calculate dJ/dh
grad_h = transpose(grad_y * RNN.V);                 

% Calculate dJ/do 
grad_o = grad_h .* tanh(c_mat(:,end)); 
grad_o = grad_o .* (1 - o_mat(:,end)) .* o_mat(:,end);

% Calculate dJ/dc
grad_c = grad_h .* o_mat(:,end) .* (1 - (tanh(c_mat(:,end))).^2);

% Calculate dJ/di
grad_i = grad_c .* c_hat_mat(:,end);
grad_i = grad_i .* (1 - i_mat(:,end)) .* i_mat(:,end);

% Calculate dJ/df
grad_f = grad_c .* c_mat(:,end-1);
grad_f = grad_f .* (1 - f_mat(:,end)) .* f_mat(:,end);

% Calculate dJ/dc_hat
grad_c_hat = grad_c .* i_mat(:,end);
grad_c_hat = grad_c_hat .* (1 - c_hat_mat(:,end).^2);

grad_h_mat(:, end) = grad_h;
grad_o_mat(:, end) = grad_o; 
grad_c_mat(:, end) = grad_c;
grad_i_mat(:, end) = grad_i;
grad_f_mat(:, end) = grad_f;
grad_c_hat_mat(:, end) = grad_c_hat;

% Recursively calculate gradients from (seq_length - 1) to 1
for ii = (RNN.seq_length - 1):-1:1
    % dJ/dy
    grad_y = g(ii,:); 
    
    % Calculate dJ/dh
    grad_h = transpose(grad_y * RNN.V + grad_f' * RNN.W_f + grad_i' * RNN.W_i ...
        + grad_c_hat' * RNN.W_c + grad_o' * RNN.W_o);

    % Calculate dJ/do 
    grad_o = grad_h .* tanh(c_mat(:,ii+1));
    grad_o = grad_o .* o_mat(:,ii) .* (1 - o_mat(:,ii)) ;
    
    % Calculate dJ/dc
    grad_c = grad_c .* f_mat(:,ii+1); 
    grad_c = grad_c + grad_h .* o_mat(:,ii) .* (1 - (tanh(c_mat(:,ii+1))).^2);
    
    % Calculate dJ/di
    grad_i = grad_c .* c_hat_mat(:,ii);
    grad_i = grad_i .* i_mat(:,ii) .* (1 - i_mat(:,ii));

    % Calculate dJ/df
    grad_f = grad_c .* c_mat(:,ii);
    grad_f = grad_f .* f_mat(:,ii) .* (1 - f_mat(:,ii));

    % Calculate dJ/dc_hat
    grad_c_hat = grad_c .* i_mat(:,ii);
    grad_c_hat = grad_c_hat .* (1 - c_hat_mat(:,ii).^2);
    
    grad_h_mat(:, ii) = grad_h;
    grad_o_mat(:, ii) = grad_o; 
    grad_c_mat(:, ii) = grad_c;
    grad_i_mat(:, ii) = grad_i;
    grad_f_mat(:, ii) = grad_f;
    grad_c_hat_mat(:, ii) = grad_c_hat;
end

% Gradient of bias b
grad.b_f = sum(grad_f_mat, 2); 
grad.b_i = sum(grad_i_mat, 2); 
grad.b_o = sum(grad_o_mat, 2); 
grad.b_c = sum(grad_c_hat_mat, 2); 

temp_h = reshape(h_mat(:,1:end-1), 1, RNN.m, RNN.seq_length); 
temp_x = reshape(x, 1, RNN.K, RNN.seq_length); 

% Gradient of weight W and weight U of forget node
temp_t = reshape(grad_f_mat, RNN.m, 1, RNN.seq_length); 
grad.W_f = sum(bsxfun(@times, temp_t, temp_h), 3);
grad.U_f = sum(bsxfun(@times, temp_t, temp_x), 3);

% Gradient of weight W and weight U of input node
temp_t = reshape(grad_i_mat, RNN.m, 1, RNN.seq_length); 
grad.W_i = sum(bsxfun(@times, temp_t, temp_h), 3);
grad.U_i = sum(bsxfun(@times, temp_t, temp_x), 3);

% Gradient of weight W and weight U of output node
temp_t = reshape(grad_o_mat, RNN.m, 1, RNN.seq_length); 
grad.W_o = sum(bsxfun(@times, temp_t, temp_h), 3);
grad.U_o = sum(bsxfun(@times, temp_t, temp_x), 3);

% Gradient of weight W and weight U of c_hat node
temp_t = reshape(grad_c_hat_mat, RNN.m, 1, RNN.seq_length);
grad.W_c = sum(bsxfun(@times, temp_t, temp_h), 3);
grad.U_c = sum(bsxfun(@times, temp_t, temp_x), 3);

% Clip gradients
for f = fieldnames(grad)'
    grad.(f{1}) = max(min(grad.(f{1}), threshold), -threshold); 
end

h_out = h_mat(:, end); 
c_out = c_mat(:, end); 

end

%% Compute Loss Function
function [L, h_mat, p_mat, c_mat, c_hat_mat, o_mat, i_mat, f_mat] = ...
    ComputeLoss(x, y, RNN, h, c)
% Function to calculate the forward-pass of the back-prob algorithm 

% Matrix to store all hidden states (h) and cell states (c) from t = 0:seq_length
h_mat = zeros(RNN.m, RNN.seq_length + 1);
h_mat(:, 1) = h; 

c_mat = zeros(RNN.m, RNN.seq_length + 1);
c_mat(:, 1) = c; 

% Matrix to store all LSTM states
c_hat_mat = zeros(RNN.m, RNN.seq_length);
o_mat = zeros(RNN.m, RNN.seq_length);
i_mat = zeros(RNN.m, RNN.seq_length);
f_mat = zeros(RNN.m, RNN.seq_length);

% Matrix to store all probabilities from t = 1:seq_length
p_mat = zeros(RNN.K, RNN.seq_length);

% Compute loss, probabilities and hidden states for gradient calculations 
% (forward-pass)
for ii = 1:RNN.seq_length
    % Forward-pass 
    [h, c, p, c_hat, o, i, f] = RNN.Evaluate(h, x(:, ii), c);
 
    % Store variables 
    p_mat(:,ii) = p;
    h_mat(:,ii + 1) = h;
    c_mat(:,ii + 1) = c; 
    c_hat_mat(:,ii) = c_hat;
    o_mat(:,ii) = o;
    i_mat(:,ii) = i;
    f_mat(:,ii) = f;
end

% Calculate loss
L = sum(y .* p_mat, 1);
L = -sum(log(L)); 

end

%% Convert One-Hot Encoding Function
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
function [data, chars, n] = LoadTweetData(filename)
% Function to read in the training data from the text file. 
tweet_len = 140;
data = [];

for k = 1:numel(filename)
    tweet = loadjson(filename{k});
    
    batch_data = cell(1, numel(tweet));

    for ii = 1:numel(tweet)
        % Check for twitter http link
        idx = strfind(tweet{ii}.text,'https://t.co');
        
        % Remove twitter http link
        if isempty(idx)
            temp = tweet{ii}.text;
        else
            temp = tweet{ii}.text(1:idx-1);
        end
        
        length = numel(temp);

        if length >= 140
            % Add end-of-tweet character
            batch_data{ii} = [temp(1:140), char(0)];
        else
            % Add end-of-tweet character and padding
            batch_data{ii} = [temp, repmat(char(0), [1, tweet_len + 1 - length])];
        end
    end
    
    % Combine all twitter data 
    data = [data, batch_data];
end

chars = unique(strjoin(data));
n = numel(chars); 

end

%% Synthesize Text Function
function output = SynthesizeText(RNN, int_to_char, h, c, x, n)
% Function to synthesize text with the RNN
output = [];

for j = 1:n
    % Calculate hidden state and probabilities     
    [h, c, p] = RNN.Evaluate(h, x, c);   

    % Randomly select a character based on output probabities 
    int = randsample(1:RNN.K, 1, true, p);
    char = int_to_char(int);
    
    % Append random character to end of output text
    output = [output, char];
    
    % Update input vector
    x = full(ind2vec(int, RNN.K)); 
end

end

