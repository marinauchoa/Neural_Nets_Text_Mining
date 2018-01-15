clear all
close all
% Load data
load('IANN_project_data_PC7.mat')

% Compute maximum number of neurons in hidden layer
% "number of weights should be no more than 1/30 of the number of training
% cases"
% (http://www.faqs.org/faqs/ai-faq/neural-nets/part3/section-10.html)
num_hidd_l = ceil((max_ind_train/30)/(2*size(input_train, 1)+6));

% Get the vector of 4 sizes of hidden layer to try out
iterate_over = [floor(num_hidd_l/4) floor(2*num_hidd_l/4) floor(3*num_hidd_l/4) num_hidd_l];

% Generate neural networks with varying transfer function and hidden layer
% size
% Save the hidden layer size, transfer function and percent errors to a
% data frame
net_errors = cell2table(cell(0,3), 'VariableNames', {'HL_size' 'transferFcn' 'percentErrors'});
for i=iterate_over
    hiddenLayerSize = i;
    for j={'elliotsig' 'logsig' 'radbas' 'tansig'}
        IANN_project_neural_network;
        net_errors = vertcat(net_errors, [i j percentErrors]);
    end
end

