%% Neural Network
% Creation of Data

num_samples = 1000; % Number of samples
num_altitudes = 50; % Number of altitude levels

sea_clutter_strength = linspace(0, 10, num_samples)'; % Linearly spaced sea clutter strength values from 0 to 10 with 1000 values

altitude_index = 1:num_altitudes; % Altitude index

% Generate refractive index profile using a simple pattern
refractive_indices = zeros(num_samples, num_altitudes); % Matrix to store refractive indices for different samples and altitudes
for i = 1:num_samples % Loop over each sample (1000)
   refractive_indices(i, :) = 1.5 + ... % Base value of 1.5 as the starting point for refractive indices
   0.05 * sea_clutter_strength(i) + ... % Dependent on sea clutter strength, unique for each sample
   0.1 * sin(0.1 * altitude_index); % Adds a sine wave pattern on altitude
end

inputs = sea_clutter_strength; % Set the input variable as sea clutter strength
% Normalizing Features
%% 
% * Makes values dimensionless and comparable across altitude levels. 
% * Helps in machine learning models where features should have similar scales.

refractive_indices_normalized = (refractive_indices - mean(refractive_indices, 1)) ./ std(refractive_indices, 0, 1); 
norminputs = (inputs - mean(inputs, 1)) ./ std(inputs, 0, 1); %Each column is now standardized to have a mean centred around 0 and a std=1.
% Check the mean and standard deviation of the normalized data
disp('Mean of normalized refractive_indices:');
disp(mean(refractive_indices_normalized, 1));

disp('Standard deviation of normalized refractive_indices:');
disp(std(refractive_indices_normalized, 0, 1));
disp('Mean of normalized inputs:');
disp(mean(norminputs, 1));
disp('Standard deviation of normalized inputs:');
disp(std(norminputs, 0, 1));
% Training and Testing

cv = cvpartition(num_samples, 'HoldOut', 0.2); % Splits data: 80% training, 20% testing
idxTrain = training(cv); % Returns a logical vector, (1) if sample in training set, (0) if sample not in training set
idxTest = test(cv); % Returns a logical vector, (1) if sample in testing set, (0) if sample not in testing set

% 80% of samples
input_train = norminputs(idxTrain, :); % Trains only input samples that are in training set
output_train = refractive_indices_normalized(idxTrain, :); % Trains only output samples that are in training set

% 20% of samples
input_test = norminputs(idxTest, :); % Trains only input samples that are in testing set
output_test = refractive_indices_normalized(idxTest, :);  % Trains only output samples that are in testing set

% Transpose training and testing to have correct format for Neural Network
input_train = input_train';  
output_train = output_train';  
input_test = input_test';    
output_test = output_test';    


hiddenLayerSizes = [20, 20, 20]; % Specifies number of neurons in each layer
net = feedforwardnet(hiddenLayerSizes, 'trainbr'); % Uses Levenberg-Marquardt (LM) backpropagation algorithm as training method 

% Training options
net.trainParam.epochs = 200; % Sets max number of iterations
net.trainParam.goal = 1e-6; % Targets MSE of 1e-8
net.trainParam.min_grad = 1e-7; % Stops if gradiant loss is below 1e-7
net.trainParam.mu = 0.01; % Sets the initial damping factor (µ) for the Levenberg-Marquardt algorithm
net.trainParam.max_fail = 6; % Specifies the maximum number of consecutive validation failures

% Train the network
% net = neural network, tr = structure containing training info
[net, tr] = train(net, input_train, output_train);
% Performance Metrics

output_predict = net(input_test); % Predicts outputs from neural network

% Lower scores for these = better performance
mse_value = mean((output_test - output_predict).^2, 'all'); % Measures the average squared difference between actual and predicted values
rmse_value = sqrt(mse_value); % Square root of MSE to bring the error to the same units as the original values.
mae_value = mean(abs(output_test - output_predict), 'all'); % Measures the average absolute difference between actual and predicted values.

% Compute R² Score
% R² = 1 - Perfect model (predictions match exactly).
% R² = 0 - Model performs no better than the mean.
% R² < 0 - Model performs worse than just predicting the mean.
SS_res = sum((output_test - output_predict).^2, 'all'); % Measures the total squared difference between the actual and predicted values.
SS_tot = sum((output_test - mean(output_test, 2)).^2, 'all'); % Measures the total variance in the actual data.
r2_score = 1 - (SS_res / SS_tot); % Compares how well your model's predictions explain the variance in 'output_test'

% Displays performance metrics
fprintf('Model Performance Metrics:\n');
fprintf('MSE  = %.8f\n', mse_value);
fprintf('RMSE = %.5f\n', rmse_value);
fprintf('MAE  = %.5f\n', mae_value);
fprintf('R² Score = %.5f\n', r2_score);
% Predictions

inputs_test = 0.8;  % Example input prop factor = 0.8, incidence angle = 60
inputs_test_normalized = (inputs_test - mean(inputs, 1)) ./ std(inputs, 0, 1);  % Normalize new input data

output_predict = net(inputs_test_normalized'); % Use trained network to predict refractivity profile
output_predict_rescaled = output_predict .* std(refractive_indices, 0, 1) + mean(refractive_indices, 1); % Ensures predicted output is on same scale

% Displays the predicted refractive index profile
disp('Predicted Refractive Index Profile (Rescaled):');
disp(output_predict_rescaled);
num_elements = numel(output_predict_rescaled); % Calculates how many elements are in the array
disp(['The array contains ', num2str(num_elements), ' elements.']);
% 
% Prediction of Refractivity Profile

given_clutter_strength = 5.0;  % Replace with desired value

% Normalize the given input (same scaling as training)
given_input = ([given_clutter_strength] - mean(inputs, 1)) ./ std(inputs, 0, 1);

% Find the closest matching index in the test set
[~, idx] = min(sum(abs(input_test' - given_input), 2));  % Find closest match

% Get the actual refractive index profile for this test case
actual_ref_profile = output_test(:, idx);

% Predict refractive index profile using the trained network
predicted_ref_profile = net(given_input');

% Denormalize results to original scale
mean_ref = mean(refractive_indices, 1);
std_ref = std(refractive_indices, 0, 1);
actual_ref_profile_denorm = actual_ref_profile .* std_ref' + mean_ref';
predicted_ref_profile_denorm = predicted_ref_profile .* std_ref' + mean_ref';

% --- Display Results ---
fprintf('For Sea Clutter Strength = %.2f', given_clutter_strength);

% Display first 10 altitude points for quick comparison
disp(table((1:15)', actual_ref_profile_denorm(1:15), predicted_ref_profile_denorm(1:15), ...
    'VariableNames', {'Altitude', 'Actual_Refractivity_Profile', 'Predicted_Refractivity_Profile'}));
% --- Plot Actual vs. Predicted Refractive Index Profile ---
figure;
plot(1:length(actual_ref_profile_denorm), actual_ref_profile_denorm, 'b', 'LineWidth', 2); % Actual
hold on;
plot(1:length(predicted_ref_profile_denorm), predicted_ref_profile_denorm, 'r--', 'LineWidth', 2); % Predicted
xlabel('Altitude Index');
ylabel('Refractivity Profile');
title('Actual vs. Predicted Refractive Index Profile');
legend('Actual', 'Predicted');