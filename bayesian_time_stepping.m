clear variables
% close all
clc

c=-1;

g_function = @(x) sin(5 * pi*x);

h_function = @(x, t) 0;
f_function = @(x, t) 0;
%mean_function = @(x, t) exp(c * pi^2 * t) .* sin(pi * x);


% Pairwise distance function without Statistics Toolbox
pairwise_dist = @(x, y) abs(x(:) - y(:)'); 


% Example parameters
sigma_1 = 3; rho_1 = 0.2;
sigma_2 = 3; rho_2 = 0.2;

Matern_32 = @(t1, t2) ...
    sigma_1^2 * (1 + sqrt(3) * pairwise_dist(t1, t2) / rho_1) ...
    .* exp(-sqrt(3) * pairwise_dist(t1, t2) / rho_1);

Matern_52 = @(x1, x2) ...
    sigma_2^2 * (1 + sqrt(5) * pairwise_dist(x1, x2) / rho_2 + ...
    5 * pairwise_dist(x1, x2).^2 / (3 * rho_2^2)) ...
    .* exp(-sqrt(5) * pairwise_dist(x1, x2) / rho_2);

product_kernel = @(x1, t1, x2, t2) Matern_52(x1, x2) .* Matern_32(t1, t2);

mean_function = @(x, t) exp( -1 * c * ( 5 * pi)^2 * t) .* sin(pi * x);
%mean_function = @(x,t) 0;


% Setup
num_x = 30;
num_t = 6;
x_vec = linspace(0, 1, num_x)';
t_vec = linspace(0, 1, num_t)';
dt = 1/(num_t-1);

[X_grid, T_grid] = meshgrid(x_vec, t_vec);
x_input = X_grid(:);
t_input = T_grid(:);


x_vec_inner = x_vec(2:end-1);
a_0 = x_vec_inner;
a_0_time = zeros(length(a_0), 1);

N = length(x_input);


%%
%initialization
% Posterior mean function
mu_functions = cell(num_t, 1);
mu_functions{1} = @(x, t) ...
    mean_function(x, t) + ...
    product_kernel(x, t, a_0, a_0_time) * ...
    (product_kernel(a_0, a_0_time, a_0, a_0_time) \ ...
    (g_function(a_0) - mean_function(a_0, a_0_time)));

sigma_functions = cell(num_t, 1);
sigma_functions{1} = @(x1, t1, x2, t2) ...
    product_kernel(x1, t1, x2, t2) - ...
    product_kernel(x1, t1, a_0, a_0_time) * ...
    (product_kernel(a_0, a_0_time, a_0, a_0_time) \ ...
     product_kernel(a_0, a_0_time, x2, t2));


% Evaluate it safely


%%
%time stepping



h_size = 0.01;
b = [0; 1];

mu_j_Dj = cell(num_t, 1);
sigma_j_D_j = cell(num_t, 1);
sigma_j_D_j_bar = cell(num_t, 1);
sigma_j_D_D_j = cell(num_t,1);
b_time = cell(num_t, 1);
v_time = cell(num_t, 1);
A_j = cell(num_t, 1);
after_A_j = cell(num_t, 1);


% Recursive construction
for i = 2:num_t
    prev_mu = mu_functions{i-1};  % previous function handle
    prev_sigma = sigma_functions{i-1};  % previous function handle

    % First derivative of mu with respect to t and second derivative in x
    mu_j_Dj{i} = @(x, t) ...
        (prev_mu(x, t + h_size) - prev_mu(x, t)) / h_size + ...
        c * (prev_mu(x + h_size, t) - 2 * prev_mu(x, t) + prev_mu(x - h_size, t)) / (h_size^2);
    
    % First derivative of sigma with respect to t1 and second derivative in x1
    sigma_j_D_j{i} = @(x1, t1, x2, t2) ...
        (prev_sigma(x1, t1 + h_size, x2, t2) - prev_sigma(x1, t1, x2, t2)) / h_size + ...
        c * (prev_sigma(x1 + h_size, t1, x2, t2) - 2 * prev_sigma(x1, t1, x2, t2) + ...
             prev_sigma(x1 - h_size, t1, x2, t2)) / (h_size^2);
    
    % First derivative of sigma with respect to t2 and second derivative in x2
    sigma_j_D_j_bar{i} = @(x1, t1, x2, t2) ...
        (prev_sigma(x1, t1, x2, t2 + h_size) - prev_sigma(x1, t1, x2, t2)) / h_size + ...
        c * (prev_sigma(x1, t1, x2 + h_size, t2) - 2 * prev_sigma(x1, t1, x2, t2) + ...
             prev_sigma(x1, t1, x2 - h_size, t2)) / (h_size^2);
    
    % Mixed derivative: derivative of sigma_i_D_i w.r.t t2 and second derivative in x2
    sigma_j_D_D_j{i} = @(x1, t1, x2, t2) ...
        (sigma_j_D_j{i}(x1, t1, x2, t2 + h_size) - sigma_j_D_j{i}(x1, t1, x2, t2)) / h_size + ...
        c * (sigma_j_D_j{i}(x1, t1, x2 + h_size, t2) - 2 * sigma_j_D_j{i}(x1, t1, x2, t2) + ...
             sigma_j_D_j{i}(x1, t1, x2 - h_size, t2)) / (h_size^2);
    
    
    b_time{i} = (i - 1) * dt * ones(2, 1);
    v_time{i} = (i - 1) * dt * ones(num_x, 1);
    
    
   % Construct A_i
A_j{i} = [ ...
    prev_sigma(b, b_time{i}, b, b_time{i}),          sigma_j_D_j_bar{i}(b, b_time{i}, x_vec, v_time{i}); 
    sigma_j_D_j{i}(x_vec, v_time{i}, b, b_time{i}),     sigma_j_D_D_j{i}(x_vec, v_time{i}, x_vec, v_time{i}) ...
];

% Construct right-hand side
after_A_j{i} = [ ...
    h_function(b, b_time{i}) - prev_mu(b, b_time{i}); 
    f_function(x_vec, v_time{i}) - mu_j_Dj{i}(x_vec, v_time{i}) ...
];

% Mu function for current iteration
mu_functions{i} = @(x, t) prev_mu(x, t) + ...
    [prev_sigma(x, t, b, b_time{i}), sigma_j_D_j_bar{i}(x, t, x_vec, v_time{i})] * (A_j{i} \ after_A_j{i});

% Sigma function for current iteration
sigma_functions{i} = @(x1, t1, x2, t2) prev_sigma(x1, t1, x2, t2) - ...
    [prev_sigma(x1, t1, b, b_time{i}), sigma_j_D_j_bar{i}(x1, t1, x_vec, v_time{i})] * ...
    (A_j{i} \ [prev_sigma(b, b_time{i}, x2, t2); sigma_j_D_j{i}(x_vec, v_time{i}, x2, t2)]);
end



%
%sample
%


mu_0 = mu_functions{1}(x_input, t_input);
sigma_0 = sigma_functions{1}(x_input, t_input, x_input, t_input);


% Add a small amount of jitter for numerical stability
sigma_0 = sigma_0 + 1e-6 * eye(size(sigma_0));

% 4. Generate Samples
L = chol(sigma_0, 'lower'); % Cholesky decomposition
z = randn(N, 1);       % Standard normal random variables


% Generate the GP sample
f = mu_0 + L * z;

% 5. Plot the sample
% Reshape the sampled function values to the grid format for plotting
f_grid_1 = reshape(f, num_t, num_x);


%

mu_0 = mu_functions{2}(x_input, t_input);
sigma_0 = sigma_functions{2}(x_input, t_input, x_input, t_input);


% Add a small amount of jitter for numerical stability
sigma_0 = sigma_0 + 1e-6 * eye(size(sigma_0));

% 4. Generate Samples
L = chol(sigma_0, 'lower'); % Cholesky decomposition
z = randn(N, 1);       % Standard normal random variables


% Generate the GP sample
f = mu_0 + L * z;

% 5. Plot the sample
% Reshape the sampled function values to the grid format for plotting
f_grid_2 = reshape(f, num_t, num_x);




%

mu_0 = mu_functions{3}(x_input, t_input);
sigma_0 = sigma_functions{3}(x_input, t_input, x_input, t_input);


% Add a small amount of jitter for numerical stability
sigma_0 = sigma_0 + 1e-6 * eye(size(sigma_0));

% 4. Generate Samples
L = chol(sigma_0, 'lower'); % Cholesky decomposition
z = randn(N, 1);       % Standard normal random variables


% Generate the GP sample
f = mu_0 + L * z;

% 5. Plot the sample
% Reshape the sampled function values to the grid format for plotting
f_grid_3 = reshape(f, num_t, num_x);





%

mu_0 = mu_functions{4}(x_input, t_input);
sigma_0 = sigma_functions{4}(x_input, t_input, x_input, t_input);


% Add a small amount of jitter for numerical stability
sigma_0 = sigma_0 + 1e-6 * eye(size(sigma_0));

% 4. Generate Samples
L = chol(sigma_0, 'lower'); % Cholesky decomposition
z = randn(N, 1);       % Standard normal random variables


% Generate the GP sample
f = mu_0 + L * z;

% 5. Plot the sample
% Reshape the sampled function values to the grid format for plotting
f_grid_4 = reshape(f, num_t, num_x);



%

mu_0 = mu_functions{5}(x_input, t_input);
sigma_0 = sigma_functions{5}(x_input, t_input, x_input, t_input);


% Add a small amount of jitter for numerical stability
sigma_0 = sigma_0 + 1e-6 * eye(size(sigma_0));

% 4. Generate Samples
L = chol(sigma_0, 'lower'); % Cholesky decomposition
z = randn(N, 1);       % Standard normal random variables


% Generate the GP sample
f = mu_0 + L * z;

% 5. Plot the sample
% Reshape the sampled function values to the grid format for plotting
f_grid_5 = reshape(f, num_t, num_x);




%

mu_0 = mu_functions{6}(x_input, t_input);
sigma_0 = sigma_functions{6}(x_input, t_input, x_input, t_input);


% Add a small amount of jitter for numerical stability
sigma_0 = sigma_0 + 1e-6 * eye(size(sigma_0));

% 4. Generate Samples
L = chol(sigma_0, 'lower'); % Cholesky decomposition
z = randn(N, 1);       % Standard normal random variables


% Generate the GP sample
f = mu_0 + L * z;

% 5. Plot the sample
% Reshape the sampled function values to the grid format for plotting
f_grid_6 = reshape(f, num_t, num_x);



%{

mu_0 = mu_functions{7}(x_input, t_input);
sigma_0 = sigma_functions{7}(x_input, t_input, x_input, t_input);


% Add a small amount of jitter for numerical stability
sigma_0 = sigma_0 + 1e-6 * eye(size(sigma_0));

% 4. Generate Samples
L = chol(sigma_0, 'lower'); % Cholesky decomposition
z = randn(N, 1);       % Standard normal random variables


% Generate the GP sample
f = mu_0 + L * z;

% 5. Plot the sample
% Reshape the sampled function values to the grid format for plotting
f_grid_7 = reshape(f, num_t, num_x);



%

mu_0 = mu_functions{8}(x_input, t_input);
sigma_0 = sigma_functions{8}(x_input, t_input, x_input, t_input);


% Add a small amount of jitter for numerical stability
sigma_0 = sigma_0 + 1e-6 * eye(size(sigma_0));

% 4. Generate Samples
L = chol(sigma_0, 'lower'); % Cholesky decomposition
z = randn(N, 1);       % Standard normal random variables


% Generate the GP sample
f = mu_0 + L * z;

% 5. Plot the sample
% Reshape the sampled function values to the grid format for plotting
f_grid_8 = reshape(f, num_t, num_x);


%

mu_0 = mu_functions{9}(x_input, t_input);
sigma_0 = sigma_functions{9}(x_input, t_input, x_input, t_input);


% Add a small amount of jitter for numerical stability
sigma_0 = sigma_0 + 1e-6 * eye(size(sigma_0));

% 4. Generate Samples
L = chol(sigma_0, 'lower'); % Cholesky decomposition
z = randn(N, 1);       % Standard normal random variables


% Generate the GP sample
f = mu_0 + L * z;

% 5. Plot the sample
% Reshape the sampled function values to the grid format for plotting
f_grid_9 = reshape(f, num_t, num_x);



%

mu_0 = mu_functions{10}(x_input, t_input);
sigma_0 = sigma_functions{10}(x_input, t_input, x_input, t_input);


% Add a small amount of jitter for numerical stability
sigma_0 = sigma_0 + 1e-6 * eye(size(sigma_0));

% 4. Generate Samples
L = chol(sigma_0, 'lower'); % Cholesky decomposition
z = randn(N, 1);       % Standard normal random variables


% Generate the GP sample
f = mu_0 + L * z;

% 5. Plot the sample
% Reshape the sampled function values to the grid format for plotting
f_grid_10 = reshape(f, num_t, num_x);

%}

save('results.mat');
