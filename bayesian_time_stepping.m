clear variables
% close all
%clear
clc

c=-1;

%load('myData.mat')

g_function = @(x) sin(pi*x);

h_function = @(x, t) 0;
f_function = @(x, t) 0;


dist_eu = @(x, y) abs(x(:) - y(:)'); 



sigma_1 = 3; rho_1 = 0.2;
sigma_2 = 3; rho_2 = 0.2;

Matern_32 = @(t1, t2) ...
    sigma_1^2 * (1 + sqrt(3) * dist_eu(t1, t2) / rho_1) ...
    .* exp(-sqrt(3) * dist_eu(t1, t2) / rho_1);

Matern_52 = @(x1, x2) ...
    sigma_2^2 * (1 + sqrt(5) * dist_eu(x1, x2) / rho_2 + ...
    5 * dist_eu(x1, x2).^2 / (3 * rho_2^2)) ...
    .* exp(-sqrt(5) * dist_eu(x1, x2) / rho_2);

init_kernel = @(x1, t1, x2, t2) Matern_52(x1, x2) .* Matern_32(t1, t2);

%mean_function = @(x,t) exp(pi^2 * t) .* sin(pi * x);
mean_function = @(x,t) 0;


N_x = 30;
N_t = 6;
x_line = linspace(0, 1, N_x)';
t_line = linspace(0, 1, N_t)';
dt = 1/(N_t-1);

[X_grid, T_grid] = meshgrid(x_line, t_line);
x_input = X_grid(:);
t_input = T_grid(:);


x_line_inner = x_line(2:end-1);
a_0 = x_line_inner;
a_0_time = zeros(length(a_0), 1);

N = length(x_input);


%%
%initialization
mu_functions = cell(N_t, 1);
mu_functions{1} = @(x, t) ...
    mean_function(x, t) + ...
    init_kernel(x, t, a_0, a_0_time) * ...
    (init_kernel(a_0, a_0_time, a_0, a_0_time) \ ...
    (g_function(a_0) - mean_function(a_0, a_0_time)));

sigma_functions = cell(N_t, 1);
sigma_functions{1} = @(x1, t1, x2, t2) ...
    init_kernel(x1, t1, x2, t2) - ...
    init_kernel(x1, t1, a_0, a_0_time) * ...
    (init_kernel(a_0, a_0_time, a_0, a_0_time) \ ...
     init_kernel(a_0, a_0_time, x2, t2));





%%
%time stepping



h_size = 0.01;
b = [0; 1];

mu_j_Dj = cell(N_t, 1);
sigma_j_D_j = cell(N_t, 1);
sigma_j_D_j_bar = cell(N_t, 1);
sigma_j_D_D_j = cell(N_t,1);
b_time = cell(N_t, 1);
v_time = cell(N_t, 1);
A_j = cell(N_t, 1);
after_A_j = cell(N_t, 1);



for i = 2:N_t
    mu_prev = mu_functions{i-1};  
    sigma_prev = sigma_functions{i-1};  

    
    mu_j_Dj{i} = @(x, t) ...
        (mu_prev(x, t + h_size) - mu_prev(x, t)) / h_size + ...
        c * (mu_prev(x + h_size, t) - 2 * mu_prev(x, t) + mu_prev(x - h_size, t)) / (h_size^2);
    
   
    sigma_j_D_j{i} = @(x1, t1, x2, t2) ...
        (sigma_prev(x1, t1 + h_size, x2, t2) - sigma_prev(x1, t1, x2, t2)) / h_size + ...
        c * (sigma_prev(x1 + h_size, t1, x2, t2) - 2 * sigma_prev(x1, t1, x2, t2) + ...
             sigma_prev(x1 - h_size, t1, x2, t2)) / (h_size^2);
    
   
    sigma_j_D_j_bar{i} = @(x1, t1, x2, t2) ...
        (sigma_prev(x1, t1, x2, t2 + h_size) - sigma_prev(x1, t1, x2, t2)) / h_size + ...
        c * (sigma_prev(x1, t1, x2 + h_size, t2) - 2 * sigma_prev(x1, t1, x2, t2) + ...
             sigma_prev(x1, t1, x2 - h_size, t2)) / (h_size^2);
    
    
    sigma_j_D_D_j{i} = @(x1, t1, x2, t2) ...
        (sigma_j_D_j{i}(x1, t1, x2, t2 + h_size) - sigma_j_D_j{i}(x1, t1, x2, t2)) / h_size + ...
        c * (sigma_j_D_j{i}(x1, t1, x2 + h_size, t2) - 2 * sigma_j_D_j{i}(x1, t1, x2, t2) + ...
             sigma_j_D_j{i}(x1, t1, x2 - h_size, t2)) / (h_size^2);
    
    
    b_time{i} = (i - 1) * dt * ones(2, 1);
    v_time{i} = (i - 1) * dt * ones(N_x, 1);
    
    
      
    A_j{i} = [ ...
        sigma_prev(b, b_time{i}, b, b_time{i}),          sigma_j_D_j_bar{i}(b, b_time{i}, x_line, v_time{i}); 
        sigma_j_D_j{i}(x_line, v_time{i}, b, b_time{i}),     sigma_j_D_D_j{i}(x_line, v_time{i}, x_line, v_time{i}) ...
    ];

    
    after_A_j{i} = [ ...
        h_function(b, b_time{i}) - mu_prev(b, b_time{i}); 
        f_function(x_line, v_time{i}) - mu_j_Dj{i}(x_line, v_time{i}) ...
    ];

    
    mu_functions{i} = @(x, t) mu_prev(x, t) + ...
        [sigma_prev(x, t, b, b_time{i}), sigma_j_D_j_bar{i}(x, t, x_line, v_time{i})] * (A_j{i} \ after_A_j{i});
    
    
    sigma_functions{i} = @(x1, t1, x2, t2) sigma_prev(x1, t1, x2, t2) - ...
        [sigma_prev(x1, t1, b, b_time{i}), sigma_j_D_j_bar{i}(x1, t1, x_line, v_time{i})] * ...
        (A_j{i} \ [sigma_prev(b, b_time{i}, x2, t2); sigma_j_D_j{i}(x_line, v_time{i}, x2, t2)]);
end







%%
%sampling

u_sampled = cell(N_t, 1);

for j = 1:N_t
    mu_mat = mu_functions{j}(x_input, t_input);
    sigma_mat = sigma_functions{j}(x_input, t_input, x_input, t_input);
    
   % sigma_sampled = sigma_sampled + 1e-6 * eye(size(sigma_mat));
    
    L = chol(sigma_mat, 'lower'); 
    z = randn(N, 1);      
    
    u_sampled{i} = mu_mat + L * z;
end


the_sampled_u = zeros(N_t, N_x);

for k = 1:N_t
    temp_func = u_sampled{i};
    the_sampled_u(k, :) = temp_func(k,:);
end

save('results.mat');
