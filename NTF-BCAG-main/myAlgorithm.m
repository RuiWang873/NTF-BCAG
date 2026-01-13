function [Q, iter, history,eta] = myAlgorithm(num_N, num_V, num_C, B_init, F_init, G_init, p, lambda1, lambda2, lambda3)

%% 参数初始化
num_Anchor = size(B_init{1}, 2);
sX1 = [num_N, num_C, num_V];
sX2 = [num_Anchor, num_C, num_V];
Isconverg = 0;
iter = 0;
maxiter=300;
eta = 1.2;
mu = 10e-5;
rho = 10e-5;
sigma = 10e-5;
max_mu = 10e12;
max_rho = 10e12;
max_sigma = 10e12;
betaf = ones(num_V, 1); % 张量的权重

% 初始化矩阵
for v = 1:num_V
    J_hat{v} = zeros(num_N, num_C);             % tensor
    Q_hat{v} = zeros(num_N, num_C);
    F_hat{v} = zeros(num_Anchor, num_C);
    Y1_hat{v} = zeros(num_N, num_C);
    Y2_hat{v} = zeros(num_N, num_C);
    Y3_hat{v} = zeros(num_Anchor, num_C);
end

B_hat = B_init;
H_hat = F_init;
G_hat = G_init;

timeStart = clock;

%% 迭代过程
while(Isconverg == 0)  
    %% update S_hat{v}
    for v = 1:num_V
        W0_hat{v} = (H_hat{v} * G_hat{v}' + lambda3 * B_hat{v}) ./ (1 + lambda3);
    end
    W0 = frequency2time(W0_hat);
    for v = 1:num_V
        BB = W0{v};
        for j = 1:num_N
            temp = BB(j,:);
            S{v}(j,:) = EProjSimplex_new(temp, 1);
        end
    end
    S_hat = time2frequency(S);
    clear W0_hat W0 temp BB;
    
    %% update H_hat{v}
    for v = 1:num_V
        A1_hat{v} = 2 * S_hat{v} * G_hat{v} + mu * Q_hat{v} - Y1_hat{v} + rho * J_hat{v} - Y2_hat{v};
        [nn{v}, ~, vv{v}] = svd(A1_hat{v}, 'econ');
        H_hat{v} = nn{v} * vv{v}';
    end
    H = frequency2time(H_hat);
    clear A1_hat nn vv;
    
    %% update G_hat{v}
    for v = 1:num_V
        W0_hat{v} = (S_hat{v}' * H_hat{v} + (sigma / 2) * F_hat{v} - Y3_hat{v} ./ 2) ./ (1 + sigma / 2);
    end
    W0 = frequency2time(W0_hat);
    for v = 1:num_V
        BB = W0{v};
        for j = 1:num_Anchor
            temp = BB(j,:);
            G{v}(j,:) = EProjSimplex_new(temp, 1);
        end
    end
    G_hat = time2frequency(G);
    clear W0_hat W0 temp BB;
    
    %% update Q_hat{v}
    for v = 1:num_V
        Q_hat{v} = H_hat{v} + Y1_hat{v} ./ mu;
        Q_hat{v}(Q_hat{v}<0) = 0;
    end
    Q = frequency2time(Q_hat);
    
    %% update J_hat{v}
    Y2 = frequency2time(Y2_hat);
    for v = 1:num_V
        temp1{v} = H{v} + Y2{v} ./ rho;
    end
    temp1_tensor = cat(3, temp1{:,:});
    [myj, ~] = wshrinkObj_weight_lp(temp1_tensor(:), lambda1 * betaf./rho, sX1, 0, 3, p);
    J_tensor = reshape(myj, sX1);
    for v = 1:num_V
        J{v} = J_tensor(:,:,v);
    end
    J_hat = time2frequency(J);
    clear temp1 temp1_tensor;
    
    %% update F_hat{v}
    Y3 = frequency2time(Y3_hat);
    for v = 1:num_V
        temp1{v} = G{v} + Y3{v} ./ sigma;
    end
    temp1_tensor = cat(3, temp1{:,:});
    [myj, ~] = wshrinkObj_weight_lp(temp1_tensor(:), lambda2 * betaf./sigma, sX2, 0, 3, p);
    F_tensor = reshape(myj, sX2);
    for v = 1:num_V
        F{v} = F_tensor(:,:,v);
    end
    F_hat = time2frequency(F);
    clear temp1 temp1_tensor;
    
    %% update Y1 Y2 Y3 mu rho sigma
    Y1 = frequency2time(Y1_hat);
    for v = 1:num_V
        Y1{v} = Y1{v} + mu * (H{v} - Q{v});
        Y2{v} = Y2{v} + rho * (H{v} - J{v});
        Y3{v} = Y3{v} + sigma * (G{v} - F{v});
    end
    Y1_hat = time2frequency(Y1);
    Y2_hat = time2frequency(Y2);
    Y3_hat = time2frequency(Y3);

    mu = min(eta * mu, max_mu);
    rho = min(eta * rho, max_rho);
    sigma = min(eta * sigma, max_sigma);
        
    %% 迭代收敛
    epson = 10e-4;
    Isconverg = 1;
    sumNormHQ = 0;
    sumNormHJ = 0;
    sumNormGF = 0;
    for i=1:num_V
        norm_H_Q = norm(H{i}-Q{i}, 'fro');
        norm_H_J = norm(H{i}-J{i}, 'fro');
        norm_G_F = norm(G{i}-F{i}, 'fro');
        sumNormHQ = sumNormHQ + norm_H_Q;
        sumNormHJ = sumNormHJ + norm_H_J;
        sumNormGF = sumNormGF + norm_G_F;
    end
    
    if (sumNormHQ > epson)
        % fprintf('norm_H_Q %7.10f | norm_H_J %7.10f | norm_G_F %7.10f\n', sumNormHQ, sumNormHJ, sumNormGF);
        history.norm_H_Q(iter+1)=sumNormHQ;
        history.norm_H_J(iter+1)=sumNormHJ;
        history.norm_G_F(iter+1)=sumNormGF;
        Isconverg = 0;
    end
    

    %%
    if iter > maxiter
        Isconverg = 1;
    end
    iter = iter + 1;
        
end


timeEnd = clock;
fprintf('Time all:%f s\n', etime(timeEnd, timeStart));
fprintf('max iter:%d\n', iter);
