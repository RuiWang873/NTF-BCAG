clear all
clc

addpath([pwd, '/funs']);
addpath([pwd, '/datasets']);

datasetName = 'MSRC';
load([datasetName, '.mat']);

gt = Y;
num_Cluster = length(unique(gt));               % 聚类数
num_V = length(X);                              % 视图个数
num_N = size(X{1},1);                           % 样本点数

%% Data preprocessing

disp('------Data preprocessing------');
tic
for v=1:num_V
    a = max(X{v}(:));
    X{v} = double(X{v}./a);
end
toc



%% 设置锚点数目
% 锚点数要大于类别数
% 超参数

anchorRate = [0.7];p=[0.1];lambda1=[100];lambda2=[10];lambda3=[1];%MSRC


anchorNum = fix(num_N * anchorRate);
for num1 = 1:length(anchorNum)
    % 下面开始进行锚点数目的循环
    fprintf('------Current Anchor number:%d------\n', anchorNum(num1));

    %% 创建记录结果的文件
    dir_name = ['.\result\', datasetName, '\'];
    file_dir = [dir_name, datasetName, '_', int2str(anchorNum(num1)), '_AnchorPoints.csv'];
    if ~exist(file_dir, 'file')
        mkdir(dir_name);
        fid = fopen(file_dir,'w');
        fprintf(fid, 'ACC, NMI, Purity, P, R, F, RI, iter, anchorNum, p, lambda1, lambda2, lambda3\n');
        fclose(fid);
    end
    fid = fopen(file_dir,'a');


    %% 生成初始锚图
    disp('----------Anchor Selection----------');
    tic;
    opt1.style = 1;
    opt1.IterMax = 50;
    opt1.toy = 0;
    [~, B_init] = FastmultiCLR(X, num_Cluster, anchorNum(num1), opt1, 10);
    toc;

    %% 初始化 F G
    B_init_hat = time2frequency(B_init);
    for v = 1:num_V
        F_init_hat{v} = eye(num_N, num_Cluster);
        G_init_hat{v} = B_init_hat{v}' * F_init_hat{v};
    end


    %% 运行算法并获得聚类结果
    for num2 = 1:length(p)
        for num3 = 1:length(lambda1)
            for num4 = 1:length(lambda2)
                for num5 = 1:length(lambda3)
                    tic;
                    [F, iter, history,eta] = myAlgorithm(num_N, num_V, num_Cluster, B_init_hat, F_init_hat, G_init_hat, p(num2), lambda1(num3), lambda2(num4), lambda3(num5));
                    disp('----------Clustering----------');
                    F_sum = F{1};
                    for v = 2:num_V
                        F_sum = F_sum + F{v};
                    end
                    [~, Y_pre] = max(F_sum, [], 2);
                    my_result = ClusteringMeasure1(Y, Y_pre);
                    my_time=toc
                    fprintf('%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %d, %d, %.1f, %.5f, %.5f, %.5f, %f\n', my_result, iter, anchorNum(num1), p(num2), lambda1(num3), lambda2(num4), lambda3(num5),eta);
                    %% 写入文件
                    fprintf(fid, '%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %d, %d, %.1f, %.5f, %.5f, %.5f, %f\n', my_result, iter, anchorNum(num1), p(num2), lambda1(num3), lambda2(num4), lambda3(num5),eta);
                end
            end
        end
    end
end

fclose(fid);

