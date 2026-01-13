% Input : X (cell)
% Output: ||X`||_Sp^p,
%        X` denotes the rotation of X (model 3)
% Author: Luhan

function [obj] = cal_tensorSp(X,p)
%X`is mode3，shiftdim(X, 1);

for v = 1:length(X)
    XX(v,:,:) = X{v};
end
Y=shiftdim(XX, 1);
Yhat = fft(Y,[],3);
obj = 0;
for v = 1: size(X,1)
    [~,shat,~] = svd(full(Yhat(:,:,v)),'econ');
    shat=diag(shat);
    obj = obj + sum(shat.^p);
end
end

