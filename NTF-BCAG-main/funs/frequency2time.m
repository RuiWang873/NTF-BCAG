function [X_cell] = frequency2time(Y_cell)
% Input:
%       - Y_cell: tensor of size x1 * x2 * v (cell), where v means the v-th view, x1 and x2 are
%               numbers that varie with the input tensor;
% Output:
%       - X_cell: time domain, size x1 * x2 * v (cell);
%
%   Written by Jing Li

Y_tensor = cat(3, Y_cell{:,:});
% X = shiftdim(Y_tensor, 1);
X = Y_tensor;
X_hat = ifft(X, [], 3);
% X_shift = shiftdim(X_hat, 2);
X_shift = X_hat;
[~, ~, num_V] = size(X_shift);
for v = 1:num_V
        X_cell{v} = X_shift(:,:,v);
end

end

