function [Y_cell] = time2frequency(X_cell)
% Input:
%       - X_cell: tensor of size x1 * x2 * v (cell), where v means the v-th view, x1 and x2 are
%               numbers that varie with the input tensor;
% Output:
%       - Y_cell: frequency domain, size x1 * x2 * v (cell);
%
%   Written by Jing Li

X_tensor = cat(3, X_cell{:,:});  % x1 * x2 * v 
% Y = shiftdim(X_tensor, 1);    % Y is  x2 * v * x1 
Y = X_tensor;
Y_hat = fft(Y, [], 3);
% Y_shift = shiftdim(Y_hat, 2);  % Y_shift is  x1 * x2 * v
Y_shift = Y_hat;
[~, ~, num_V] = size(Y_shift);
for v = 1:num_V
        Y_cell{v} = Y_shift(:,:,v);
end

end

