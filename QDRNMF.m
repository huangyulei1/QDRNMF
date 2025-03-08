function [X_new, U, V, W, F, objsum, score_f, idx, iter] = QDRNMF(X, kk, Ds, Ws, alpha, beta, lambda, NIter, l)
% QDRNMF: Quadratic Dimensionality Reduction Nonnegative Matrix Factorization for Data Representation.
% 
% Input:
%   X       - Data matrix (d*n). Each column vector of data is a sample vector.
%   kk      - The first dimensionality reduction dimension.
%   Ds      - The degree matrix of data space.
%   Ws      - The similarity matrix of data space.
%   alpha   - The balance parameter.
%   beta    - The sparse constraint parameter.
%   lambda   - The graph regularization parameter.
%   NIter   - The maximum number of iterations.
%   l       - The number of classes.
%
% Output:
%   X_new   - The final low-dimensional representation (n*£¨i-1£©).
%--------------------------------------------------------------------------
%    Examples:
%       load('COIL20.mat');
%       X = NormalizeFea(fea); l = length(unique(gnd));
%       NIter=100; kk=l+5; alpha=1e0; beta=1e0; lambda=1e0;
%       options = []; options.k =5; options.NeighborMode = 'KNN';
%       options.WeightMode = 'HeatKernel'; options.Metric = 'Euclidean';
%       options.t = 1e0; Ws = constructW1(X,options); Ds=diag(sum(Ws));
%       [X_new, U, V, W, F, objsum, score_f, idx, iter] = QDRNMF(X, kk, Ds, Ws, alpha, beta, lambda, NIter, l);
%       % The features in X_new is the final low-dimensional representation.
[m,n] = size(X);
[~,XV] = litekmeans(X,kk); 
U = rand(m,kk);
V = rand(n,kk);
F = rand(l,kk);
W = init_W(XV,l);
W = W';
for iter = 1:NIter
    G = sqrt(diag(sum(W'*W,2)));
    W = W*pinv(G);
    F = G*F;
    H = W*W';
	% ===================== update U & Q ========================
    U = U.*(X*V*H')./(U*H*V'*V*H');
    Pi = sqrt(sum(W.*W,2) + eps);
    q = 0.5./Pi;
    Q = diag(q);
	% ===================== update V & W & F========================
	V = V.*(X'*U*H + 2*alpha*V*F'*W' + lambda*Ws*V)./(V*H'*U'*U*H + alpha*V + alpha*V*W*F*F'*W' + lambda*Ds*V);
    W = W.*(alpha*V'*V*F' + 2*U'*X*V*W)./(alpha*V'*V*W*F*F' + beta*Q*W+2*U'*U*W*W'*V'*V*W);
    F = F.*(W'*V'*V)./(W'*V'*V*W*F);
    % ==============================================================
    obj(1,iter) = trace((X-U*H*V')*(X-U*H*V')');
    obj(2,iter) = alpha*(trace((V-V*W*F)*(V-V*W*F)'));
    obj(3,iter) = lambda*trace(V'*(Ds-Ws)*V);
    obj(4,iter) = beta*sum(sqrt(Pi));
    objsum(iter) = sum(obj(:,iter));
    if iter>2
        if abs(objsum(iter)-objsum(iter-1))<0.01
            break
        end
    end
end
score = sum(W.*W,2);
[score_f, idx] = sort(score,'descend');
qq = length(score_f);
% score_f = NormalizeFea(score_f');
% score_f = full(score_f');
for i = 2:qq
        if abs(score_f(1)-score_f(i))>5e-1
            break
        end
end
if abs(score_f(1)-score_f(qq))<5e-1
    i=i+1;
end
if i-1>1
X_new = V(:,idx(1:i-1));
else
X_new = V(:,idx(1:l));
end
str = 'The recommended number of reduced dimensions is ';
str = [str,num2str(i-1)];
disp(str)
end