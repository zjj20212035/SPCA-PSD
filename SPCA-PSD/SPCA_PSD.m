function [ omega,id , obj ] = SPCA_PSD( X , lambda, eta)
% Fast Sparse PCA via Positive Semidefinite Projection for Unsupervised
% Feature selection

% X: data matrix, each column is a sample
% lambda: the regularization parameter for l2,1-norm
% eta: the regularization parameter for nuclear norm

[ d , n ] = size( X );
Id = eye(d);
In = eye(n);
X = X - repmat(mean(X,2),1,n);
S = X*X';
X_T = X';
p1 = 1/lambda;
p2 = eta/(2*lambda) * 10^3;

delta = inf;
omega = diag(rand(1,d));
obj = zeros(1,50);
k = 1;
obj(k) = norm(X-omega*X,'fro')^2 + lambda*sum((sqrt(sum(omega.^2)))) + eta * trace(omega);
Niter = 100;

while delta > 10^-5
    diag_W = 0.5 * (sqrt(sum(omega.^2)+eps)).^(-1);
    W = diag(diag_W);
    iD = diag(1./(lambda*diag_W + 0.01));
    
    if d < n
        omega = ( S - eta / 2 * Id)/( S + lambda * W + 0.001*eye(d) );
    elseif d >= n
        omega = iD * X / (In + X_T *iD*X) * X_T .* repmat(diag(Id + eta/2*iD)',d,1) - eta/2*iD;
    end
    omega = Keep_PSD(omega);
    
    obj(k+1) =norm(X-omega*X,'fro')^2 + lambda*sum((sqrt(sum(omega.^2)))) + eta * trace(omega);
    delta = abs(obj(k+1)-obj(k)); 
    k = k + 1;
    
    if k > Niter
        break
    end

end

sqomega = (omega.^2);
sumomega = sum(sqomega);
[~,id] = sort(sumomega,'descend');

end


