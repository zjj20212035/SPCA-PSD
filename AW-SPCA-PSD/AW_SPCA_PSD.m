function [ id,OBJ ] = AW_SPCA_PSD( X,lambda )
% Adaptive Weighted Sparse Principal Component Analysis with PSD constraint
% X: data matrix, each column is a sample
% lambda: the regularization parameter for l2,1-norm
% id: The rank of features
% OBJ: The objective function value

[m,n] = size(X);
X = X - repmat(mean(X,2),1,n);
v = zeros( m , 1 );
one = ones( n , 1);
W1 = eye(n);
W2 = eye(m);
E1 = X * W1 * X';
I = eye(m);

delta = inf;
k=1;
OBJ = zeros(1,50);
DELTA = zeros(1,100);
Niter = 100;

while delta > 10^-5
    A = ( E1 - v * one' * W1 * X') / ( E1 + lambda * W2 + 0.001*I);
    A = Keep_PSD(A);
    v = ( X * W1 - A * X * W1 ) * one / ( one' * W1 * one );

    W1 = diag( 1./( 2 * sqrt( sum(( X - A*X - v*one' ).^2 ) + 0.001) ) );
    W2 = diag( 1./( 2 * sqrt( sum( A .^2 ) + 0.001) ) );
    E1 = X*W1*X';
    obj = norm( sqrt( sum(( X - A*X - v*one' ).^2 ) ), 1) + lambda * norm( sqrt( sum( A.^2 ) ), 1);
    if k == 1
        OBJ(1) = obj;
    else
        OBJ(k) = obj;
        delta = abs(OBJ(k) - OBJ(k-1));
        DELTA(k-1) = delta;
    end
    
    if k > Niter
        break
    end
    
    k = k+1;
end
 
sqA = (A.^2);
sumA = sum(sqA);
[~,id] = sort(sumA,'descend');


end


