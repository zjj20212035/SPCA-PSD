function [id,OBJ] = CSPCA_PSD(X,alpha,beta)
% Convex Sparse Principal Component Analysis with PSD constraint
% X: data matrix, each column is a sample
% alpha: the regularization parameter for l2,1-norm
% beta: the regularization parameter for nuclear norm
% id: The rank of features
% OBJ: The objective function value

[d,n] = size(X);
W = diag(rand(1,d));
X = X - repmat(mean(X,2),1,n);
E = (W.'*X-X)';
D1 = diag(1./(2*sqrt(sum(E.^2,2)+0.001)));
D2 = diag(1./(2*sqrt(sum(W.^2,2)+0.001)));

OBJ = zeros(1,100);
delta = 1;
t = 1;
I = eye(d);
Niter = 100;

while delta > 10^-5
    if t==1
        nuclear_norm = sum(diag(W));  
        obj = norm(sqrt(sum(E.^2,2)),1) + alpha * norm(sqrt(sum(W.^2,2)),1) + beta * nuclear_norm;
        OBJ(1) = obj;
    end
    W = (X*D1*X' + alpha*D2 + 0.001*I)\(X*D1*X'-beta*I);
    W = Keep_PSD(W);
    E = (W.'*X-X)';
    D1 = diag(1./(2*sqrt(sum(E.^2,2)+0.001)));
    D2 = diag(1./(2*sqrt(sum(W.^2,2)+0.001)));
    
    nuclear_norm = sum(diag(W));
    obj1 = norm(sqrt(sum(E.^2,2)),1) + alpha * norm(sqrt(sum(W.^2,2)),1) + beta *  nuclear_norm;
    delta = abs(obj1-obj);
    OBJ(t+1) = obj1;
    obj = obj1;
    t = t+1;
    if t > Niter
        break
    end

end

sqW = (W.^2);
sumW = sum(sqW);
[~,id] = sort(sumW,'descend');
end

