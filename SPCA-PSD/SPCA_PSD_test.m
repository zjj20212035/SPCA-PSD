function [ omega,id,warning, obj ] = SPCA_PSD_test( X , lambda, eta)
% Sparse PCA based on Positive semi-definite constraint for Unsupervised Feature Selection
[ d , n ] = size( X );
Id = eye(d);
In = eye(n);
X = X - repmat(mean(X,2),1,n);
S = X*X';
X_T = X';
[~,Sigma,V] = svd(X');
Sigma = diag(Sigma(1:min(d,n),1:min(d:n)));
p1 = 1/lambda;
p2 = eta/(2*lambda) * 10^3;

delta = inf;
omega = diag(rand(1,d));
obj = zeros(1,50);
k = 1;
obj(k) = norm(X-omega*X,'fro')^2 + lambda*sum((sqrt(sum(omega.^2)))) + eta * trace(omega);

%DELTA = zeros(1,100);
warning = 0;
while k < 2
    diag_W = 0.5 * (sqrt(sum(omega.^2)+eps)).^(-1);
    W = diag(diag_W);
    iD = diag(1./(lambda*diag_W + 0.01));
    
    if d < n
        omega = ( S - eta / 2 * Id)/( S + lambda * W + 0.001*eye(d) );
    elseif d >= n
        omega = iD * X / (In + X_T *iD*X) * X_T .* repmat(diag(Id + eta/2*iD)',d,1) - eta/2*iD;
        
    end
    [omega, EigOmega] = Keep_PSD(omega);
    obj(k+1) =norm(X-omega*X,'fro')^2 + lambda*sum((sqrt(sum(omega.^2)))) + eta * trace(omega);
    k = k+1;
end

rankOmega = min(length(Sigma),length(EigOmega));
[EigOmega,~] = sort(EigOmega,'descend');
EigOmega = EigOmega(1:rankOmega);
Sigma = Sigma(1:rankOmega);
V = V(:,1:rankOmega);

while delta > 10^-5
    EigOmega = Sigma.^2 ./ (Sigma.^2 + lambda./EigOmega);
    omega = V*diag(EigOmega)*V';
    omega = (omega+omega')/2;
    obj(k+1) =norm(X-omega*X,'fro')^2 + lambda*sum((sqrt(sum(omega.^2)))) + eta * trace(omega);
    delta = abs(obj(k+1)-obj(k)); 
    k = k + 1;
    
    if k > 100
        break
    end
%     if k > 50 
%         if delta > 0.01
%            warning = 1;
%            break
%         else
%             break
%         end
%     end
    
end
if warning == 0
    sqomega = (omega.^2);
    sumomega = sum(sqomega);
    [~,id] = sort(sumomega,'descend');
else
    id = [];
end
end


