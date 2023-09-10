function [P_corrected,diagS] = Keep_PSD(P)
%Positive semidefinite projection for a square matrix
P = (P + P') / 2;
[U, S] = eig(P);
S = sign(S).* max(S,0);

index = find(diag(S));
S = S(index,index);
diagS = diag(S);
U = U(:,index);
P_corrected = U * S * U';
P_corrected = (P_corrected + P_corrected') / 2;
end

