function [mean_acc,mean_nmi,std_acc,std_nmi] = Evaluation(X,label,id,features,class_num,N)
%calculate accuracy and NMI
ACC = zeros(1,N);
NMI = zeros(1,N);
n = length(label);
X_r = X(id(1:features),:);
for k = 1:N
    idx = kmeans(X_r',class_num);
    final_idx = BestMapping(label,idx );
    NMI(k) = nmi(label, final_idx);
    ACC(k) = 1 - length(find(final_idx -label)) / n;
end
mean_acc = mean(ACC);
std_acc = std(ACC);
mean_nmi = mean(NMI);
std_nmi = std(NMI);
end





