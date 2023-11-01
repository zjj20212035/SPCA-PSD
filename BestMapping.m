function [ NewLabel ] = BestMapping( La1,La2 )
%将聚类结果标签映射到真实标签上
%La1 : 真实标签；La2：聚类结果标签；NewLabel : 映射后的标签

Label1 = unique(La1');
L1 = length(Label1);
Label2 = unique(La2');
L2 = length(Label2);

% 构建计算两种分类标签重复度的矩阵G
G = zeros(max(L1,L2),max(L1,L2));
for i = 1:L1
    index1 = La1==Label1(1,i);
    for j=1:L2
        index2 = La2==Label2(1,j);
        G(i,j)=sum(index1.*index2);
    end
end

%利用匈牙利算法计算出映射重排后的矩阵
[index]=munkres(-G);
%将映射重排结果转换维一个存储有映射重排后标签顺序的行向量
[temp]=MarkReplace(index);
%生成映射重排后的标签NewLabel
NewLabel=zeros(size(La2));
for i=1:L2
    NewLabel(La2==Label2(i))=temp(i); 
end


end

