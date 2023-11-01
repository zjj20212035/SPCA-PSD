function [ assignment ] = MarkReplace( MarkMat )
%将存储标签顺序的空间矩阵转换维一个行向量
[rows,cols] = size(MarkMat);

assignment = zeros(1,cols);

for i=1:rows
    for j=1:cols
        if MarkMat(i,j)==1
            assignment(1,j)=i;
        end
    end
end


end

