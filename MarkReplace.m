function [ assignment ] = MarkReplace( MarkMat )
%���洢��ǩ˳��Ŀռ����ת��άһ��������
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

