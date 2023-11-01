function [ NewLabel ] = BestMapping( La1,La2 )
%����������ǩӳ�䵽��ʵ��ǩ��
%La1 : ��ʵ��ǩ��La2����������ǩ��NewLabel : ӳ���ı�ǩ

Label1 = unique(La1');
L1 = length(Label1);
Label2 = unique(La2');
L2 = length(Label2);

% �����������ַ����ǩ�ظ��ȵľ���G
G = zeros(max(L1,L2),max(L1,L2));
for i = 1:L1
    index1 = La1==Label1(1,i);
    for j=1:L2
        index2 = La2==Label2(1,j);
        G(i,j)=sum(index1.*index2);
    end
end

%�����������㷨�����ӳ�����ź�ľ���
[index]=munkres(-G);
%��ӳ�����Ž��ת��άһ���洢��ӳ�����ź��ǩ˳���������
[temp]=MarkReplace(index);
%����ӳ�����ź�ı�ǩNewLabel
NewLabel=zeros(size(La2));
for i=1:L2
    NewLabel(La2==Label2(i))=temp(i); 
end


end

