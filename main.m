% Clustering experiment
clear
data_name = 'pie_normalized';
load([pwd,'/',data_name,'.mat']);
X = X';
[nFea,nSamp] = size(X);
label = Y;
class_num = length(unique(label));
chosen_method = ["SPCA-PSD", "All features"]; % chose algorithms for running
%chosen_method = ["CSPCA-PSD","AW-SPCA-PSD"]
%chosen_method = ['All']
save_path = [pwd,'/result.mat'];

k = 5; % knn neighbour
N = 30;
if nFea > 300
   features = 50:50:300;
else
   features = 10:20:110;
end

para1 = 10.^[-4:1:4];
para2 = 10.^[-4:1:4];

%% SPCA-PSD
if  ~isempty(find(chosen_method == "SPCA-PSD", 1)) || ~isempty(find(chosen_method == "All", 1))
    
    ACC_SPCA_PSD = cell(length(para1),1);
    NMI_SPCA_PSD= cell(length(para1),1);
    STD_ACC_SPCA_PSD = cell(length(para1),1);
    STD_NMI_SPCA_PSD = cell(length(para1),1);
    TIME_SPCA_PSD = zeros(length(para1),length(para2));
    OBJ_SPCA_PSD = cell(length(para1),1);
    for i1 = 1:length(para1)
        OBJ_SPCA_PSD{i1} = cell(length(para2),1);
        Acc_SPCA_PSD = cell(length(para2),length(features));
        Nmi_SPCA_PSD = cell(length(para2),length(features));
        Std_Acc_SPCA_PSD = cell(length(para2),length(features));
        Std_Nmi_SPCA_PSD = cell(length(para2),length(features));
        lambda = para1(i1);
        for i2 = 1:length(para2)
            eta = para2(i2);
            tic
            [ omega, id_SPCA_PSD, obj ] = SPCA_PSD( X , lambda, eta);
            TIME_SPCA_PSD(i1,i2) = toc;
            [~,trivial] = find(obj == 0);
            obj(trivial) = [];
            OBJ_SPCA_PSD{i1}{i2} = obj;
            parfor i3 = 1:length(features)
                [Acc_SPCA_PSD{i2,i3},Nmi_SPCA_PSD{i2,i3},Std_Acc_SPCA_PSD{i2,i3},Std_Nmi_SPCA_PSD{i2,i3}] = Evaluation(X,label,id_SPCA_PSD,features(i3),class_num,N);
            end

            disp(['para1=',num2str(para1(i1)),' and ', 'para2=', num2str(para2(i2)), ' SPCA-PSD ', data_name ,' ',num2str(TIME_SPCA_PSD(i1,i2)), ' seconds'])
            save(save_path);
        end
        ACC_SPCA_PSD{i1} = cell2mat(Acc_SPCA_PSD);
        NMI_SPCA_PSD{i1} = cell2mat(Nmi_SPCA_PSD);
        STD_ACC_SPCA_PSD{i1} = cell2mat(Std_Acc_SPCA_PSD);
        STD_NMI_SPCA_PSD{i1} = cell2mat(Std_Nmi_SPCA_PSD);      
    end
    temp_acc = cell2mat(ACC_SPCA_PSD);
    temp_nmi = cell2mat(NMI_SPCA_PSD);
    [acc_SPCA_PSD,op_SPCA_PSD] = max(temp_acc,[],1);
    for ii = 1:length(features)
        a = mod(op_SPCA_PSD(ii),length(para2));
        if a == 0
            opp1_SPCA_PSD(ii) = para1(floor(op_SPCA_PSD(ii)./length(para2)));
        else
            opp1_SPCA_PSD(ii) = para1(floor(op_SPCA_PSD(ii)./length(para2))+1);
        end
    end
        
    index = [];
    for ii = 1:length(features)
        a = mod(op_SPCA_PSD(ii),length(para2));
        if op_SPCA_PSD(ii) == 1
            index = [index,1];
            continue
        elseif op_SPCA_PSD(ii)~= 1 && a == 0
            index = [index,length(para2)];
        else 
            index = [index,a];
        end
    end
    opp2_SPCA_PSD = para2(index);
    [nmi_SPCA_PSD,~] = max(temp_nmi,[],1);
    disp(['the best ACC of SPCA-PSD with ', '(',num2str(features), ')',' features: ', num2str(acc_SPCA_PSD)]);
    disp(['the best NMI of SPCA-PSD with ','(', num2str(features), ')',' features: ', num2str(nmi_SPCA_PSD)]);
    disp('SPCA-PSD finished');
    save(save_path);
end


if  ~isempty(find(chosen_method == "All features", 1))  || ~isempty(find(chosen_method == "All", 1))
    ACC_All_Feature = cell(1,N);
    NMI_All_Feature = cell(1,N);
    parfor i1 = 1:N
        idx = kmeans(X',class_num);
        final_idx = BestMapping(label,idx );
        ACC_All_Feature{i1} = 1 - length(find(final_idx -label)) / nSamp;
        NMI_All_Feature{i1} = nmi(label, final_idx);
    end
ACC_All_Feature = repmat(mean(cell2mat(ACC_All_Feature)),[1,length(features)]);
NMI_All_Feature = repmat(mean(cell2mat(NMI_All_Feature)),[1,length(features)]);
STD_ACC_All_Feature = std(ACC_All_Feature);
STD_NMI_All_Feature = std(NMI_All_Feature);
save(save_path);
end

%% AW-SPCA-PSD
if  ~isempty(find(chosen_method == "AW-SPCA-PSD", 1)) || ~isempty(find(chosen_method == "All", 1))
    ACC_AW_SPCA_PSD = cell(length(para1),length(features));
    NMI_AW_SPCA_PSD = cell(length(para1),length(features));
    STD_ACC_AW_SPCA_PSD = cell(length(para1),length(features));
    STD_NMI_AW_SPCA_PSD = cell(length(para1),length(features));      
    OBJ_AW_SPCA_PSD = cell(1,length(para1));
    TIME_AW_SPCA_PSD = zeros(1,length(para1));
    
    for i1 = 1:length(para1)
        lambda = para1(i1);
        tic
        [ id_AW_SPCA_PSD,obj ] = AW_SPCA_PSD(X,lambda);
        TIME_AW_SPCA_PSD(i1) = toc;
        [~,trivial] = find(obj == 0);
        obj(trivial) = [];
        OBJ_AW_SPCA_PSD{i1} = obj;
        parfor i2 = 1:length(features)
            [ACC_AW_SPCA_PSD{i1,i2},NMI_AW_SPCA_PSD{i1,i2},STD_ACC_AW_SPCA_PSD{i1,i2},STD_NMI_AW_SPCA_PSD{i1,i2}] = Evaluation(X,label,id_AW_SPCA_PSD,features(i2),class_num,N);
        end

        disp(['para1=',num2str(para1(i1)),' AW-SPCA-PSD ', data_name,' ',num2str(TIME_AW_SPCA_PSD(i1)),' seconds'])
        save(save_path);
    end
    temp_acc = cell2mat(ACC_AW_SPCA_PSD);
    temp_nmi = cell2mat(NMI_AW_SPCA_PSD);
    [acc_AW_SPCA_PSD,op_AW_SPCA_PSD] = max(temp_acc,[],1);
    opp_AW_SPCA_PSD = para1(op_AW_SPCA_PSD);
    [nmi_AW_SPCA_PSD,~] = max(temp_nmi,[],1);
    disp(['the best ACC of AW-SPCA-PSD with ','(', num2str(features), ')',' features: ', num2str(acc_SPCA_PSD)]);
    disp(['the best NMI of AW-SPCA-PSD with ', '(',num2str(features), ')',' features: ', num2str(nmi_SPCA_PSD)]);
    disp('AW-SPCA-PSD finished');
    save(save_path);
end

%% CSPCA_PSD
if  ~isempty(find(chosen_method == "CSPCA-PSD", 1)) || ~isempty(find(chosen_method == "All", 1))
    
    ACC_CSPCA_PSD = cell(length(para1),1);
    NMI_CSPCA_PSD= cell(length(para1),1);
    STD_ACC_CSPCA_PSD = cell(length(para1),1);
    STD_NMI_CSPCA_PSD = cell(length(para1),1);
    TIME_CSPCA_PSD = zeros(length(para1),length(para2));
    OBJ_CSPCA_PSD = cell(length(para1),1);
    p = 1;
    for i1 = 1:length(para1)
        OBJ_CSPCA_PSD{i1} = cell(length(para2),1);
        Acc_CSPCA_PSD = cell(length(para2),length(features));
        Nmi_CSPCA_PSD = cell(length(para2),length(features));
        Std_Acc_CSPCA_PSD = cell(length(para2),length(features));
        Std_Nmi_CSPCA_PSD = cell(length(para2),length(features));
        alpha = para1(i1);
        for i2 = 1:length(para2)
            beta = para2(i2);
            tic
            [id_CSPCA_PSD, obj ] = CSPCA_PSD(X,alpha,beta);
            TIME_CSPCA_PSD(i1,i2) = toc;
            [~,trivial] = find(obj == 0);
            obj(trivial) = [];
            OBJ_CSPCA_PSD{i1}{i2} = obj;
            parfor i3 = 1:length(features)
                [Acc_CSPCA_PSD{i2,i3},Nmi_CSPCA_PSD{i2,i3},Std_Acc_CSPCA_PSD{i2,i3},Std_Nmi_CSPCA_PSD{i2,i3}] = Evaluation(X,label,id_CSPCA_PSD,features(i3),class_num,N);
            end
            disp(['para1=',num2str(para1(i1)),' and ', 'para2=', num2str(para2(i2)),' CSPCA_PSD ', data_name,' ',num2str(TIME_CSPCA_PSD(i1,i2)),' seconds'])
            save(save_path);
        end
        ACC_CSPCA_PSD{i1} = cell2mat(Acc_CSPCA_PSD);
        NMI_CSPCA_PSD{i1} = cell2mat(Nmi_CSPCA_PSD);
        STD_ACC_CSPCA_PSD{i1} = cell2mat(Std_Acc_CSPCA_PSD);
        STD_NMI_CSPCA_PSD{i1} = cell2mat(Std_Nmi_CSPCA_PSD);
    end
    temp_acc = cell2mat(ACC_CSPCA_PSD);
    temp_nmi = cell2mat(NMI_CSPCA_PSD);
    [acc_CSPCA_PSD,op_CSPCA_PSD] = max(temp_acc,[],1);
    for ii = 1:length(features)
        a = mod(op_CSPCA_PSD(ii),length(para2));
        if a == 0
            opp1_CSPCA_PSD(ii) = para1(floor(op_CSPCA_PSD(ii)./length(para2)));
        else
            opp1_CSPCA_PSD(ii) = para1(floor(op_CSPCA_PSD(ii)./length(para2))+1);
        end
    end
        
    index = [];
    for ii = 1:length(features)
        a = mod(op_CSPCA_PSD(ii),length(para2));
        if op_CSPCA_PSD(ii) == 1
            index = [index,1];
            continue
        elseif op_CSPCA_PSD(ii)~= 1 && a == 0
            index = [index,length(para2)];
        else 
            index = [index,a];
        end
    end
    opp2_CSPCA_PSD = para2(index);
    [nmi_CSPCA_PSD,~] = max(temp_nmi,[],1);
    disp(['the best ACC of CSPCA-PSD with ','(', num2str(features), ')',' features: ', num2str(acc_SPCA_PSD)]);
    disp(['the best NMI of CSPCA-PSD with ', '(', num2str(features), ')', ' features: ', num2str(nmi_SPCA_PSD)]);
    disp('CSPCA-PSD finished');
    save(save_path);
end

if  ~isempty(find(chosen_method == "All features", 1))  || ~isempty(find(chosen_method == "All", 1))
    ACC_All_Feature = cell(1,N);
    NMI_All_Feature = cell(1,N);
    parfor i1 = 1:N
        idx = kmeans(X',class_num);
        final_idx = BestMapping(label,idx );
        ACC_All_Feature{i1} = 1 - length(find(final_idx -label)) / nSamp;
        NMI_All_Feature{i1} = nmi(label, final_idx);
    end
ACC_All_Feature = repmat(mean(cell2mat(ACC_All_Feature)),[1,length(features)]);
NMI_All_Feature = repmat(mean(cell2mat(NMI_All_Feature)),[1,length(features)]);
disp(['ACC with all features: ', num2str(ACC_All_Feature(1))]);
disp(['NMI with all features: ', num2str(NMI_All_Feature(1))]);
save(save_path);
end


