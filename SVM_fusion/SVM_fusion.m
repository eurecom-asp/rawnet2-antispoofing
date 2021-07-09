clear;clc;closeall;

%% We fused our all 3 RawNet2 (linear,mel and inverse-mel) systems with our high-resolution LFCC baseline system (Interspeech 2020 paper)



%% Load devlopment and evalution scores with target labels

load S_dev.mat 
load S_eval.mat


% sort the eval scores and targets 
[eval_labels, ind] = sort(eval_labels,'descend');
S_eval = S_eval(ind,:);


%% To generate S_dev and S_eval matrix by putting scores of all the four systems (High-resolution LFCC-GMM system and all three RawNet2-based systems) 
%% size of S_dev=(no. of development trials,no. of fused systems) --> (24844,4)
%% size of S_eval=(no. of eval trials,no. of fused systems) --> (71237,4)

S_dev = S_dev';   
S_eval = S_eval';

% get indices of genuine and spoof files by giving labels for bonafide=1 and spoof=0
bonafideIdx_dev = find(dev_labels==1);
spoofIdx_dev = find(dev_labels==0);

bonafideIdx_eval = find(eval_labels==1);
spoofIdx_eval = find(eval_labels==0);


%% SVM fusion

% train SVM on S_dev (development scores generated from all 4 systems) and
% test on S_eval (evalution scores generated from all 4 systems)

SVMModel = fitcsvm(S_dev',dev_labels','KernelFunction','polynomial','KernelScale','auto','PolynomialOrder',5,'Nu',0.5,'Standardize',true,'OutlierFraction',0);

% Score prediction on dev data
[~,scores_cm_dev] = predict(SVMModel,S_dev'); scores_cm_dev = scores_cm_dev(:,2);  % dev scores in scores_cm_dev

% read development protocol
fileID = fopen(fullfile('ASVspoof2019.LA.cm.dev.trl.txt'));
protocol = textscan(fileID, '%s%s%s%s');
fclose(fileID);

attacks = protocol{4};
type = protocol{3};

% metric for ASVspoof2019
evaluate_tDCF_asvspoof19_v1(attacks, type, scores_cm_dev, 'dev')

% Score prediction on eval data 

[~,scores_cm_eval] = predict(SVMModel,S_eval'); scores_cm_eval = scores_cm_eval(:,2);  % eval scores in scores_cm_eval


