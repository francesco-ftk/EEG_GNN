% clear all
% clc
% CARICAMENTO EEG
% load('Environment_EEG_bands.mat');
% load('Environment_EEG_images.mat');

% PARAMETRI

momax = 20;     % maximum model order for model order estimation
regmode='OLS';  % VAR model estimation regression mode
morder = 'AIC'; % model order to use 
icregmode = 'LWR';  % information criteria regression mode

% COSTRUISCO MATRICE DI INPUT 18x3000x1 per ogni finestra (60) per ogni banda (per ora solo beta)

n=18; % elettrodi
m=3000; % campioni per finestra
N=1; % finestre immagini

EEG_subject_norm = normalize(EEG_beta_subject_26); % Normalizza per colonne la matrice
trueGround = features_images_26(4,1);
answer = features_images_26(5,1);
eval(['GCmatrix_26_1_' num2str(trueGround) '_' num2str(answer) '=1;']);
save("PythonDatas/GC/GCmatrix_beta_26.mat", sprintf('GCmatrix_26_1_%d_%d',features_images_26(4,1),features_images_26(5,1)));
InputGCNN_26_1 = 1;
save("PythonDatas/Input/Input_beta_26.mat", 'InputGCNN_26_1'); 
for i=1:60
     M= transpose(EEG_subject_norm(features_images_26(2,i):features_images_26(3,i)-1,:));
     
     % CALCOLA ORDINE DEL MODELLO

    [AIC,BIC] = tsdata_to_infocrit(M,momax,icregmode); % Estrae criteri Akaike e Bayesian
    [~,bmo_AIC] = min(AIC); % minimo del criterio AIC (best order in min)
    [~,bmo_BIC] = min(BIC); % minimo del criterio BIC

    if strcmpi(morder,'AIC')
        morder = bmo_AIC;
    elseif strcmpi(morder,'BIC')
        morder = bmo_BIC;
    end
    
    % CALCOLO Granger Causality MATRIX

    [F,A,SIG] = GCCA_tsdata_to_pwcgc(M,morder,regmode); % use same model order for reduced as for full regressions
    
    trueGround = features_images_26(4,i);
    answer = features_images_26(5,i);
    
    eval(['GCmatrix_26_' num2str(i) '_' num2str(trueGround) '_' num2str(answer) '=F;']); % GCmatrix_idSoggetto_numFinestra_trueGround_answer
    save("PythonDatas/GC/GCmatrix_beta_26.mat", sprintf('GCmatrix_26_%d_%d_%d',i,features_images_26(4,i),features_images_26(5,i)), '-append');
    eval(['InputGCNN_26_' num2str(i) '=M;']);
    save("PythonDatas/Input/Input_beta_26.mat", sprintf('InputGCNN_26_%d',i), '-append'); 
end

