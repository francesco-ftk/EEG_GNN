%clear all
%clc
% CARICAMENTO EEG
load('Environment_EEG_bands.mat');
load('Environment_EEG_images.mat');

% PARAMETRI

momax = 20;     % maximum model order for model order estimation
regmode='OLS';  % VAR model estimation regression mode
morder = 'AIC'; % model order to use 
icregmode = 'LWR';  % information criteria regression mode

% COSTRUISCO MATRICE DI INPUT 18x3000x60

n=18; % elettrodi
m=3000; % campioni per finestra
N=60; % finestre immagini

EEG_alpha_subject_2_norm = normalize(EEG_alpha_subject_2); % Normalizza per colonne la matrice 
M = transpose(EEG_alpha_subject_2_norm(features_images_2(2,1):features_images_2(3,1)-1,:));
for i=2:60
     window_alpha_subject_2 = transpose(EEG_alpha_subject_2_norm(features_images_2(2,i):features_images_2(3,i)-1,:));
     M(:,:,i)=window_alpha_subject_2;
end

% CALCOLA ORDINE DEL MODELLO

[AIC,BIC] = tsdata_to_infocrit(M,momax,icregmode); % Estrae criteri Akaike e Bayesian
[~,bmo_AIC] = min(AIC); % minimo del criterio AIC (best order in min)
[~,bmo_BIC] = min(BIC); % minimo del criterio BIC

% Plot information criteria.

figure(1); clf;
plot((1:momax)',[AIC BIC]);
legend('AIC','BIC');

fprintf('\nbest model order (AIC) = %d\n',bmo_AIC);
fprintf('best model order (BIC) = %d\n',bmo_BIC);

if strcmpi(morder,'AIC')
    morder = bmo_AIC;
    fprintf('\nusing AIC best model order = %d\n',morder);
elseif strcmpi(morder,'BIC')
    morder = bmo_BIC;
    fprintf('\nusing BIC best model order = %d\n',morder);
end

[F,A,SIG] = GCCA_tsdata_to_pwcgc(M,morder,regmode); % use same model order for reduced as for full regressions


% Check for failed (full) regression

assert(~isbad(A),'VAR estimation failed');

% Check for failed GC calculation

assert(~isbad(F,false),'GC calculation failed');

fig= figure(2); clf;
plot_pw(F);
title('Pairwise-conditional GC');

% savefig(sprintf('granger_causality/subject_2_alpha')); % subject_2_banda
saveas(fig,'granger_causality/subject_2_alpha.png');