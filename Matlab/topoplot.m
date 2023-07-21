%clear all  
%clc

% CARICO TUTTE LE EEG PER OGNI SOGGETTO

%load('Environment_EEG.mat');

%subject_index_to_use=[2,3,6,9,11,12,13,15,17,18,19,22,23,24,26];

% save...
% FILTRO PASSA BANDA A TUTTE GLI EEG

% for d=1:length(subject_index_to_use)
% 
%     index=subject_index_to_use(d);
%     
%     y = bandpass(eval(['EEG_subject_' num2str(index)]),[1,50],300);
%     
%     eval(['EEG_filtered_subject_' num2str(index) '=y;'])
% end

% save...
% load('Environment_EEG_filtered.mat');

% SUDDIVISIONE IN BANDE

% bands=["delta","theta","alpha","beta","gamma"];
% 
% for d=1:length(subject_index_to_use)
%     
%     index=subject_index_to_use(d);
%      
%     y = bandpass(eval(['EEG_filtered_subject_' num2str(index)]),[1,4],300);   
%     eval(['EEG_delta_subject_' num2str(index) '=y;'])
%     
%     y = bandpass(eval(['EEG_filtered_subject_' num2str(index)]),[4,8],300);   
%     eval(['EEG_theta_subject_' num2str(index) '=y;'])
%     
%     y = bandpass(eval(['EEG_filtered_subject_' num2str(index)]),[8,14],300);   
%     eval(['EEG_alpha_subject_' num2str(index) '=y;'])
%     
%     y = bandpass(eval(['EEG_filtered_subject_' num2str(index)]),[14,30],300);   
%     eval(['EEG_beta_subject_' num2str(index) '=y;'])
%     
%     y = bandpass(eval(['EEG_filtered_subject_' num2str(index)]),[30,50],300);   
%     eval(['EEG_gamma_subject_' num2str(index) '=y;'])
%     
%  end

% save...

% trigger_modified.m
% save("Environment_EEG_images.mat", "features_images_2", "features_images_3", "features_images_6", "features_images_9", "features_images_11", "features_images_12", "features_images_13", "features_images_15", "features_images_17", "features_images_18", "features_images_19", "features_images_22", "features_images_23","features_images_24", "features_images_26");

% figure; topoplot(x,'channellocs.ced'); cbar('vert',0,[-1 1]*max(abs(x)));

% **** ESTRAZIONE IMMAGINI TOPOPLOT ***

%    clear all  
%    clc
    load('Environment_EEG_bands.mat');
    load('Environment_EEG_images.mat');
% 
%  EEG_alpha_subject_2_norm = normalize(EEG_alpha_subject_2); % Normalizza per colonne la matrice
% % 
%  for i=1:60
%      window_alpha_subject_2 = EEG_alpha_subject_2_norm(features_images_2(2,i):features_images_2(3,i),:);
%      x=bandpower(window_alpha_subject_2);
%      %x=normalize(x);
%      fig= figure; topoplot(x,'channellocs.ced', 'maplimits', 'maxmin'); cbar('vert',0);
%      saveas(fig,sprintf('topoplot/subject_2_%d_%d_%d.png',i,features_images_2(4,i),features_images_2(5,i)));
%      % savefig(sprintf('topoplot/subject_2_%d_%d_%d.fig',i,features_images_2(4,i),features_images_2(5,i))); % subject_2_idImage_true_answer
%      close;
%  end
% 

