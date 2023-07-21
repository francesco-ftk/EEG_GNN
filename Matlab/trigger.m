%% CODICE PER ESTRAZIONE DEGLI EVENTI DAL TRIGGER HUB (CONNESSO AL DSI-24)
%% ATTENZIONE: IL CODICE DI CUI DI SEGUITO SI LIMITA ALL'ESTRAZIONE DEI PRIMI 60 TASK (SINGOLA IMMAGINE PRESENTATA)

clear all
close all

subject_index_to_use=[2,3,6,9,11,12,13,15,17,18,19,22,23,24,26];
subject_index_to_exclude=[1,4,5,7,8,10,14,16,20,21,25];

subject_list=["01_Francesca","02_Caterina", "03_Paolo", "04_Claudio", ...
    "05_Andrea", "06_Angelo", "07_Antonio","08_Alice", "09_Giacomo", ...
    "10_Francesco","11_Marco", "12_Pietro", "13_Mirco", "14_Marco", ...
    "15_Giulia", "16_Sara", "17_Carla", "18_Francesca", ...
    "19_Margherita", "20_Francesca","21_Filippo", "22_Margherita", ...
    "23_Alessio", "24_Chiara", "25_Martina", "26_Chiara"];

%% INIZIO ANALISI SOGGETTI

per_h=struct;
per_s=struct;
per_n=struct;

for d=1:length(subject_index_to_use)

    index=subject_index_to_use(d);
    
    P = 'C:\Users\Acer\OneDrive\Documenti\Progetto\File txt dati cavie\';
    S = dir(fullfile(P,'*.txt'));
    N={S.name};
    
    subject = impsub_txt(strcat("C:\Users\Acer\OneDrive\Documenti\Progetto\File txt dati cavie\",N(index)), [2, Inf],subject_list(index));

    if contains(string(subject{2,2}),'ff') || contains(string(subject{2,2}),'mf')
         [true_h,false_h,ans_h_A,ans_h_NA,ima_h,answ_h,per_h,answtime_h,label_h]=extract_info_txt(subject,d,per_h,2,78,'happy');
         if contains(string(subject{82,2}),'ft') || contains(string(subject{82,2}),'mt')
             [true_s,false_s,ans_s_A,ans_s_NA,ima_s,answ_s,per_s,answtime_s,label_s]=extract_info_txt(subject,d,per_s,82,158,'sad');
             [true_n,false_n,ans_n_A,ans_n_NA,ima_n,answ_n,per_n,answtime_n,label_n]=extract_info_txt(subject,d,per_n,162,238,'neutral');
         else
             [true_n,false_n,ans_n_A,ans_n_NA,ima_n,answ_n,per_n,answtime_n,label_n]=extract_info_txt(subject,d,per_n,82,158,'neutral');
             [true_s,false_s,ans_s_A,ans_s_NA,ima_s,answ_s,per_s,answtime_s,label_s]=extract_info_txt(subject,d,per_s,162,238,'sad');
         end
    elseif contains(string(subject{2,2}),'ft') || contains(string(subject{2,2}),'mt')
        [true_s,false_s,ans_s_A,ans_s_NA,ima_s,answ_s,per_s,answtime_s,label_s]=extract_info_txt(subject,d,per_s,2,78,'sad');
        if contains(string(subject{82,2}),'ff') || contains(string(subject{82,2}),'mf')
            [true_h,false_h,ans_h_A,ans_h_NA,ima_h,answ_h,per_h,answtime_h,label_h]=extract_info_txt(subject,d,per_h,82,158,'happy');
            [true_n,false_n,ans_n_A,ans_n_NA,ima_n,answ_n,per_n,answtime_n,label_n]=extract_info_txt(subject,d,per_n,162,238,'neutral');
        else
            [true_n,false_n,ans_n_A,ans_n_NA,ima_n,answ_n,per_n,answtime_n,label_n]=extract_info_txt(subject,d,per_n,82,158,'neutral');
            [true_h,false_h,ans_h_A,ans_h_NA,ima_h,answ_h,per_h,answtime_h,label_h]=extract_info_txt(subject,d,per_h,162,238,'happy');
        end
    elseif contains(string(subject{2,2}),'fn') || contains(string(subject{2,2}),'mn')
        [true_n,false_n,ans_n_A,ans_n_NA,ima_n,answ_n,per_n,answtime_n,label_n]=extract_info_txt(subject,d,per_n,2,78,'neutral');
        if contains(string(subject{82,2}),'ff') || contains(string(subject{82,2}),'mf')
            [true_h,false_h,ans_h_A,ans_h_NA,ima_h,answ_h,per_h,answtime_h,label_h]=extract_info_txt(subject,d,per_h,82,158,'happy');
            [true_s,false_s,ans_s_A,ans_s_NA,ima_s,answ_s,per_s,answtime_s,label_s]=extract_info_txt(subject,d,per_s,162,238,'sad');
        else
            [true_s,false_s,ans_s_A,ans_s_NA,ima_s,answ_s,per_s,answtime_s,label_s]=extract_info_txt(subject,d,per_s,82,158,'sad');
            [true_h,false_h,ans_h_A,ans_h_NA,ima_h,answ_h,per_h,answtime_h,label_h]=extract_info_txt(subject,d,per_h,162,238,'happy');
        end
    end
    
    %% PERCENTAGES ON 60 TOTAL IMAGES

    per.tot.corr=(per_h.corr{d}+per_s.corr{d}+per_n.corr{d})/3;
    per.tot.err=(per_h.err{d}+per_s.err{d}+per_n.err{d})/3;
    per.tot.tt=(per_h.tt{d}+per_s.tt{d}+per_n.tt{d})/3;
    per.tot.tf=(per_h.tf{d}+per_s.tf{d}+per_n.tf{d})/3;
    per.tot.ff=(per_h.ff{d}+per_s.ff{d}+per_n.ff{d})/3;
    per.tot.ft=(per_h.ft{d}+per_s.ft{d}+per_n.ft{d})/3;
    per.tot.sens=(per.tot.tt)/(per.tot.tt+per.tot.tf);
    per.tot.spec=(per.tot.ff)/(per.tot.ff+per.tot.ft);
    
    %% EDF EXTRACT FOR EEG 
    
    P = strcat('C:\Users\Acer\OneDrive\Documenti\Progetto\Soggetti\',subject_list(index),'\DSI_EEG\');
    
    [mat_eeg,info_der_eeg,mat_poli,info_poli,fs_eeg,fs_poli,length_Sig,header_edf] = get_edf_mat(P,'data_raw.edf');
    
    fs_EEG=300; %specify EEG sampling frequency
    mat_eeg([1:40],:) = []; %remove first 40 rows of signal (maybe used by DSI-24 for synchronization purposes)
    mat_poli([1:40],:) = []; %remove first 40 rows of signal (maybe used by DSI-24 for synchronization purposes)

    EEG_tv=(0:1000/fs_EEG:(length(mat_eeg)-1)*1000/fs_EEG).';

    % *******************************************************
    % FUNZIONE PER SALVARE LE MATRICI EEG DI TUTTI I SOGGETTI
    % *******************************************************
    
    eval(['EEG_subject_' num2str(index) '=mat_eeg;'])

    for i=1:size(mat_poli,1)
        if mat_poli(i,2)~=1 && mat_poli(i,2)~=0 && i<60000
            mat_poli(i,2)=0;
        end
    end
    
    %% TRIGGER SEPARATION OF IMAGES
    
    up=0;
    down=0;
    start=0;
    finish=0;
    
    for i=1:length(mat_poli(:,2))
        if mat_poli(i,2)==16 || mat_poli(i,2)==24
            up=up+1;
        elseif mat_poli(i,2)==0 
            down=down+1;
        elseif mat_poli(i,2)==1
            start=start+1;
        elseif mat_poli(i,2)==17
            finish=finish+1;
        end
    end
    
    trigger_up=zeros(up,1);
    trigger_down=zeros(down,1);
    trigger_start=zeros(start,1);
    trigger_finish=zeros(finish,1);
    
    i_up=1;
    i_down=1;
    i_start=1;
    i_finish=1;
    
    for i=1:length(mat_poli(:,2))
        if mat_poli(i,2)==16 || mat_poli(i,2)==24
            trigger_up(i_up,1)=i;
            i_up=i_up+1;
        elseif mat_poli(i,2)==0
            trigger_down(i_down,1)=i;
            i_down=i_down+1;
        elseif mat_poli(i,2)==1
            trigger_start(i_start,1)=i;
            i_start=i_start+1;
        elseif mat_poli(i,2)==17
            trigger_finish(i_finish,1)=i;
            i_finish=i_finish+1;
        end
    end
    
    baseline_finish_eyes_open=trigger_up(1,1)-5*fs_EEG;
    baseline_start_eyes_open=baseline_finish_eyes_open-2*60*fs_EEG;
    
    events=[baseline_start_eyes_open, baseline_finish_eyes_open,trigger_up(1)];
    
    tr=0;
    scart=0;

    for i=1:length(trigger_up)-1
        if trigger_up(i+1)-trigger_up(i)~=1 && trigger_up(i)+1<trigger_finish(1)
            if tr==20 || tr==60
                scart=scart+0.5;
            else
                events(end+1)=trigger_up(i);
                tr=tr+1;
            end
            if tr==40 || tr==68
                scart=scart+1;
            else
                events(end+1)=trigger_up(i)+1;
                events(end+1)=trigger_up(i+1)-1;
                tr=tr+1;
            end
            if tr==20 || tr==60
                scart=scart+0.5;
            else
                events(end+1)=trigger_up(i+1);
            end
        end
    end

    clear tr
    clear scart
    
    events(end+1)=trigger_finish(1);
    
    ima=ima_h+ima_s+ima_n;
    answ=answ_h+answ_s+answ_n;
    
    images=iminfo(ima,answ,events); % imfinfo
    
    % *******************************************************
    % FUNZIONE PER SALVARE LE MATRICI images DI TUTTI I SOGGETTI
    % *******************************************************
    eval(['features_images_' num2str(index) '=images;']);

end
