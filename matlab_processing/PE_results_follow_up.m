
%% clear all and load params
close all; clear

PE_config;

%% load study : 2nd Level inference

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 1, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

% load IMT_v1 EEGLAB study struct, keeping at most 1 dataset in memory
input_path_STUDY = [bemobil_config.study_folder bemobil_config.study_level];
if isempty(STUDY)
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    [STUDY ALLEEG] = pop_loadstudy('filename', bemobil_config.study_filename, 'filepath', input_path_STUDY);
    CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
    
    eeglab redraw
end
STUDY_sets = cellfun(@str2num, {STUDY.datasetinfo.subject});

% select subjects out of clusters of int
clusters_of_int = [3, 7, 9, 24, 25, 28, 30, 33, 34];
% clusters
% 3: right parietal
% 7: right motor?
% 24: right SMA
% 25: left parietal
% 28: interesting
% 33: ACC

channels_of_int = [5, 25, 65];
% channels
% 5: Fz
% 25: Pz
% 65: FCz

%% FINAl discussion results: single trial regression velocity ERP

% Questions:
% 1. do participants slow down more in mismatch trials with haptics?
% significant effect that this participant slowed down more in haptic
% mismatch trials compared to just visual feedback

% 2. Are trials after mismatch trials differently initiated? this is 
% interesting but not much there, must check across subjects

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

% get betas from single subject level of design:
models = {'vel_erp_sample ~ haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ after_mismatch * haptics + trial_nr + direction + was_sequence'};
robustfit = 0; % fit robust regression with squared weights, see fitlm
event_sample = 750;
window = event_sample-250:event_sample+250; %[-1 1]seconds start and end of interesting, to be analyzed, samples
count = 1;

for model = models
    model = model{1};

    % outpath
    save_fpath = [bemobil_config.study_folder bemobil_config.study_level ...
        'analyses/mocap/' bemobil_config.study_filename(1:end-6) ...
        '/vel/' model];
    if ~exist(save_fpath, 'dir')
        mkdir(save_fpath);
    end        

    % load data
    count = 1;
    for s = ALLEEG
        disp(['Now running analysis for velocity and subject: ' num2str(count+1)]);

        %DESIGN make continuous and dummy coded predictors
        congruency = s.etc.epoching.oddball';
        haptics = s.etc.epoching.haptics';
        trial_nr = s.etc.epoching.trial_number';
        direction = categorical(s.etc.epoching.direction)';
        sequence = s.etc.epoching.sequence';
        after_mismatch = [0; congruency(1:end-1)];
        was_sequence = [0 s.etc.epoching.sequence(1:end-1)]';
        if strcmp(model, models{1})
            sequence = sequence(congruency);
            direction = direction(congruency);
            trial_nr = trial_nr(congruency);
            haptics = haptics(congruency);
        end

        tic
        for sample = window

            vel_erp_sample = s.etc.analysis.mocap.mag_vel(sample,:)';
            
            % design matrix per sample
            if strcmp(model, models{1})
                vel_erp_sample = vel_erp_sample(congruency);                
                design = table(vel_erp_sample, haptics, trial_nr, direction, sequence);
            else
                design = table(vel_erp_sample, haptics, trial_nr, direction, sequence, after_mismatch, was_sequence);
            end
            
            if robustfit
                mdl = fitlm(design, model, 'RobustOpts', 'on');
            else
                mdl = fitlm(design, model);
            end

            ix = sample-window(1)+1;
            res.betas(count,ix,:) = mdl.Coefficients.Estimate;
            res.t(count,ix,:) = mdl.Coefficients.tStat;
            res.p(count,ix,:) = mdl.Coefficients.pValue;
            res.r2(count,ix,:) = mdl.Rsquared.Ordinary;
            res.adj_r2(count,ix,:) = mdl.Rsquared.Adjusted;
        end
        toc
        count = count + 1;
    end

    % add parameter names
    res.model = model;
    res.parameter_names = string(mdl.CoefficientNames)';
    %save([save_fpath '/res_' model '_robust-' num2str(robustfit) '.mat'], 'res');

    %LIMO ttests
    % settings
    sig_alpha = .05;
    save_info.robustfit = 0;
    save_info.model = model;
    save_info.load_p = save_fpath;
    save_info.parameters = res.parameter_names;

    % main effects: run limo ttest against 0
    for i = 1:size(save_info.parameters,1)
        save_info.parameter = save_info.parameters{i};
        PE_limo(save_info, res, 0, [], [], [], [], []);

        % load LIMO output: save mean value and sig. mask to res and resave res
        save_name = regexprep(save_info.parameter, ':' , '_');
        save_name = regexprep(save_name, '(' , '');
        save_name = regexprep(save_name, ')' , '');
        load([save_info.load_p '/ttest_' save_name '/one_sample_ttest_parameter_1.mat']);
        res.ttest.(save_name).mean_value = squeeze(one_sample(1,:,1)); % mean betas in this case

        % mcc
        % load bootstrap results
        load([save_info.load_p '/ttest_' save_name '/H0/tfce_H0_one_sample_ttest_parameter_1.mat']);
        % get max dist of tfce
        for j = 1:size(tfce_H0_one_sample,3)
            this_tfce = squeeze(tfce_H0_one_sample(1,:,j));
            max_dist(j) = max(this_tfce(:));
        end
        % threshold
        thresh = prctile(max_dist, (1-sig_alpha)*100);
        % load true result
        load([save_info.load_p '/ttest_' save_name '/tfce/tfce_one_sample_ttest_parameter_1.mat']);
        % threshold true data with bootstrap prctile thresh
        res.ttest.(save_name).sig_mask = find(tfce_one_sample>thresh);    
    end
    
    % save res
    save([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '.mat'], 'res');

    % clear results struct
    clear res
end
