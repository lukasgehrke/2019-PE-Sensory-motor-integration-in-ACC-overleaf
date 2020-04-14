%% clear all and load params
close all; clear all; clc;

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

%% FINAL single trial regression first-level and group level statistics

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

% get betas from single subject level of design:
%ERP:
model = 'erp_sample ~ velocity * haptics + rt';
chan = 65;
%MOCAP
%model = 'erp_sample ~ haptics + rt';
%chan = 'vel';

robustfit = 0; % fit robust regression with squared weights, see fitlm
event_sample = 750;
window = event_sample-25:event_sample+200; %[-.1 .8]seconds start and end of interesting, to be analyzed, samples
count = 1;
% outpath
save_fpath = [bemobil_config.study_folder bemobil_config.study_level ...
    'analyses/erp/' bemobil_config.study_filename(1:end-6) ...
    '/channel_' num2str(chan) '/' model];
if ~exist(save_fpath, 'dir')
    mkdir(save_fpath);
end        

% load data and run lmfit as 1st level summary
count = 1;
for s = ALLEEG
    disp(['Now running analysis for channel: ' num2str(chan) ' and subject: ' num2str(count+1)]);

    %DESIGN make continuous and dummy coded predictors
    congruency = s.etc.analysis.design.oddball';
    haptics = s.etc.analysis.design.haptics';
    trial_nr = s.etc.analysis.design.trial_number';
    direction = categorical(s.etc.analysis.design.direction)';
    sequence = s.etc.analysis.design.sequence';
    velocity = s.etc.analysis.mocap.mag_vel(event_sample,:)';
    rt = s.etc.analysis.design.rt_spawned_touched';

    tic
    for sample = window

        % make full design table
        if strcmp(chan,'vel')
            erp_sample  = s.etc.analysis.mocap.mag_vel(sample,:)';
        else
            erp_sample = squeeze(s.etc.analysis.erp.data(chan,sample,:));
        end
        design = table(erp_sample, congruency, haptics, trial_nr, direction, sequence, velocity, rt);
        % remove bad trials
        design(s.etc.analysis.erp.rm_ixs,:) = [];

        % select only mismatch trials for velocity analysis
        if contains(model,'velocity')
            design(design.congruency==false,:) = [];
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

% LIMO one-sample ttest main effects and tfce mcc
res.model = model;
res.parameter_names = string(mdl.CoefficientNames)';
save([save_fpath '/res_' model '_robust-' num2str(robustfit) '.mat'], 'res');
sig_alpha = .05;
save_info.robustfit = 0;
save_info.model = model;
% select model
if contains(model, 'velocity')
    save_info.parameters = {'(Intercept)', 'haptics_1', 'velocity', 'haptics_1:velocity', 'rt'}; %
else
    save_info.parameters = {'(Intercept)', 'rt', 'haptics_1'}; %
end
save_info.load_p = save_fpath;
for param = save_info.parameters
    save_info.parameter = param{1};
    PE_limo(save_info, res, 0, [], [], [], [], []);

    % load LIMO output: save mean value and sig. mask to res and resave res
    save_name = regexprep(save_info.parameter, ':' , '_');
    save_name = regexprep(save_name, '(' , '');
    save_name = regexprep(save_name, ')' , '');
    load([save_info.load_p '/ttest_' save_name '/one_sample_ttest_parameter_1.mat']);
    res.ttest.(save_name).t = squeeze(one_sample(1,:,4)); % mean betas in this case
    res.ttest.(save_name).p = squeeze(one_sample(1,:,5)); % mean betas in this case
    res.ttest.(save_name).beta = squeeze(one_sample(1,:,1)); % mean betas in this case

    % mcc
    % load bootstraps and select 800 bootstraps with including more
    % than 12 unique subjects
    load([save_info.load_p '/ttest_' save_name '/H0/boot_table.mat']);
    boots = boot_table{1};
    for i = 1:size(boots,2)
        uniques(i) = size(unique(boots(:,i)),1);
    end
    %min_uniques_factor = .8;
    min_uniques = 14; %ceil(min_uniques_factor * max(uniques));
    ixs_all = find(uniques>=min_uniques);
    ixs = randsample(ixs_all,800);
    % load bootstrap results
    load([save_info.load_p '/ttest_' save_name '/H0/tfce_H0_one_sample_ttest_parameter_1.mat']);
    % get max dist of tfce
    for i = 1:size(tfce_H0_one_sample,3)
        this_tfce = squeeze(tfce_H0_one_sample(1,:,i));
        max_dist(i) = max(this_tfce(:));
    end
    res.ttest.(save_name).min_uniques = min_uniques;
    res.ttest.(save_name).ixs_all = ixs_all;
    res.ttest.(save_name).max_dist_all = max_dist;
    res.ttest.(save_name).max_dist_800 = max_dist(ixs);

    % threshold
    res.ttest.(save_name).thresh = prctile(res.ttest.(save_name).max_dist_800, (1-sig_alpha)*100);
    % load true result
    load([save_info.load_p '/ttest_' save_name '/tfce/tfce_one_sample_ttest_parameter_1.mat']);
    res.ttest.(save_name).tfce = tfce_one_sample;
    % threshold true data with bootstrap prctile thresh
    res.ttest.(save_name).sig_mask = find(tfce_one_sample>res.ttest.(save_name).thresh);

    % resave res
    save([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '.mat'], 'res');
end
        
% clear results struct
clear res
