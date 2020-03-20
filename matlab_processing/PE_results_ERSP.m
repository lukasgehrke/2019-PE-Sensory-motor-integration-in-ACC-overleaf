%% clear all and load params
close all; clear all;

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

%% single-trial regression first-level and group level statistics

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

% get betas from single subject level of design:
model = 'ersp_sample ~ velocity * haptics + base';
robustfit = 0; % fit robust regression with squared weights, see fitlm
event_sample = 750;
log_scale = 0;

for cluster = 11 %13 %[10, 20, 21, 35, 38]
        
    % remove baseline from the model
    if log_scale
        model = model(1:end-7);
    end

    % outpath
    save_fpath = [bemobil_config.study_folder bemobil_config.study_level ...
        'analyses/ersp/' bemobil_config.study_filename(1:end-6) ...
        '/cluster_' num2str(cluster) '/' model];
    if ~exist(save_fpath, 'dir')
        mkdir(save_fpath);
    end        

    % get matching datasets from EEGLAB Study struct
    unique_setindices = unique(STUDY.cluster(cluster).sets);
    unique_subjects = STUDY_sets(unique_setindices);
    all_setindices = STUDY.cluster(cluster).sets;
    all_sets = STUDY_sets(all_setindices);
    all_comps = STUDY.cluster(cluster).comps;

    % fit model to IC data
    count = 1;
    for subject = unique_subjects

        % select EEG dataset
        [~, ix] = find(subject==subjects);
        s = ALLEEG(ix);

        %DESIGN make continuous and dummy coded predictors
        congruency = s.etc.analysis.design.oddball';
        haptics = s.etc.analysis.design.haptics';
        trial_nr = s.etc.analysis.design.trial_number';
        direction = categorical(s.etc.analysis.design.direction)';
        sequence = s.etc.analysis.design.sequence';
        rt = s.etc.analysis.design.rt_spawned_touched';
        velocity = s.etc.analysis.mocap.mag_vel(event_sample,:)';

        % select 1st comp if more than 1 comp per subject
        this_sets = find(all_sets==subject);
        comps = all_comps(this_sets);

        % fitlm for each time frequency pixel
        tic
        disp(['Now running analysis for cluster: ' num2str(cluster) ' and subject: ' num2str(subject) ' with component: ' num2str(comps)]);
        for t = 1:size(s.etc.analysis.ersp.tf_event_times,2)
            for f = 1:size(s.etc.analysis.ersp.tf_event_freqs,2)

                % add ersp and baseline sample to design matrix
                if size(comps,2) > 1
                    ersp_sample = squeezemean(s.etc.analysis.ersp.tf_event_raw_power(comps,f,t,:),1);
                    base = squeezemean(s.etc.analysis.ersp.tf_base_raw_power(comps,f,:),1);
                else
                    ersp_sample = squeeze(s.etc.analysis.ersp.tf_event_raw_power(comps,f,t,:));
                    base = squeeze(s.etc.analysis.ersp.tf_base_raw_power(comps,f,:));
                end

                % make full design table either including baseline or
                % not
                if log_scale
                    base = mean(base,1);
                    ersp_sample = 10.*log10(ersp_sample./base);
                    % not including baseline as divided ensemble
                    % baseline correction was applied
                    design = table(ersp_sample, congruency, haptics, trial_nr, direction, sequence, velocity, rt);    
                else
                    % including baseline as a vector/regressor
                    design = table(ersp_sample, base, congruency, haptics, trial_nr, direction, sequence, velocity, rt);    
                end

                % remove bad trials
                design(s.etc.analysis.ersp.rm_ixs,:) = [];

                % select only mismatch trials for velocity analysis
                if contains(model,'velocity')
                    design(design.congruency==false,:) = [];
                end

                if robustfit
                    mdl = fitlm(design, model, 'RobustOpts', 'on');
                else
                    mdl = fitlm(design, model);
                end

                res.betas(count,f,t,:) = mdl.Coefficients.Estimate;
                res.t(count,f,t,:) = mdl.Coefficients.tStat;
                res.p(count,f,t,:) = mdl.Coefficients.pValue;
                res.r2(count,f,t,:) = mdl.Rsquared.Ordinary;
                res.adj_r2(count,f,t,:) = mdl.Rsquared.Adjusted;
            end
        end
        toc
        count = count + 1;
    end

    % add parameter names
    res.model = model;
    res.parameter_names = string(mdl.CoefficientNames)';
    save([save_fpath '/res_' model '_robust-' num2str(robustfit) '.mat'], 'res');

    %% LIMO: MCC
    % settings

    load([save_fpath '/res_' model '_robust-' num2str(robustfit) '.mat'], 'res');
    save_fpath = [bemobil_config.study_folder bemobil_config.study_level ...
        'analyses/ersp/' bemobil_config.study_filename(1:end-6) ...
        '/cluster_' num2str(cluster) '/' model];
    sig_alpha = .05;
    save_info.robustfit = 0;
    save_info.model = model;

    % load times and freqs
    load_path = [bemobil_config.study_folder bemobil_config.study_level 'analyses/ersp'];
    times = s.etc.analysis.ersp.tf_event_times;
    %times = s.etc.analysis.ersp.tf_event_times(94:118);
    times_ixs = [1, size(times,2)];
    
    freqs = s.etc.analysis.ersp.tf_event_freqs;
    %freqs = s.etc.analysis.ersp.tf_event_freqs(1:33);
    %max_freq_ix = 33;
    max_freq_ix = size(freqs,2);
    res.times = times;
    res.freqs = freqs;

    % select model
    if contains(model, 'velocity')
        save_info.parameters = {'(Intercept)', 'haptics_1', 'velocity', 'haptics_1:velocity'}; %
    else
        save_info.parameters = {'(Intercept)', 'congruency_1', 'haptics_1', 'rt', 'congruency_1:haptics_1'}; %
    end
    save_info.load_p = save_fpath;

    % main effects: run limo ttest against 0
    for param = save_info.parameters
        save_info.parameter = param{1};

        % run limo
        PE_limo(save_info, res, 1, times, times_ixs, freqs, max_freq_ix, []);

        % load LIMO output: save mean value and sig. mask to res and resave res
        save_name = regexprep(save_info.parameter, ':' , '_');
        save_name = regexprep(save_name, '(' , '');
        save_name = regexprep(save_name, ')' , '');

        % todo what is the correct name here
        load([save_info.load_p '/ttest_' save_name '/one_sample_ttest_parameter_1.mat']);
        res.ttest.(save_name).t = squeeze(one_sample(1,:,:,4)); % mean betas in this case
        res.ttest.(save_name).p = squeeze(one_sample(1,:,:,5)); % mean betas in this case
        res.ttest.(save_name).beta = squeeze(one_sample(1,:,:,1)); % mean betas in this case

        % mcc
        % load bootstraps and select 600 bootstraps with including more
        % than 12 unique subjects
        load([save_info.load_p '/ttest_' save_name '/H0/boot_table.mat']);
        boots = boot_table{1};
        for i = 1:size(boots,2)
            uniques(i) = size(unique(boots(:,i)),1);
        end
        min_uniques_factor = .8;
        min_uniques = floor(min_uniques_factor * max(uniques));
        ixs_all = find(uniques>=min_uniques);
        ixs = randsample(ixs_all,600);
        % load bootstrap results
        load([save_info.load_p '/ttest_' save_name '/H0/tfce_H0_one_sample_ttest_parameter_1.mat']);
        % get max dist of tfce
        for i = 1:size(tfce_H0_one_sample,3)
            this_tfce = squeeze(tfce_H0_one_sample(:,:,i));
            max_dist(i) = max(this_tfce(:));
        end
        % remove NaN
        %max_dist(isnan(max_dist)) = [];
        res.ttest.(save_name).min_uniques = min_uniques;
        res.ttest.(save_name).ixs_all = ixs_all;
        res.ttest.(save_name).max_dist_all = max_dist;
        res.ttest.(save_name).max_dist_600 = max_dist(ixs);
        
        % threshold
        res.ttest.(save_name).thresh = prctile(res.ttest.(save_name).max_dist_600, (1-sig_alpha)*100);
        % load true result
        load([save_info.load_p '/ttest_' save_name '/tfce/tfce_one_sample_ttest_parameter_1.mat']);
        % threshold true data with bootstrap prctile thresh
        res.ttest.(save_name).tfce = squeeze(tfce_one_sample);
        res.ttest.(save_name).sig_mask = res.ttest.(save_name).tfce>res.ttest.(save_name).thresh;

        % save res
        save([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '.mat'], 'res');
    end

    % clear results struct
    clear res    

end

%% permutation ttest to assess significance on betas

% load results
robustfit = 0;
cluster = 6;
model = 'ersp_sample ~ velocity * haptics + base';
save_fpath = [bemobil_config.study_folder bemobil_config.study_level ...
    'analyses/ersp/' bemobil_config.study_filename(1:end-6) ...
    '/cluster_' num2str(cluster) '/' model];
load([save_fpath '/res_' model '_robust-' num2str(robustfit) '.mat'], 'res');

%     "(Intercept)"
%     "base"
%     "haptics_1"
%     "velocity"
%     "haptics_1:velocity"

param = 2;
t = 91:141;
f = size(freqs,1):size(freqs,2);

% do permutation test. here the correct freq and time range must be taken since this will affect the values.
betas = permute(res.betas(:,f,t,param), [2,3,1]);
betas_zero = zeros(size(betas));

% % test anna
% test_data1 = squeeze(res.betas(:,1,:,2))';
% % size(test_data1)
% % 
% % ans =
% % 
% %    150    18
% test_data2 = squeeze(res.betas(:,1,:,3))';
% [s, df, betas_p_vals, surrog] = statcond({test_data1 test_data2},...
%     'method', 'perm', 'naccu', 10000);
% %[fdr_p_vals, fdr_p_mask] = fdr(betas_p_vals, .05);

[~, ~, betas_p_vals, ~] = statcond({betas betas_zero},...
    'method', 'perm', 'naccu', 10000);
% threshold surrogate data with max t statistic

% correct for multiple comparison using false discovery rate
%[fdr_p_vals, fdr_p_mask] = fdr(betas_p_vals, .05);

figure;
plotersp(res.times(t), res.freqs(f), squeezemean(betas,3), betas_p_vals, .05,...
    'frequency (Hz)', 'time in ms', res.parameter_names{param}, 'beta');
