%% clear all and load params
clear all;

if ~exist('ALLEEG','var')
	eeglab;
end

% add downloaded analyses code to the path
addpath(genpath('/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing'));
% TODO add to path bemobil_pipeline repository download folder
% TODO add to path custom scripts repository Lukas Gehrke folder

% BIDS data download folder
bemobil_config.BIDS_folder = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/ds003552';
% Results output folder -> external drive
bemobil_config.study_folder = fullfile('/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error', 'derivatives');

% init
config_processing_pe;
subjects = 1:19;

%% load study

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 1, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

if isempty(STUDY)
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    [STUDY ALLEEG] = pop_loadstudy('filename', ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_' bemobil_config.study_filename], 'filepath', bemobil_config.study_folder);
    CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];    
    eeglab redraw
end
STUDY_sets = cellfun(@str2num, {STUDY.datasetinfo.subject});

%% compute movement predictors

for subject = subjects
    
    % no effect and correctly calculated
    ALLEEG(subject).etc.analysis.design.reaction_time = ...
        (ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample - ALLEEG(subject).etc.analysis.design.spawn_event_sample) / ALLEEG(subject).srate;
    
    % 
    ALLEEG(subject).etc.analysis.design.action_time = ...
        (abs(ALLEEG(subject).etc.analysis.design.movements.reach_off_sample) - ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample) / ALLEEG(subject).srate;
    
    ALLEEG(subject).etc.analysis.design.time_to_reach_vel_peak = ...
        ALLEEG(subject).etc.analysis.design.movements.reach_max_vel_ix - ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample;
    
    event_sample_ix = abs(bemobil_config.epoching.event_epochs_boundaries(1)) * ALLEEG(subject).srate; % epoched [-3 2] seconds = 1250 samples
    ALLEEG(subject).etc.analysis.design.peak_vel_to_contact = ...
        event_sample_ix - ALLEEG(subject).etc.analysis.design.movements.reach_max_vel_ix;
    
    ALLEEG(subject).etc.analysis.design.peak_vel_to_stop = ...
        ALLEEG(subject).etc.analysis.design.movements.reach_off_sample - ALLEEG(subject).etc.analysis.design.movements.reach_max_vel_ix;
end

% compute these measures in the results scripts    
%     EEG.etc.analysis.design.reaction_time = (EEG.etc.analysis.design.movements.movement_onset_sample - EEG.etc.analysis.design.spawn_event_sample) / EEG.srate;
%     EEG.etc.analysis.design.action_time = (abs(event_sample_ix) - EEG.etc.analysis.design.movements.movement_onset_sample) / EEG.srate;

%% settings results loop

cluster = 12;
c3 = 13;
c4 = 15;
disp(['cluster: ', num2str(cluster), ', location: ', num2str(STUDY.cluster(cluster).dipole.posxyz)]);

fit.match_trial_count = 1;

    %% load grand averages
% 
%         unique_setindices = unique(STUDY.cluster(cluster).sets);
%         unique_subjects = STUDY_sets(unique_setindices);
%         all_setindices = STUDY.cluster(cluster).sets;
%         all_sets = STUDY_sets(all_setindices);
%         all_comps = STUDY.cluster(cluster).comps;
%         for subject = unique_subjects
%             % select components
%             this_sets = find(all_sets==subject);
%             comp = all_comps(this_sets);
% 
%             % get trials
%             bad_trials = ALLEEG(subject).etc.analysis.design.bad_touch_epochs;
%             async_trials = ALLEEG(subject).etc.analysis.design.oddball == 'true';
%             async_trials(bad_trials) = []; % remove bad trials
%             sync_trials = ~async_trials;
% 
%             % load ersp
%             load(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_ersp.mat']));
%             
%             % remove ICs identical to Study IC selection
%             good_comps_ori_ix = find(ALLEEG(subject).etc.ic_classification.ICLabel.classifications(:,1) > bemobil_config.lda.brain_threshold);
%             
%             % select time window
%             if isempty(time_window_for_analysis)
%                 ersp.tf_event_raw_power = ersp.tf_event_raw_power(good_comps_ori_ix,:,:,:);
%                 ersp.tf_base_raw_power = ersp.tf_base_raw_power(good_comps_ori_ix,:,:,:);
%                 ixs = 1:size(ersp.tf_event_raw_power,3);
%             else
%                 [~, t1_ix] = min(abs(ersp.tf_event_times-(time_window_for_analysis(1))));
%                 [~, t2_ix] = min(abs(ersp.tf_event_times-(time_window_for_analysis(2))));
%                 ixs = t1_ix:t2_ix;
%                 ersp.tf_event_raw_power = ersp.tf_event_raw_power(good_comps_ori_ix,:,ixs,:);
%                 ersp.tf_base_raw_power = ersp.tf_base_raw_power(good_comps_ori_ix,:,:);
%             end
% 
%             sync_event = squeeze(ersp.tf_event_raw_power(comp,:,:,sync_trials));
%             sync_base = squeeze(ersp.tf_base_raw_power(comp,:,sync_trials));
%             async_event = squeeze(ersp.tf_event_raw_power(comp,:,:,async_trials));
%             async_base = squeeze(ersp.tf_base_raw_power(comp,:,async_trials));
% 
%             % if more than one component, average component power
%             if size(comp ,2)>1
%                 sync_event = squeezemean(sync_event,1);
%                 sync_base = squeezemean(sync_base,1);
%                 async_event = squeezemean(async_event,1);
%                 async_base = squeezemean(async_base,1);
%             end
% 
%             % average sync. and async. ERSP, significance against baseline = baseline
%             % corrected ersp one-sample limo ttest, tfce thresh
%             grand_avg.ersp.sync.event(subject,:,:) = squeezemean(sync_event,3);
%             grand_avg.ersp.sync.base(subject,:) = squeezemean(sync_base,2)';
%             grand_avg.ersp.async.event(subject,:,:) = squeezemean(async_event,3);
%             grand_avg.ersp.async.base(subject,:) = squeezemean(async_base,2)';
% 
%             % single subject db transformed, this is not correct but good for
%             % checking
%             grand_avg.ersp.sync.base_corrected(subject,:,:) = 10.*log10(grand_avg.ersp.sync.event(subject,:,:) ./ grand_avg.ersp.sync.base(subject,:));
%             grand_avg.ersp.async.base_corrected(subject,:,:) = 10.*log10(grand_avg.ersp.async.event(subject,:,:) ./ grand_avg.ersp.async.base(subject,:));
%         end
% 
%         % remove missing subjects
%         grand_avg.ersp.sync.event = grand_avg.ersp.sync.event(unique_subjects,:,:);
%         grand_avg.ersp.sync.base = grand_avg.ersp.sync.base(unique_subjects,:);
%         grand_avg.ersp.async.event = grand_avg.ersp.async.event(unique_subjects,:,:);
%         grand_avg.ersp.async.base = grand_avg.ersp.async.base(unique_subjects,:);
% 
%         %% grand average oddball conditions permutation t-test
% 
%         conditions = {'sync', 'async'};
% 
%         for condition = conditions
%             [stats, df, p_vals, ~] = statcond({permute(grand_avg.ersp.(condition{1}).event, [2,3,1]),...
%                 permute(repmat(grand_avg.ersp.(condition{1}).base, [1,1,size(grand_avg.ersp.(condition{1}).event,3)]), [2,3,1])},...
%                 'method', 'perm', 'naccu', 1000);
%             [~, fdr_mask] = fdr(p_vals, .05);
%             p_vals = p_vals .* fdr_mask;
%             p_vals(p_vals==0) = 1;
% 
%             to_plot = 10.*log10(squeezemean(grand_avg.ersp.(condition{1}).event,1) ./ squeezemean(grand_avg.ersp.(condition{1}).base,1)');
%             normal;figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 960 300]);
%             plotersp(ersp.tf_event_times(ixs), ersp.tf_event_freqs, to_plot, p_vals, .05, [-2 2], '-', 'frequency (Hz)', 'time (ms)', condition{1}, 'dB', 1);    
%             print(gcf, fullfile(bemobil_config.study_folder, 'results', num2str(cluster), [condition{1} '.eps']), '-depsc');
%             close(gcf);
%         end
% 
%         %% grand average difference oddball conditions
% 
%         [stats, df, p_vals, ~] = statcond({permute(grand_avg.ersp.async.event, [2,3,1]),...
%             permute(grand_avg.ersp.sync.event, [2,3,1])},...
%             'method', 'perm', 'naccu', 1000);
%         alpha_fdr = .05;
%         [~, fdr_mask] = fdr(p_vals, alpha_fdr);
%         p_vals = p_vals .* fdr_mask;
%         p_vals(p_vals==0) = 1;
% 
%         asy = 10.*log10(squeezemean(grand_avg.ersp.async.event,1) ./ squeezemean(grand_avg.ersp.async.base,1)');
%         sy = 10.*log10(squeezemean(grand_avg.ersp.sync.event,1) ./ squeezemean(grand_avg.ersp.sync.base,1)');
%         difference = asy - sy;
%         normal;figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 960 300]);
%         plotersp(ersp.tf_event_times(ixs), ersp.tf_event_freqs, difference, p_vals, .05, [-1 1], '-', 'frequency (Hz)', 'time (ms)', 'difference', 'dB', 1);
%         print(gcf, fullfile(bemobil_config.study_folder, 'results', num2str(cluster), 'difference_oddball.eps'), '-depsc');
%         close(gcf);

%% single trials ERP regressions
% ersp ~ vel_at_col * haptics + rt + base
% velocity: does it impact multisensory integration during spatio-temporal binding
% haptics: same as velocity, does object rigidity further perturb spatio-temporal binding
% rt: is there a task clock? self initiated movement timing?
% base: what activity at baseline impacts activity at spatio-temporal binding event -> that part of the activity is not processing related

% get matching datasets from EEGLAB Study struct
unique_setindices = unique(STUDY.cluster(cluster).sets);
unique_subjects = STUDY_sets(unique_setindices);
all_setindices = STUDY.cluster(cluster).sets;
all_sets = STUDY_sets(all_setindices);
all_comps = STUDY.cluster(cluster).comps;

for subject = unique_subjects

    % select components
    this_sets = find(all_sets==subject);
    comps = all_comps(this_sets);

    % predictors task
    oddball = double((ALLEEG(subject).etc.analysis.design.oddball=='true'))';
    post_error = [0; oddball(1:end-1)];
    isitime = ALLEEG(subject).etc.analysis.design.isitime';
    sequence = ALLEEG(subject).etc.analysis.design.sequence';
    trial_number = ALLEEG(subject).etc.analysis.design.trial_number';
    haptics = double(ALLEEG(subject).etc.analysis.design.haptics)';
    direction = categorical(ALLEEG(subject).etc.analysis.design.direction');
    pID = repmat(subject,size(direction,1),1);
    diff_at = [diff(ALLEEG(subject).etc.analysis.design.action_time), mean(diff(ALLEEG(subject).etc.analysis.design.action_time))]'; % from movement onset to touch
    velocity_at_impact = ALLEEG(subject).etc.analysis.motion.mag_vel(event_sample_ix,:)';

    reg_t = table(post_error, isitime, sequence, haptics, trial_number, direction, oddball, diff_at, velocity_at_impact, pID);
    reg_t(ALLEEG(subject).etc.analysis.design.bad_touch_epochs,:)= [];

    % match trial count
    if fit.match_trial_count
        match_ixs = find(reg_t.oddball==0);
        mismatch_ixs = find(reg_t.oddball==1);
        match_ixs = randsample(match_ixs, numel(mismatch_ixs));
        matched_trials = union(match_ixs, mismatch_ixs);
        reg_t = reg_t(matched_trials,:);
    end

    % load movement
    load(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_motion.mat']));
    motion.mag_vel(:, ALLEEG(subject).etc.analysis.design.bad_touch_epochs) = [];
    sample_time = (-3:1/ALLEEG(subject).srate:2) * 1000;
    sample_time(end) = [];

    % load erp
    load(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_filtered_erp.mat']));

    % remove ICs identical to Study IC selection
    good_comps_ori_ix = find(ALLEEG(subject).etc.ic_classification.ICLabel.classifications(:,1) > bemobil_config.lda.brain_threshold);
    filtered_erp.comp = filtered_erp.comp(good_comps_ori_ix,:,:);
    
    % add ersp and baseline sample to design matrix
    if size(comps,2) > 1
        comp_rp = squeezemean(filtered_erp.comp(comps,:,:),1);
    else
        comp_rp = squeeze(filtered_erp.comp(comps,:,:));
    end
    
    spawn = ceil(mean(ALLEEG(subject).etc.analysis.design.spawn_event_sample));
    start = spawn - ceil(mean(ALLEEG(subject).etc.analysis.design.isitime*250));
    start_move = ceil(mean(ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample));
    
    figure;
%     plot(squeezemean(filtered_erp.chan(c3,:,:),3)); hold on;
%     plot(squeezemean(filtered_erp.chan(c4,:,:),3)); hold on;
    plot(squeezemean(filtered_erp.chan(c4,:,:),3) - squeezemean(filtered_erp.chan(c3,:,:),3)); hold on;
%     plot(squeezemean(comp_rp,2));
    vline(spawn,'-','spawn');
    vline(start,'-','start');
    vline(start_move,'-','startmove');
    legend({'erp'});
%     legend({'c3','c4','diff','motor_cluster', 'spawn','start_movement'});


    %% ttest all significant channels 0tostart vs starttospawn
	spawns = ALLEEG(subject).etc.analysis.design.spawn_event_sample';
    starts = ceil(spawns - (ALLEEG(subject).etc.analysis.design.isitime*250)');
    
    bad_epochs = ALLEEG(subject).etc.analysis.design.bad_touch_epochs;
    starts(bad_epochs) = [];
    spawns(bad_epochs) = [];
    
    bad_epochs2 = find(starts<=0);
    starts(bad_epochs2) = [];
    spawns(bad_epochs2) = [];
    
    for c = 1:numel(ALLEEG(subject).chanlocs)
        data = squeeze(filtered_erp.chan(c,:,:));
        data(:,bad_epochs2) = [];
        for t = 1:size(data,2)
            chan_data_0_to_start(c,t) = mean(mean(data(1:starts(t),:),1));
            chan_data_start_to_spawn(c,t) = mean(mean(data(starts(t):spawns(t),:),1));
        end
        
        [H,P,CI,STATS] = ttest(chan_data_0_to_start(c,:), chan_data_start_to_spawn(c,:));
        t_scores(c) = STATS.tstat;
    end
%     clear chan_data_0_to_start chan_data_start_to_spawn
    figure;topoplot(t_scores, ALLEEG(subject).chanlocs,'electrodes','labels'); cbar; % scalp map of t scores of channels best separating rest vs. movement anticipation
    
    sorted_t_scores = sort(abs(t_scores));
    best_chan_ix = find(t_scores>prctile(sorted_t_scores,80));
    best_chan_labels = {ALLEEG(subject).chanlocs(best_chan_ix).labels};
    
    class1 = chan_data_0_to_start(best_chan_ix,:)';
    class2 = chan_data_start_to_spawn(best_chan_ix,:)';
    vals = array2table([class1; class2], 'VariableNames', best_chan_labels);
    classes = table([zeros(1,size(class1,1)), ones(1,size(class2,1))]', 'VariableNames', {'class'});
    t = [vals,classes];
    
    %%
    
    k = 25;
    indices = crossvalind('Kfold',t.class,k);
    for i = 1:k
        test = indices==i;
        train = ~test;

        modelfit = fitglm(t(train,:), 'class ~ TP9 + TP10');
    %     modelfit = fitlme(dmatrix(train,:), 'diff_at ~ vel_event + (1|pID)');

        ypred = predict(modelfit, t(test,:));
        ypred(ypred>=.5) = 1;
        ypred(ypred<.5) = 0;
        accuracy(i) = sum(ypred==t(test,:).class) / size(t(test,:).class,1);

        pconf = simulateChance(round(sum(test)/2) * [1 1], .05);
        sim_chance(i) = pconf(3);

        disp(['accuracy: ' num2str(accuracy(i))])
    end
    
    [H,P,CI,STATS] = ttest(accuracy,sim_chance)
%     clear accuracy sim_chance
    
    %%
    Mdl = fitcdiscr(t.TP9,t.class)
    
    figure;
    h1 = gscatter(t.TP9,t.TP10,t.class,'krb','ov^',[],'off');
    h1(1).LineWidth = 2;
    h1(2).LineWidth = 2;
    legend('TP9','TP10','Location','best')
    hold on
    
    K = Mdl.Coeffs(1,2).Const;  
    L = Mdl.Coeffs(1,2).Linear;
    
    f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
    h2 = fimplicit(f);
    h2.Color = 'r';
    h2.LineWidth = 2;
    h2.DisplayName = 'Boundary between Versicolor & Virginica';

    
%     % select time window
%     t1_ix = time_window_for_analysis(1) * ALLEEG(subject).srate / 1000;
%     t2_ix = time_window_for_analysis(2) * ALLEEG(subject).srate / 1000;
%     ixs = t1_ix:t2_ix;
%     ixs = 750 + ixs;
% 
%     % fitlm for each time frequency pixel
%     tic
%     disp(['now fitting data for subject: ' num2str(subject) ' and model: ' fit.model '...']);
%     for t = ixs
% 
%         % add matching velocity for this frame
%         velocity_this_frame = motion.mag_vel(t,:)';
% 
%         if fit.match_trial_count
%             velocity_this_frame = velocity_this_frame(matched_trials);
%         end
% 
%         % add ersp and baseline sample to design matrix
%         if size(comps,2) > 1
%             erp_sample = squeezemean(filtered_erp.comp(comps,t,:),1);
%         else
%             erp_sample = squeeze(filtered_erp.comp(comps,t,:));
%         end
% 
%         if fit.match_trial_count
%             erp_sample = erp_sample(matched_trials);
%         end
% 
%         reg_t = addvars(reg_t, erp_sample, velocity_this_frame);
% 
%         % fit model and save
%         if fit.log_regression
%             mdl = fitglm(reg_t, fit.model);
%             ypred = predict(mdl);
%             ypred(ypred>=.5) = 1;
%             ypred(ypred<.5) = 0;
%             fit.acc(subject,f,t-ixs(1)+1,:) = sum(ypred==reg_t.oddball) / size(reg_t.oddball,1);
%         else
%             if ~contains(fit.model, 'oddball') % mismatch trials only
%                 tmp_reg_t = reg_t;
%                 tmp_reg_t(tmp_reg_t .oddball==0,:) = [];
%                 mdl = fitlm(tmp_reg_t, fit.model);
%             else
%                 mdl = fitlm(reg_t, fit.model);
%             end
%         end
% 
%         fit.betas(subject,t-ixs(1)+1,:) = mdl.Coefficients.Estimate;
%         fit.t(subject,t-ixs(1)+1,:) = mdl.Coefficients.tStat;
%         fit.p(subject,t-ixs(1)+1,:) = mdl.Coefficients.pValue;
%         fit.r2(subject,t-ixs(1)+1,:) = mdl.Rsquared.Ordinary;
%         fit.adj_r2(subject,t-ixs(1)+1,:) = mdl.Rsquared.Adjusted;
% 
%         reg_t = removevars(reg_t, {'erp_sample', 'velocity_this_frame'});
%     end
%     toc
%     clear reg_t

end

fit.predictor_names = string(mdl.CoefficientNames);

out_folder = fullfile(bemobil_config.study_folder, 'results', ...
    ['cluster_ROI_' ...
    num2str(bemobil_config.STUDY_cluster_ROI_talairach(i).x) '_' ...
    num2str(bemobil_config.STUDY_cluster_ROI_talairach(i).y) '_' ...
    num2str(bemobil_config.STUDY_cluster_ROI_talairach(i).z)], ...        
    num2str(cluster), 'erp');

if ~exist(out_folder)
    mkdir(out_folder);
end

save(fullfile(out_folder, ...
    [fit.model '_base-shuffled-' num2str(fit.base_shuffled) '_matched-trial-count-' num2str(fit.match_trial_count) ...
    '_log-regression-' num2str(fit.log_regression) '.mat']), 'fit'); 

clear fit