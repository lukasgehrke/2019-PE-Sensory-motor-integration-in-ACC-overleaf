%% clear all and load params
close all; clear all; clc;

PE_config;

%% load study

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

%% RT

for subject = subjects
    
    subject = subject - 1;
    bad_trials = ALLEEG(subject).etc.analysis.design.rm_ixs;
    async_trials = ALLEEG(subject).etc.analysis.design.oddball;
    async_trials(bad_trials) = 0; % remove bad trials
    % match number of async trials for sync trials
    sync_trials = ~async_trials;
    sync_trials(bad_trials) = 0; % remove bad trials
    
    % sync
    rt_sync(subject,:) = mean(ALLEEG(subject).etc.analysis.design.rt_spawned_touched(sync_trials),2);
    % async
    rt_async(subject,:) = mean(ALLEEG(subject).etc.analysis.design.rt_spawned_touched(async_trials),2);
end

disp(['sync trials mean rt: ' num2str(mean(rt_sync,1)) ' and sd: ' num2str(std(rt_sync))]);
disp(['async trials mean rt: ' num2str(mean(rt_async,1)) ' and sd: ' num2str(std(rt_async))]);
disp(['rt difference: ' num2str((mean(rt_sync,1) - mean(rt_async,1))*1000)])

%% (DONE & FIGURES READY) grand average velocity sync/async

for subject = subjects
    
    % get trials
    subject = subject-1; % STUDY index is -1 as first subject data is missing
    bad_trials = ALLEEG(subject).etc.analysis.design.rm_ixs;
    async_trials = ALLEEG(subject).etc.analysis.design.oddball;
    async_trials(bad_trials) = 0; % remove bad trials
    % match number of async trials for sync trials
    sync_trials = ~async_trials;
    sync_trials(bad_trials) = 0; % remove bad trials
    sync_trials_ixs = randsample(find(sync_trials==1),sum(async_trials));
    sync_trials = logical(zeros(1,size(async_trials,2)));
    sync_trials(sync_trials_ixs)=1;
    
    % find ersp start and end times as anchors for vel plot
    t1 = ALLEEG(subject).etc.analysis.ersp.tf_event_times(1);
    tend = ALLEEG(subject).etc.analysis.ersp.tf_event_times(end);
    % match with times in velocity epoch
    times = 1000*(bemobil_config.epoching.event_epochs_boundaries(1):(1/ALLEEG(subject).srate):bemobil_config.epoching.event_epochs_boundaries(end));
    % find nearest element
    [~, t1_ix] = min(abs(times-t1));
    [~, tend_ix] = min(abs(times-tend));
    ixs = t1_ix:tend_ix;
    % find zero for plot
    [~, xline_zero] = min(abs(times-0));
    xline_zero = xline_zero - t1_ix;
    
    % sync. velocity, no significicance test
    sync(subject,:) = mean(ALLEEG(subject).etc.analysis.mocap.mag_vel(ixs,sync_trials),2);
    async(subject,:) = mean(ALLEEG(subject).etc.analysis.mocap.mag_vel(ixs,async_trials),2);
    
end

% prepare plot
normal; % plot normal window, not docked
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 960 100]);

% plot condition 1
colors = brewermap(5, 'Spectral');
colors1 = colors(2, :);
ploterp_lg(sync, [], [], xline_zero, 1, 'ERV m/s', '', [0 .8], colors1, '-');
hold on
% plot condition 2
colors2 = colors(5, :);
ploterp_lg(async, [], [], xline_zero, 1, '', '', [0 .8], colors2, '-.');
% add legend
legend('sync.','async.');

save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/vel_erp_sync_async/';
mkdir(save_path)
print(gcf, [save_path 'vel.eps'], '-depsc');
close(gcf);
clear sync async

%% (DONE & FIGURES READY) grand average channel ERP sync/async

t1 = -200; % -2444 (ersp start)
tend = 650; % 1440 (ersp end)
baset1 = -50;
baseend = 0;

for chan = 1:size(bemobil_config.channels_of_int,2)
    for subject = subjects
        % get trials
        subject = subject-1; % STUDY index is -1 as first subject data is missing
        bad_trials = ALLEEG(subject).etc.analysis.design.rm_ixs;
        trials = 1:size(ALLEEG(subject).etc.analysis.design.oddball,2);
        async_trials = ALLEEG(subject).etc.analysis.design.oddball;
        async_trials(bad_trials) = 0; % remove bad trials

        sync_trials = ~async_trials;
        sync_trials(bad_trials) = 0; % remove bad trials
        sync_trials = trials(sync_trials);
        sync_trials = randsample(sync_trials, sum(async_trials));
        
        % match with times in velocity epoch
        times = 1000*(bemobil_config.epoching.event_epochs_boundaries(1):(1/ALLEEG(subject).srate):bemobil_config.epoching.event_epochs_boundaries(end));
        % find nearest element
        [~, t1_ix] = min(abs(times-t1));
        [~, tend_ix] = min(abs(times-tend));
        ixs = t1_ix:tend_ix;
        % find zero for plot
        [~, xline_zero] = min(abs(times-0));
        xline_zero = xline_zero - t1_ix;

        % get data
        sync(subject,:) = squeezemean(ALLEEG(subject).etc.analysis.filtered_erp.chan(chan,ixs,sync_trials),3);
        async(subject,:) = squeezemean(ALLEEG(subject).etc.analysis.filtered_erp.chan(chan,ixs,async_trials),3);
        
        % base correct
        sync(subject,:) = sync(subject,:) - mean(sync(subject,xline_zero-12:xline_zero),2); % 12.5 sample is 50 ms with 250 Hz EEG.srate
        async(subject,:) = async(subject,:) - mean(async(subject,xline_zero-12:xline_zero),2); % 12.5 sample is 50 ms with 250 Hz EEG.srate
    end
    
    % statistics: permutation t-test
    [~, ~, p_vals, ~] = statcond({permute(sync, [2,1]) permute(async,[2,1])},...
        'method', 'perm', 'naccu', 1000);
    
    % prepare plot
    normal; % plot normal window, not docked
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 300 200]);
    title(bemobil_config.channels_of_int_labels{chan});
    colors = brewermap(5, 'Spectral');
    % plot condition 1
    colors1 = colors(2, :);
    ploterp_lg(sync, [], [], xline_zero, 1, 'ERP \muV', '', '', colors1, '-');
    hold on
    % plot condition 2 and add sigmask
    colors2 = colors(5, :);
    ploterp_lg(async, p_vals, .05, xline_zero, 1, '', '', '', colors2, '-.');

    % save & clear
    save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/chan_erp_sync_async/';
    mkdir(save_path)
    savefig([save_path bemobil_config.channels_of_int_labels{chan}]);
    print(gcf, [save_path bemobil_config.channels_of_int_labels{chan} '.eps'], '-depsc');
    close(gcf);
    clear sync async
end

%% (DONE & FIGURES READY) grand average component ERP sync/async

t1 = -2444; % -2444 (ersp start)
tend = 1440; % 1440 (ersp end)
baset1 = -50;
baseend = 0;

% load clustering solution
cluster_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing/';
load([cluster_path 'clustering_vaa.mat']); % clustering_parietal
STUDY.cluster = clustering_results_STUDY_vaa; % clustering_parietal
cluster = 9; % 8

unique_setindices = unique(STUDY.cluster(cluster).sets);
unique_subjects = STUDY_sets(unique_setindices);
all_setindices = STUDY.cluster(cluster).sets;
all_sets = STUDY_sets(all_setindices);
all_comps = STUDY.cluster(cluster).comps;
for subject = unique_subjects

    % select components
    this_sets = find(all_sets==subject);
    comp = all_comps(this_sets);

    % get trials
    subject = subject-1; % STUDY index is -1 as first subject data is missing
    bad_trials = ALLEEG(subject).etc.analysis.design.rm_ixs;
    trials = 1:size(ALLEEG(subject).etc.analysis.design.oddball,2);
    async_trials = ALLEEG(subject).etc.analysis.design.oddball;
    async_trials(bad_trials) = 0; % remove bad trials

    sync_trials = ~async_trials;
    sync_trials(bad_trials) = 0; % remove bad trials
    sync_trials = trials(sync_trials);
    sync_trials = randsample(sync_trials, sum(async_trials)); % match number of trials

    % match with times in velocity epoch
    times = 1000*(bemobil_config.epoching.event_epochs_boundaries(1):(1/ALLEEG(subject).srate):bemobil_config.epoching.event_epochs_boundaries(end));
    % find nearest element
    [~, t1_ix] = min(abs(times-t1));
    [~, tend_ix] = min(abs(times-tend));
    ixs = t1_ix:tend_ix;
    % find zero for plot
    [~, xline_zero] = min(abs(times-0));
    xline_zero = xline_zero - t1_ix;

    % sync. velocity, no significicance test
    sync_data = squeeze(ALLEEG(subject).etc.analysis.filtered_erp.comp(comp,ixs,sync_trials));
    async_data = squeeze(ALLEEG(subject).etc.analysis.filtered_erp.comp(comp,ixs,async_trials));
    if size(comp,2)>1
        sync_data = squeezemean(sync_data,1);
        async_data = squeezemean(async_data,1);
    end
    sync(subject+1,:) = squeezemean(sync_data,2); % first subject is missing
    async(subject+1,:) = squeezemean(async_data,2);
    
    % subtract baseline?
    sync(subject+1,:) = sync(subject+1,:) - mean(sync(subject+1,xline_zero-12:xline_zero),2); % 12.5 sample is 50 ms with 250 Hz EEG.srate
    async(subject+1,:) = async(subject+1,:) - mean(async(subject+1,xline_zero-12:xline_zero),2); % 12.5 sample is 50 ms with 250 Hz EEG.srate
    
end
sync = sync(unique_subjects,:); % remove missing subjects
async = async(unique_subjects,:);

% statistics: permutation t-test
[~, ~, p_vals, ~] = statcond({permute(sync, [2,1]) permute(async,[2,1])},...
    'method', 'perm', 'naccu', 10000);

% prepare plot
normal; % plot normal window, not docked
%figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 300 200]);
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 960 200]);
title(num2str(cluster));
% plot condition 1
colors = brewermap(5, 'Spectral');
colors1 = colors(2, :);
ploterp_lg(sync, [], [], xline_zero, 1, 'ERP \muV', '', '', colors1, '-');
hold on
% plot condition 2
colors2 = colors(5, :);
ploterp_lg(async, p_vals, .05, xline_zero, 1, '', '', '', colors2, '-.');

% save
save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/comp_erp_sync_async/';
mkdir(save_path);
savefig([save_path num2str(cluster) '_win_' num2str(t1) '-' num2str(tend)]);
print(gcf, [save_path num2str(cluster) '_win_' num2str(t1) '-' num2str(tend) '.eps'], '-depsc');
close(gcf);

clear sync async

%% (DONE & FIGURES READY) grand average ERSP sync and async (and diff)

% load clustering solution
cluster_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing/';
%load([cluster_path 'clustering_vaa.mat']); % clustering_parietal
%STUDY.cluster = clustering_results_STUDY_vaa; % clustering_parietal
%cluster = 9; % parietal = 8, vaa = 9

unique_setindices = unique(STUDY.cluster(cluster).sets);
unique_subjects = STUDY_sets(unique_setindices);
all_setindices = STUDY.cluster(cluster).sets;
all_sets = STUDY_sets(all_setindices);
all_comps = STUDY.cluster(cluster).comps;
for subject = unique_subjects

    % select components
    this_sets = find(all_sets==subject);
    comp = all_comps(this_sets);

    % get trials
    subject = subject-1; % STUDY index is -1 as first subject data is missing
    bad_trials = ALLEEG(subject).etc.analysis.design.rm_ixs;
    async_trials = ALLEEG(subject).etc.analysis.design.oddball;
    async_trials(bad_trials) = 0; % remove bad trials
    sync_trials = ~async_trials;
    sync_trials(bad_trials) = 0; % remove bad trials

    % get data
    sync_event = squeeze(ALLEEG(subject).etc.analysis.ersp.tf_event_raw_power(comp,:,:,sync_trials));
    sync_base = squeeze(ALLEEG(subject).etc.analysis.ersp.tf_base_raw_power(comp,:,sync_trials));
    async_event = squeeze(ALLEEG(subject).etc.analysis.ersp.tf_event_raw_power(comp,:,:,async_trials));
    async_base = squeeze(ALLEEG(subject).etc.analysis.ersp.tf_base_raw_power(comp,:,async_trials));

    % if more than one component, average component power
    if size(comp,2)>1
        sync_event = squeezemean(sync_event,1);
        sync_base = squeezemean(sync_base,1);
        async_event = squeezemean(async_event,1);
        async_base = squeezemean(async_base,1);
    end

    % average sync. and async. ERSP, significance against baseline = baseline
    % corrected ersp one-sample limo ttest, tfce thresh
    grand_avg.ersp.sync.event(subject,:,:) = squeezemean(sync_event,3);
    grand_avg.ersp.sync.base(subject,:) = squeezemean(sync_base,2)';
    grand_avg.ersp.async.event(subject,:,:) = squeezemean(async_event,3);
    grand_avg.ersp.async.base(subject,:) = squeezemean(async_base,2)';

    % single subject db transformed, this is not correct but good for
    % checking
    grand_avg.ersp.sync.base_corrected(subject,:,:) = 10.*log10(grand_avg.ersp.sync.event(subject,:,:) ./ grand_avg.ersp.sync.base(subject,:));
    grand_avg.ersp.async.base_corrected(subject,:,:) = 10.*log10(grand_avg.ersp.async.event(subject,:,:) ./ grand_avg.ersp.async.base(subject,:));
end

% remove missing subjects
grand_avg.ersp.sync.event = grand_avg.ersp.sync.event(unique_subjects-1,:,:);
grand_avg.ersp.sync.base = grand_avg.ersp.sync.base(unique_subjects-1,:);
grand_avg.ersp.async.event = grand_avg.ersp.async.event(unique_subjects-1,:,:);
grand_avg.ersp.async.base = grand_avg.ersp.async.base(unique_subjects-1,:);

% statistics asynchronous trials permutation t-test
[stats, df, p_vals, ~] = statcond({permute(grand_avg.ersp.async.event, [2,3,1]) permute(repmat(grand_avg.ersp.async.base, [1,1,size(grand_avg.ersp.async.event,3)]),[2,3,1])},...
    'method', 'perm', 'naccu', 1000);

% plot
asy = 10.*log10(squeezemean(grand_avg.ersp.async.event,1) ./ squeezemean(grand_avg.ersp.async.base,1)');
normal;figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 960 300]);
plotersp(ALLEEG(subject).etc.analysis.ersp.tf_event_times, ALLEEG(subject).etc.analysis.ersp.tf_event_freqs,...
    asy, p_vals, .05, 'frequency (Hz)', 'time (ms)', 'asynchrony', 'dB', 1);

% extract stats
% 1. theta spawn
t = 166; % -300ms
ALLEEG(subject).etc.analysis.ersp.tf_event_times(t)
f = 11; % 5 Hz
ALLEEG(subject).etc.analysis.ersp.tf_event_freqs(f)
stats(f,t)
df
p_vals(f,t)

% 2. theta object
t = 210; % 270ms
ALLEEG(subject).etc.analysis.ersp.tf_event_times(t)
f = 8; % 4.5 Hz
ALLEEG(subject).etc.analysis.ersp.tf_event_freqs(f)
stats(f,t)
df
p_vals(f,t)

% 3. alpha movement
t = 190; % 0ms
ALLEEG(subject).etc.analysis.ersp.tf_event_times(t)
f = 23; % 10 Hz
ALLEEG(subject).etc.analysis.ersp.tf_event_freqs(f)
stats(f,t)
df
p_vals(f,t)

% 4. beta movement
t = 190; % 0ms
ALLEEG(subject).etc.analysis.ersp.tf_event_times(t)
f = 36; % 21 Hz
ALLEEG(subject).etc.analysis.ersp.tf_event_freqs(f)
stats(f,t)
df
p_vals(f,t)

% 5. beta pre-movement
t = 19; % 0ms
ALLEEG(subject).etc.analysis.ersp.tf_event_times(t)
f = 36; % 21 Hz
ALLEEG(subject).etc.analysis.ersp.tf_event_freqs(f)
stats(f,t)
df
p_vals(f,t)

% 5. beta post-movement
t = 274; % 0ms
ALLEEG(subject).etc.analysis.ersp.tf_event_times(t)
f = 36; % 21 Hz
ALLEEG(subject).etc.analysis.ersp.tf_event_freqs(f)
stats(f,t)
df
p_vals(f,t)

%clear grand_avg p_vals

% %sy = 10.*log10(squeezemean(grand_avg.ersp.sync.event,1) ./ squeezemean(grand_avg.ersp.sync.base,1)');
% %figure; imagesc(sy, [-1 1]); axis xy; cbar;
% sy = squeezemean(grand_avg.ersp.sync.base_corrected,1);
% figure; 
% sgtitle(['cluster ' num2str(cluster)]);
% subplot(3,1,1);
% imagesclogy(ALLEEG(subject).etc.analysis.ersp.tf_event_times, ALLEEG(subject).etc.analysis.ersp.tf_event_freqs, sy, [-.5 .5]); axis xy; xline(0); cbar;
% %asy = 10.*log10(squeezemean(grand_avg.ersp.async.event,1) ./ squeezemean(grand_avg.ersp.async.base,1)');
% %figure; imagesc(asy, [-1 1]); axis xy; cbar;
% asy = squeezemean(grand_avg.ersp.async.base_corrected,1);
% subplot(3,1,2);
% imagesclogy(ALLEEG(subject).etc.analysis.ersp.tf_event_times, ALLEEG(subject).etc.analysis.ersp.tf_event_freqs, asy, [-.5 .5]); axis xy; xline(0); cbar;
% diff = sy - asy;
% subplot(3,1,3);
% imagesclogy(ALLEEG(subject).etc.analysis.ersp.tf_event_times, ALLEEG(subject).etc.analysis.ersp.tf_event_freqs, diff, [-.5 .5]); axis xy; xline(0); cbar;

%% statistics synchrony vs. asynchrony, this is a question whether to do it or not???
% vel, 2 grand averages sync. and async., two-sample limo ttest tfce thresh
% ersp, betas async., one-sample ttest

%% (DONE & FIGURE READY) VEL statistics ASYNC ONLY, haptics and trial nr.
% trial number to show there is no change over time in learning the trial
% haptics to show whether it impacts stopping

clear res
res.t1 = -2444; % -2444 (ersp start)
res.xline_zero = 750;
res.tend = 1440; % 1440 (ersp end)
% single-trial model fitting
res.model = 'erv_sample ~ haptics * trial_nr';
for subject = subjects
    
    % get trials
    subject = subject-1; % STUDY index is -1 as first subject data is missing in this study
    % match with times in velocity epoch
    times = 1000*(bemobil_config.epoching.event_epochs_boundaries(1):(1/ALLEEG(subject).srate):bemobil_config.epoching.event_epochs_boundaries(end));
    % find nearest element
    [~, res.t1_ix] = min(abs(times-res.t1));
    [~, res.tend_ix] = min(abs(times-res.tend));
    res.ixs = res.t1_ix:res.tend_ix;
    
    % prepare design
    synchrony = ALLEEG(subject).etc.analysis.design.oddball';
    haptics = ALLEEG(subject).etc.analysis.design.haptics';
    trial_nr = ALLEEG(subject).etc.analysis.design.trial_number';
    direction = categorical(ALLEEG(subject).etc.analysis.design.direction)';
    sequence = ALLEEG(subject).etc.analysis.design.sequence';
    velocity = ALLEEG(subject).etc.analysis.mocap.mag_vel(res.xline_zero,:)';
    rt = ALLEEG(subject).etc.analysis.design.rt_spawned_touched';
    
    % boolean to calculate correlation once
    corr_rt_vel = 1;
    
    tic
    disp(['now fitting data for subject: ' num2str(subject) ' and model: ' res.model '...']);
    for ix = res.ixs

        % get sample
        erv_sample = ALLEEG(subject).etc.analysis.mocap.mag_vel(ix,:)';
        % design matrix
        design = table(erv_sample, synchrony, haptics, trial_nr, direction, sequence, velocity, rt);
        % remove bad trials
        design(ALLEEG(subject).etc.analysis.design.rm_ixs,:) = [];
        % select only async
        async = design.synchrony == 1;
        design = design(async,:);
        
        % fit model
        mdl = fitlm(design, res.model);
        
        save_ix = ix-res.ixs(1)+1;
        res.betas(subject,save_ix,:) = mdl.Coefficients.Estimate;
        res.t(subject,save_ix,:) = mdl.Coefficients.tStat;
        res.p(subject,save_ix,:) = mdl.Coefficients.pValue;
        res.r2(subject,save_ix,:) = mdl.Rsquared.Ordinary;
        res.adj_r2(subject,save_ix,:) = mdl.Rsquared.Adjusted;
    end
    disp(['time elapsed to fit data for subject: ' num2str(subject) ' -> ' num2str(toc) ' seconds...']);
end

% statistics
res.predictor_names = string(mdl.CoefficientNames);
res.save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/results';
res.alpha = .05;
res.perm = 1000;
for i = 2:size(res.predictor_names,2)
    
    % get betas per predictor
    betas = res.betas(:,:,i);
    betas = permute(betas, [2, 1]);
    zero = zeros(size(betas));

    % permutation t-test
    [~, ~, res.stats(i).betas_p_vals, res.stats(i).surrogate_data] = statcond({betas zero},...
        'method', 'perm', 'naccu', res.perm);

    % compute tfce transform of t_maps surrogate data, add max tfce
    % dist
    for j = 1:size(res.stats(i).surrogate_data,2)
        tfce(j,:) = limo_tfce(1,res.stats(i).surrogate_data(:,j)',[],0);
        this_max = tfce(j,:);
        res.stats(i).tfce_max_dist(j) = max(this_max(:));
    end

    % threshold true t_map
    [~,~,~,STATS] = ttest(permute(betas, [2,1]));
    res.stats(i).perm_t = STATS;
    res.stats(i).tfce_true = limo_tfce(1,STATS.tstat,[],0);
    res.stats(i).tfce_thresh = prctile(res.stats(i).tfce_max_dist,95);
    res.stats(i).tfce_sig_mask = res.stats(i).tfce_true>res.stats(i).tfce_thresh;
end

% resave res
save([res.save_path '/res_' res.model '_' num2str(res.t1) '-' num2str(res.tend) '.mat'], 'res');

% effect X at sample Y
effect = 3;
i = 635;
res.stats(effect).perm_t.tstat(i)
res.stats(effect).perm_t.df(i)
res.stats(effect).betas_p_vals(i)
mean(res.r2(:,i),1)

% % plot
% normal; % plot normal window, not docked
% figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 960 150]);
% colors = brewermap(5, 'Spectral');
% xline_zero = 750-res.t1_ix;
% % condition 1 normal
% colors1 = colors(2, :);
% cond1 = squeeze(res.betas(:,:,1));
% ploterp_lg(cond1, [], [], xline_zero, 1, 'ERV m/s', '', [0 .8], colors1, '-');
% hold on
% % condition 2 following async
% colors2 = colors(5, :);
% cond2 = squeeze(sum(res.betas(:,:,[1,2]),3));
% ploterp_lg(cond2, res.stats(2).betas_p_vals, .05, xline_zero, 1, '', '', [0 .8], colors2, '-.');
% % add legend
% legend('haptic','no haptic');
% % save plot
% print(gcf, [res.save_path(1:end-7) 'figures/vel_haptic.eps'], '-depsc');
% close(gcf);

%% (DONE & FIGURE READY) VEL post asynchrony detection adaptation / discussion

% single-trial model fitting
clear res
res.model = 'erv_sample ~ following_haptics * following_asynchrony';
for subject = subjects
    
    % get trials
    subject = subject-1; % STUDY index is -1 as first subject data is missing in this study
    % find ersp end times for window for vel measures
    tend = ALLEEG(subject).etc.analysis.ersp.tf_event_times(end);
    tstart = ALLEEG(subject).etc.analysis.ersp.tf_event_times(1);
    % match with times in velocity epoch
    times = 1000*(bemobil_config.epoching.event_epochs_boundaries(1):(1/ALLEEG(subject).srate):bemobil_config.epoching.event_epochs_boundaries(end));
    % find nearest element
    [~, tend_ix] = min(abs(times-tend));
    [~, t1_ix] = min(abs(times-tstart));
    %xline_zero = 750; % 750 is sample 0 in epoch [-3 2] with 250 srate
    %ixs = xline_zero:tend_ix;
    ixs = t1_ix:tend_ix;
    
    % prepare design
    synchrony = ALLEEG(subject).etc.analysis.design.oddball';
    haptics = ALLEEG(subject).etc.analysis.design.haptics';
    trial_nr = ALLEEG(subject).etc.analysis.design.trial_number';
    direction = categorical(ALLEEG(subject).etc.analysis.design.direction)';
    sequence = ALLEEG(subject).etc.analysis.design.sequence';
    velocity = ALLEEG(subject).etc.analysis.mocap.mag_vel(xline_zero,:)';
    rt = ALLEEG(subject).etc.analysis.design.rt_spawned_touched';
    
    % add predictors
    following_haptics = [0; haptics(1:end-1)];
    following_asynchrony = [0; synchrony(1:end-1)];
    
    % boolean to calculate correlation once
    corr_rt_vel = 1;
    
    tic
    disp(['now fitting data for subject: ' num2str(subject) ' and model: ' res.model '...']);
    for ix = ixs

        % get sample
        erv_sample = ALLEEG(subject).etc.analysis.mocap.mag_vel(ix,:)';
        % design matrix
        design = table(erv_sample, synchrony, haptics, trial_nr, direction, sequence, velocity, rt, following_haptics, following_asynchrony);
        % remove bad trials
        design(ALLEEG(subject).etc.analysis.design.rm_ixs,:) = [];
        
        % fit model
        mdl = fitlm(design, res.model);
        
        save_ix = ix-ixs(1)+1;
        res.betas(subject,save_ix,:) = mdl.Coefficients.Estimate;
        res.t(subject,save_ix,:) = mdl.Coefficients.tStat;
        res.p(subject,save_ix,:) = mdl.Coefficients.pValue;
        res.r2(subject,save_ix,:) = mdl.Rsquared.Ordinary;
        res.adj_r2(subject,save_ix,:) = mdl.Rsquared.Adjusted;
    end
    disp(['time elapsed to fit data for subject: ' num2str(subject) ' -> ' num2str(toc) ' seconds...']);
end

% statistics
res.predictor_names = string(mdl.CoefficientNames);
res.save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/results';
res.alpha = .05;
res.perm = 1000;
for i = 2:size(res.predictor_names,2)
    
    % get betas per predictor
    betas = res.betas(:,:,i);
    betas = permute(betas, [2, 1]);
    zero = zeros(size(betas));

    % permutation t-test
    [~, ~, res.stats(i).betas_p_vals, res.stats(i).surrogate_data] = statcond({betas zero},...
        'method', 'perm', 'naccu', res.perm);

    % compute tfce transform of t_maps surrogate data, add max tfce
    % dist
    for j = 1:size(res.stats(i).surrogate_data,2)
        tfce(j,:) = limo_tfce(1,res.stats(i).surrogate_data(:,j)',[],0);
        this_max = tfce(j,:);
        res.stats(i).tfce_max_dist(j) = max(this_max(:));
    end

    % threshold true t_map
    [~,~,~,STATS] = ttest(permute(betas, [2,1]));
    res.stats(i).tfce_true = limo_tfce(1,STATS.tstat,[],0);
    res.stats(i).tfce_thresh = prctile(res.stats(i).tfce_max_dist,95);
    res.stats(i).tfce_sig_mask = res.stats(i).tfce_true>res.stats(i).tfce_thresh;
end

% resave res
save([res.save_path '/res_' res.model '.mat'], 'res');

% prepare plot
normal; % plot normal window, not docked
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 960 200]);
colors = brewermap(5, 'Spectral');
xline_zero = 750-t1_ix;
% condition 1 normal
colors1 = colors(2, :);
cond1 = squeeze(res.betas(:,:,1));
ploterp_lg(cond1, [], [], xline_zero, 1, 'ERV m/s', '', [0 .8], colors1, '-');
hold on
% condition 2 following async
colors2 = colors(5, :);
cond2 = squeeze(sum(res.betas(:,:,[1,3]),3));
ploterp_lg(cond2, res.stats(3).betas_p_vals, .05, xline_zero, 1, '', '', [0 .8], colors2, '-.');
% add legend
legend('following sync.','follwing async.');
% save plot
print(gcf, [res.save_path(1:end-7) '/figures/vel_post_async.eps'], '-depsc');
close(gcf);

%% (DONE & FIGURE READY) ERSP statistics async only
% ersp ~ vel_at_col * haptics + rt + base
% velocity: does it impact multisensory integration during spatio-temporal binding
% haptics: same as velocity, does object rigidity further perturb spatio-temporal binding
% rt: is there a task clock? self initiated movement timing?
% base: what activity at baseline impacts activity at spatio-temporal binding event -> that part of the activity is not processing related

% single-trial model fitting
res.model = 'ersp_sample ~ velocity * haptics + rt + base';
zero = 750;
test_multicoll = 1;
t1 = -400; % -2444 (ersp start)
tend = 1200; % 1440 (ersp end)

% load clustering solution
cluster_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing/';
load([cluster_path 'clustering_parietal.mat']); % clustering_vaa
STUDY.cluster = clustering_parietal; % clustering_results_STUDY_vaa
cluster = 8; % vaa = 9

% get matching datasets from EEGLAB Study struct
unique_setindices = unique(STUDY.cluster(cluster).sets);
unique_subjects = STUDY_sets(unique_setindices);
all_setindices = STUDY.cluster(cluster).sets;
all_sets = STUDY_sets(all_setindices);
all_comps = STUDY.cluster(cluster).comps;

count = 1;
for subject = unique_subjects

    % select components
    this_sets = find(all_sets==subject);
    comps = all_comps(this_sets);

    % prepare design
    subject = subject-1; % STUDY index is -1 as first subject data is missing
    asynchrony = ALLEEG(subject).etc.analysis.design.oddball';
    haptics = zscore(ALLEEG(subject).etc.analysis.design.haptics)'; % zscore
    trial_nr = ALLEEG(subject).etc.analysis.design.trial_number';
    direction = categorical(ALLEEG(subject).etc.analysis.design.direction)';
    sequence = ALLEEG(subject).etc.analysis.design.sequence';
    velocity = zscore(ALLEEG(subject).etc.analysis.mocap.mag_vel(zero,:))'; % zscore
    rt = ALLEEG(subject).etc.analysis.design.rt_spawned_touched';

    % find nearest element
    [~, t1_ix] = min(abs(ALLEEG(subject).etc.analysis.ersp.tf_event_times-(t1)));
    [~, tend_ix] = min(abs(ALLEEG(subject).etc.analysis.ersp.tf_event_times-tend));
    ixs = t1_ix:tend_ix;

    % fitlm for each time frequency pixel
    tic
    disp(['now fitting data for subject: ' num2str(subject) ' and model: ' res.model '...']);
    for t = ixs
        for f = 1:size(ALLEEG(subject).etc.analysis.ersp.tf_event_freqs,2)
            % add ersp and baseline sample to design matrix
            if size(comps,2) > 1
                ersp_sample = squeezemean(ALLEEG(subject).etc.analysis.ersp.tf_event_raw_power(comps,f,t,:),1);
                base = squeezemean(ALLEEG(subject).etc.analysis.ersp.tf_base_raw_power(comps,f,:),1);
            else
                ersp_sample = squeeze(ALLEEG(subject).etc.analysis.ersp.tf_event_raw_power(comps,f,t,:));
                base = squeeze(ALLEEG(subject).etc.analysis.ersp.tf_base_raw_power(comps,f,:));
            end

            design = table(ersp_sample, base, asynchrony, haptics, trial_nr, direction, sequence, velocity, rt);    

            % remove bad trials
            design(ALLEEG(subject).etc.analysis.design.rm_ixs,:) = [];
            % select only asynchronous
            design(design.asynchrony==0,:) = [];
            % test multicolinearity
            if test_multicoll
                res.corr.rt_vel(subject) = corr(rt, velocity);
                test_multicoll = 0;
            end

            % fit model and save
            mdl = fitlm(design, res.model);
            res.betas(count,f,t-ixs(1)+1,:) = mdl.Coefficients.Estimate;
            res.t(count,f,t-ixs(1)+1,:) = mdl.Coefficients.tStat;
            res.p(count,f,t-ixs(1)+1,:) = mdl.Coefficients.pValue;
            res.r2(count,f,t-ixs(1)+1,:) = mdl.Rsquared.Ordinary;
            res.adj_r2(count,f,t-ixs(1)+1,:) = mdl.Rsquared.Adjusted;
        end
    end
    toc
    count = count + 1;
    test_multicoll = 1;
end

% group-level statistics: settings
res.predictor_names = string(mdl.CoefficientNames);
res.times = ALLEEG(subject).etc.analysis.ersp.tf_event_times(ixs);
res.freqs = ALLEEG(subject).etc.analysis.ersp.tf_event_freqs;
res.save_path = ['/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/results/cluster_' num2str(cluster) '/async_trial/'];
res.alpha = .05;
res.perm = 1000;
for i = 2:size(res.predictor_names,2)

    % get betas per predictor
    betas = res.betas(:,:,:,i);
    betas = permute(betas, [2, 3, 1]);
    zero = zeros(size(betas));

    % permutation t-test
    [res.stats(i).t_stats, ~, res.stats(i).betas_p_vals, res.stats(i).surrogate_data] = statcond({betas zero},...
        'method', 'perm', 'naccu', res.perm);

    % compute tfce transform of t_maps surrogate data, add max tfce
    % dist
    for j = 1:size(res.stats(i).surrogate_data,3)
        tfce(j,:,:) = limo_tfce(2,squeeze(res.stats(i).surrogate_data(:,:,j)),[],0);
        this_max = tfce(j,:,:);
        res.stats(i).tfce_max_dist(j) = max(this_max(:));
    end

    % threshold true t_map
    [~,~,~,STATS] = ttest(permute(betas, [3, 1, 2]));
    res.stats(i).tfce_true = limo_tfce(2,squeeze(STATS.tstat),[],0);
    res.stats(i).tfce_thresh = prctile(res.stats(i).tfce_max_dist,95);
    res.stats(i).tfce_sig_mask = res.stats(i).tfce_true>res.stats(i).tfce_thresh;
end

% save res
save([res.save_path 'res_' res.model '.mat'], 'res');

% plot
normal; % plot normal window, not docked
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1400 300]);

measures = 2:6;
c = 1;
for measure = measures
    subplot(1,size(measures,2),measure-1);
    to_plot = squeezemean(res.betas(:,:,:,measure),1);
    p = res.stats(measure).betas_p_vals;

    %figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 500 300]);
    plotersp(res.times, res.freqs, to_plot, p, .05, 'frequency (Hz)', 'time (ms)', 'asynchrony', 'dB', 1);
end

% save plot
%print(gcf, [res.save_path 'st_betas_' num2str(measure) '.eps'], '-depsc');
%close(gcf);

% extract stats

% 1. alpha baseline post event
effect = 2;
t = 48; % 200 ms
res.times(t)
f = 21; % 9 Hz
res.freqs(f)
res.stats(effect).t_stats(f,t)
df
res.stats(effect).betas_p_vals(f,t)

% 2. alpha rt post event
effect = 5;
t = 52; % 250 ms
res.times(t)
f = 26; % 12 Hz
res.freqs(f)
res.stats(effect).t_stats(f,t)
df
res.stats(effect).betas_p_vals(f,t)

% 3. alpha rt pre event
effect = 5;
t = 16; % -200 ms
res.times(t)
f = 16; % 12 Hz
res.freqs(f)
res.stats(effect).t_stats(f,t)
df
res.stats(effect).betas_p_vals(f,t)

% 4. alpha haptics post event
effect = 3;
t = 46; % 180 ms
res.times(t)
f = 11; % 5 Hz
res.freqs(f)
res.stats(effect).t_stats(f,t)
df
res.stats(effect).betas_p_vals(f,t)

% 5. alpha haptics post event
effect = 3;
t = 46; % 180 ms
res.times(t)
f = 26; % 12 Hz
res.freqs(f)
res.stats(effect).t_stats(f,t)
df
res.stats(effect).betas_p_vals(f,t)

% 6. alpha velocity pre event
effect = 4;
t = 16; % 180 ms
res.times(t)
f = 26; % 12 Hz
res.freqs(f)
res.stats(effect).t_stats(f,t)
df
res.stats(effect).betas_p_vals(f,t)

% 7. theta interaction post event
effect = 6;
t = 46; % 180 ms
res.times(t)
f = 6; % 12 Hz
res.freqs(f)
res.stats(effect).t_stats(f,t)
df
res.stats(effect).betas_p_vals(f,t)

%% correlation post error vel and prior ERSP
