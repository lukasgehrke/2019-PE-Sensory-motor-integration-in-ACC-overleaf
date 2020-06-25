% create figure 1
normal; % plot normal window, not docked
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 960 400]);

%% plot velocity profile of sync and async trials
% with loaded study run
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
subplot(3,1,1);
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

%% plot vel. with haptics vs without

subplot(3,1,2);
load('/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/results/res_erv_sample ~ haptics * trial_nr_-2444-1440.mat')
colors = brewermap(5, 'Spectral');
xline_zero = 750-t1_ix;
% condition 1 normal
colors1 = colors(2, :);
cond1 = squeeze(res.betas(:,:,1));
ploterp_lg(cond1, [], [], xline_zero, 1, 'ERV m/s', '', [0 .8], colors1, '-');
hold on
% condition 2 following async
colors2 = colors(5, :);
cond2 = squeeze(sum(res.betas(:,:,[1,2]),3));
ploterp_lg(cond2, res.stats(2).betas_p_vals, .05, xline_zero, 1, '', '', [0 .8], colors2, '-.');
% add legend
legend('haptic','no haptic');

%% plot vel. following sync. vs async.

subplot(3,1,3);
load('/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/results/res_erv_sample ~ following_haptics * following_asynchrony.mat')
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