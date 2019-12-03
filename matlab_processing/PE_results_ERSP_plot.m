
%% params
PE_config;

% select subjects out of clusters of int
clusters_of_int = [3, 7, 9, 24, 25, 28, 30, 33, 34];
% clusters
% 3: right parietal
% 7: right motor?
% 24: right SMA
% 25: left parietal
% 28: interesting
% 33: ACC

% data settings
robustfit = 1;
model = 'ersp_sample ~ immersion * vel + trial + direction + sequence';
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);
seconds_before_event = .3;
samples_before_event = seconds_before_event * 250;
ts_of_ints = (event_onset-samples_before_event):25:event_onset+50;
% ts_of_ints = ts_of_ints - mocap_aos_samples;
ts_of_ints = ts_of_ints(4); % select best tf_of_ints
this_ts = (event_onset - ts_of_ints) / 250;
load_p = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/ersp/cluster_';

% set significance level for masking
alpha = .1;

%% plot pERSPs per cluster

% load times and freqs
load([load_p(1:end-8) 'times.mat']);
load([load_p(1:end-8) 'freqs.mat']);

for c = clusters_of_int

    % load data per cluster
    load([load_p num2str(c) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(c) '.mat']);
    
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1200 1800]);

    % plot mean ERSP -> sum all betas
    subplot(4,2,1);
    data = squeezemean(sum(res.t,4),1);
    plotersp(times, freqs, data, [], [], 'frequency (Hz)', 'time in ms', 'mean ERSP', 'dB');

    % plot r^2
    subplot(4,2,2);
    data = squeezemean(res.r2,1);
    plotersp(times, freqs, data, [], [], 'frequency (Hz)', 'time in ms', 'R^2', []);

    % plot haptics coefficient
    subplot(4,2,3);
    data = squeezemean(res.t(:,:,:,2),1);
    p = squeezemean(u.p(:,:,:,2),1);
    figure;
    plotersp(times, freqs, p, p, alpha, 'frequency (Hz)', 'time in ms', 'immersion coeff.', 'beta');
    
    % the problem is the clustering! very different results between
    % subjects
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1200 1800]);
    
    for s = 1:size(res.t,1)
        subplot(8,2,s)
        data = squeeze(res.t(s,:,:,2));
        p = squeeze(res.p(s,:,:,2));    
        plotersp(times, freqs, data, p, alpha,'frequency (Hz)', 'time in ms', 'immersion coeff.', 'beta');
    end

    % plot velocity coefficient
    subplot(4,2,4);
    data = squeezemean(res.t(:,:,:,3),1);
    p = squeezemean(res.p(:,:,:,3),1);
    plotersp(times, freqs, data, p, alpha, 'frequency (Hz)', 'time in ms', 'velocity coeff.', 'beta');

    % plot haptics * velocity coefficient (= interaction effect of haptic immersion and velocity)
    subplot(4,2,5);
    data = squeezemean(res.t(:,:,:,8),1);
    p = squeezemean(res.p(:,:,:,8),1);
    plotersp(times, freqs, data, p, alpha, 'frequency (Hz)', 'time in ms', 'velocity X immersion coeff.', 'beta');
    
    % trial
    subplot(4,2,6);
    data = squeezemean(res.t(:,:,:,4),1);
    p = squeezemean(res.p(:,:,:,4),1);
    plotersp(times, freqs, data, p, alpha, 'frequency (Hz)', 'time in ms', 'trial coeff.', 'beta');
    
    % sequence
    subplot(4,2,7);
    data = squeezemean(res.t(:,:,:,7),1);
    p = squeezemean(res.p(:,:,:,7),1);
    plotersp(times, freqs, data, p, alpha, 'frequency (Hz)', 'time in ms', 'sequence coeff.', 'beta');

    tightfig;
    
    saveas(gcf, [load_p num2str(c) '/res_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(c) '.png'], 'png')
    close(gcf);

end