%% params
PE_config;

% general
robustfit = 0; % fit robust regression with squared weights, see fitlm
event_sample = 750;
sample_erp_zero = 25;
window = event_sample-sample_erp_zero:event_sample+200; %[-.1 .8]seconds start and end of interesting, to be analyzed, samples
alpha = .05;

% what to plot
metric = 'ersp';
cluster = 26;
channel_name = 'ACC';
models = {'ersp_sample ~ congruency * haptics + trial_nr + direction + sequence + base', ...
    'ersp_sample ~ velocity * haptics + trial_nr + direction + sequence + base'};
model = models{2};

% paths
load_p = ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/' ...
    metric '/' bemobil_config.study_filename(1:end-6) '/'];

% load data
load([load_p 'cluster_' num2str(cluster) '/' model '/res_' model '_robust-' num2str(robustfit) '.mat']);

%% plot pERSPs per cluster

% load times and freqs
load([load_p(1:(end-(size(bemobil_config.study_filename,2))+5)) 'times.mat']);
load([load_p(1:(end-(size(bemobil_config.study_filename,2))+5)) 'times_all.mat']);
load([load_p(1:(end-(size(bemobil_config.study_filename,2))+5)) 'freqs.mat']);

% subplot ix
s_ix = 1;

% load data
load([load_p sensor '_' num2str(c) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event.mat']);

figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 2800 1200]);

% plot mean ERSP -> sum all betas
subplot(subplots(1),subplots(2),s_ix);
s_ix = s_ix + 1;
data = squeezemean(sum(res.betas,4),1);
data = data(1:max_freq_ix,first_ix:last_ix);
plotersp(times, freqs, data, [], [], 'frequency (Hz)', 'time in ms', 'mean ERSP', 'power');

% plot r^2
subplot(subplots(1),subplots(2),s_ix);
s_ix = s_ix + 1;
data = squeezemean(res.r2,1);
data = data(1:max_freq_ix,first_ix:last_ix);
plotersp(times, freqs, data, [], [], 'frequency (Hz)', 'time in ms', 'R^2', []);

% plot coefficients
for coeff = coeffs_to_plot
    subplot(subplots(1),subplots(2),s_ix);
    s_ix = s_ix + 1;

    data = squeezemean(res.betas(:,:,:,coeff),1);
    data = data(1:max_freq_ix,first_ix:last_ix);

    p = res.ttest.(parameters{coeff}).tfce_map;
    alpha = res.ttest.(parameters{coeff}).thresh;

    plotersp(times, freqs, data, p, alpha, 'frequency (Hz)', 'time in ms', coeffs_to_plot_names(coeff), 'beta');
end

%tightfig;

%     saveas(gcf, [load_p 'cluster_' num2str(c) '/res_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(c) '.png'], 'png')
%     close(gcf);
