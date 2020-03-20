%% params
PE_config;

% general
% get betas from single subject level of design:
models = {'ersp_sample ~ congruency * haptics + base',... % + base + trial_nr + direction + sequence
    'ersp_sample ~ velocity * haptics + base'}; % trial_nr + direction + sequence
robustfit = 0; % fit robust regression with squared weights, see fitlm
event_sample = 750;
log_scale = 0;

% what to plot
metric = 'ersp';
cluster = 22;
coeffs_to_plot = 3:5;
%parameters = {'(Intercept)', 'congruency_1', 'haptics_1', 'congruency_1_haptics_1'};
parameters = {'(Intercept)', 'velocity', 'haptics_1', 'rt', 'haptics_1_velocity'};

% paths
% remove baseline from the model
model = models{2};
if log_scale
    model = model(1:end-7);
end
load_p = ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/' ...
    metric '/' bemobil_config.study_filename(1:end-6) '/'];

% load data
load([load_p 'cluster_' num2str(cluster) '/' model '/res_' model '_robust-' num2str(robustfit) '.mat']);

% plot pERSPs per cluster

% subplot ix
s_ix = 1;

figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 800 1200]);

% plot mean ERSP -> sum all betas
subplot(5,1,s_ix);
s_ix = s_ix + 1;
data = squeezemean(sum(res.betas,4),1);
plotersp(res.times, res.freqs, data, [], [], 'frequency (Hz)', 'time in ms', 'mean ERSP', 'power');

% plot r^2
subplot(5,1,s_ix);
s_ix = s_ix + 1;
data = squeezemean(res.r2,1);
plotersp(times, freqs, data, [], [], 'frequency (Hz)', 'time in ms', 'R^2', []);

% plot coefficients
for coeff = coeffs_to_plot
    subplot(5,1,s_ix);
    s_ix = s_ix + 1;

    %data = squeezemean(res.betas(:,:,:,coeff),1);
    data = res.ttest.(parameters{coeff}).beta;
    p = res.ttest.(parameters{coeff}).tfce;
    alpha = res.ttest.(parameters{coeff}).thresh;
    %alpha = prctile(res.ttest.(parameters{coeff}).max_dist_600,95)

    figure;
    plotersp(times, freqs, data, p, alpha, 'frequency (Hz)', 'time in ms', parameters{coeff}, 'beta');
end

%tightfig;

%     saveas(gcf, [load_p 'cluster_' num2str(c) '/res_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(c) '.png'], 'png')
%     close(gcf);
