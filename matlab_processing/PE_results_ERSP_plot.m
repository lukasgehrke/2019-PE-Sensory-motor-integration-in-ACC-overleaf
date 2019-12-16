%% params
PE_config;

robustfit = 0;

% try out several ts prior to event
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);
seconds_before_event = .3;
samples_before_event = seconds_before_event * 250;
ts_of_ints = (event_onset-samples_before_event):25:event_onset+50;

% % shift due to EEG age of sample
% mocap_aos = .011; % age of sample mocap samples = 11 ms
% mocap_aos_samples = ceil(mocap_aos * 250);
% ts_of_ints = ts_of_ints - mocap_aos_samples;

% select best tf_of_ints
ts_of_ints = ts_of_ints(4);
this_ts = (event_onset - ts_of_ints) / 250;

ersp_type = 'non_corrected';

% colors
cols = brewermap(2, 'Spectral');

% load
load_p = ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/ersp/' bemobil_config.study_filename(1:end-6) '/'];
model = 'ersp_sample ~ immersion * vel + trial + direction + sequence';

zero = 3*250; % [-3 2] epoch around event    
event_win_samples = zero + (bemobil_config.epoching.event_win(1) * 250):zero+(bemobil_config.epoching.event_win(2) * 250);
event_win_samples = event_win_samples - event_onset;
event_win_times = event_win_samples / 250;
event_0_ix = find(event_win_times<0, 1, 'last');
event_0 = 0;

alpha = .05;
coeffs_to_plot_names = {'', 'immersion', 'velocity', 'trial number', '', '', 'sequence', 'vel. x immersion'};
parameters = {'', 'immersion_1', 'vel', 'trial', '', '', 'sequence', 'immersion_1_vel'};

%% plot pERSPs per cluster

% load times and freqs
load([load_p(1:(end-(size(bemobil_config.study_filename,2))+5)) 'times.mat']);
load([load_p(1:(end-(size(bemobil_config.study_filename,2))+5)) 'times_all.mat']);
load([load_p(1:(end-(size(bemobil_config.study_filename,2))+5)) 'freqs.mat']);

first_ix = find(times_all==times(1));
last_ix = find(times_all==times(end));
max_freq_ix = find(freqs>=40,1,'first');
freqs = freqs(1:max_freq_ix);

coeffs_to_plot = [2:4, 7:8];
nr_of_plots = 2 + size(coeffs_to_plot,2);
subplots = [ceil(sqrt(nr_of_plots)), ceil(sqrt(nr_of_plots))];

cs = channels_of_int; % clusters_of_int
sensor = 'channel';

for c = cs
    
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
    
    tightfig;
    
%     saveas(gcf, [load_p 'cluster_' num2str(c) '/res_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(c) '.png'], 'png')
%     close(gcf);

end