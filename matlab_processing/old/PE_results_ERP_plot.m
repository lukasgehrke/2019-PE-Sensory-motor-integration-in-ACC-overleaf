%% Params
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
% ts_of_ints = ts_of_ints(4);
ts_all = (event_onset - ts_of_ints) / 250;

erp_type = 'non_corrected';

% colors
cols = brewermap(2, 'Spectral');

% load
load_p = ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/erp/' bemobil_config.study_filename(1:end-6) '/'];
model = 'erp_sample ~ immersion * vel + trial + direction + sequence';

zero = 3*250; % [-3 2] epoch around event    
event_win_samples = zero + (bemobil_config.epoching.event_win(1) * 250):zero+(bemobil_config.epoching.event_win(2) * 250);
event_win_samples = event_win_samples - event_onset;
event_win_times = event_win_samples / 250;
event_0_ix = find(event_win_times<0, 1, 'last');
event_0 = 0;

alpha = .05;
coeffs_to_plot_names = {'', 'immersion', 'velocity', 'trial number', '', '', 'sequence', 'vel. x immersion'};
coeff_names =  {'', 'immersion_1', 'vel', 'trial', '', '', 'sequence', 'immersion_1_vel'};

%% plot r^2s for timepoints and channels FZ and PZ

cols = brewermap(size(channels_of_int,2), 'Accent');
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 800 400]);
ylim([0 .2]);
hold on;

% select -.2 -.1 0
ts_short = ts_all(2:4);

for this_ts = ts_short
    ix = 1;
    for c = channels_of_int

        load([load_p 'channel_' num2str(c) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event.mat']);
        data = res.r2;
        x_plot = (-event_0_ix:size(data,2)) / 250;
        x_plot(end-event_0_ix:end) = [];
        x_plot = x_plot * 1000; % seconds to ms

        l = plot(x_plot, mean(data,1));
        l.LineWidth = 3;
        l.Color = cols(ix,:);
        ix = ix + 1;

    end
end
title({'Modelfit of channels of model:',model});
xlabel('time in ms')
ylabel('R^2')
l2 = line([0 0], [0 max(ylim)]);
l2.LineStyle = '-';
l2.Color = 'k';
l2.LineWidth = 4;

legend({'Fz', 'Pz', 'FCz'});

% format further
grid on
set(gca,'FontSize',20)
tightfig;

% save    
saveas(gcf, [load_p 'r2_res_robust-' num2str(robustfit) '_modelfit_across_timepoints.png'], 'png')
close(gcf);

%% Channels & Clusters: make figure with mean results across all participants, mark significant pvals and

cols = brewermap(2, 'Spectral');

sensor = 'cluster'; % channel
cs = clusters_of_int; % channels_of_int

coeffs_to_plot = [2:4, 7:8];
nr_of_plots = 2 + size(coeffs_to_plot,2);
subplots = [2, size(coeffs_to_plot,2)];

this_ts = ts_all(4);

for c = cs
    
    % subplot ix
    s_ix = 1;

    % best R^2: this_ts(6) 
    load([load_p sensor '_' num2str(c) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event.mat']);

    % make figure
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1800 600]);

    % plot mean ERP -> sum all betas
    subplot(subplots(1),subplots(2),s_ix);
    s_ix = s_ix + 1;
    data = sum(res.betas,3);
    ploterp_lg(data, [], event_0_ix, 1, 'mean amplitude', '', [], cols);

    % plot r^2
    subplot(subplots(1),subplots(2),s_ix);
    s_ix = s_ix + 4;
    data = res.r2;
    ploterp_lg(data, [], event_0_ix, 1, 'R^2', '', 1, cols);

    for coeff = coeffs_to_plot
        
        % plot coefficient
        subplot(subplots(1),subplots(2),s_ix);
        s_ix = s_ix + 1;
        
        % get data
        data = res.betas(:,:,coeff);
        
        % make tfce thresh sig mask
        load([load_p sensor '_' num2str(c) '/ttest_' coeff_names{coeff} '/H0/tfce_H0_one_sample_ttest_parameter_1.mat']); % bootstraps tfce
        load([load_p sensor '_' num2str(c) '/ttest_' coeff_names{coeff} '/tfce/tfce_one_sample_ttest_parameter_1.mat']); % true tfce
        for i = 1:size(tfce_H0_one_sample,3)
            max_i(i) = max(tfce_H0_one_sample(1,event_0_ix:end,i));
        end
        thresh = prctile(max_i, (1-alpha)*100);
        sig_ix = find(tfce_one_sample>thresh);
        
        % plot
        ploterp_lg(data, sig_ix, event_0_ix, 1, coeffs_to_plot_names(coeff), '', [], cols);
    end

    tightfig;
    
    saveas(gcf, [load_p sensor '_' num2str(c) '/res_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event.png'], 'png')
    close(gcf);

end

%% Channels & Clusters: make figure with regression IPQ results, mark significant pvals and

cols = brewermap(2, 'Spectral');

sensor = 'cluster'; % channel
cs = clusters_of_int; % channels_of_int

coeff_names =  {'mean', 'immersion_1', 'vel', 'trial', 'sequence', 'immersion_1_vel'};
subplots = [2, 6]; % top coefficient, below R^2

for c = cs
    
    % subplot ix
    s_ix = 1;
    % make figure
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1800 600]);
        
    for coeff = coeff_names

        subplot(subplots(1),subplots(2),s_ix);
        s_ix = s_ix + 1;
        load([load_p sensor '_' num2str(c) '/regress_' coeff{1} '/Betas.mat']);
        data = squeeze(Betas(1,:,2));
        
        % make tfce thresh sig mask
        load([load_p sensor '_' num2str(c) '/regress_' coeff{1} '/H0/tfce_H0_Covariate_effect_1.mat']); % bootstraps tfce
        load([load_p sensor '_' num2str(c) '/regress_' coeff{1} '/TFCE/tfce_Covariate_effect_1.mat']); % true tfce
        for i = 1:size(tfce_H0_score,3)
            max_i(i) = max(tfce_H0_score(event_0_ix:end,2,i));
        end
        thresh = prctile(max_i, (1-alpha)*100);
        sig_ix = find(tfce_score>thresh);
        ploterp_lg(data, sig_ix, event_0_ix, 1, ['IPQ: ' coeff{1}], '', [], cols);

        % plot r^2
        subplot(subplots(1),subplots(2),s_ix + (subplots(2)-1));
        load([load_p sensor '_' num2str(c) '/regress_' coeff{1} '/R2.mat']);
        data = squeeze(R2(1,:,1));
        ploterp_lg(data, [], event_0_ix, 1, 'R^2', '', [0 1], cols);
    end

    tightfig;
    
    saveas(gcf, [load_p sensor '_' num2str(c) '/ipq_res.png'], 'png')
    close(gcf);

end

%% plot r^2s for many clusters and one vel timepoint

cols = brewermap(size(clusters_of_int,2), 'Spectral');
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 800 400]);
ylim([0 .3]);
hold on;

ix = 1;
for c = clusters_of_int
    
    load([load_p 'cluster_' num2str(c) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event.mat']);
    data = res.r2;
    x_plot = (-event_0_ix:size(data,2)) / 250;
    x_plot(end-event_0_ix:end) = [];
    
    l = plot(x_plot, mean(data,1));
    l.LineWidth = 3;
    l.Color = cols(ix,:);
    ix = ix + 1;
    
end
title({'Modelfit of clusters of ICs of model:',model});
xlabel('time in ms')
ylabel('R^2')
l2 = line([0 0], [0 max(ylim)]);
l2.LineStyle = '-';
l2.Color = 'k';
l2.LineWidth = 4;

legend(string(clusters_of_int));

% format further
grid on
set(gca,'FontSize',20)
tightfig;

% save    
saveas(gcf, [load_p 'r2_res_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event.png'], 'png')
close(gcf);

