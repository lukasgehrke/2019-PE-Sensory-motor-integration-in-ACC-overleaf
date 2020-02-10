%% Params
PE_config;

robustfit = 0; % fit robust regression with squared weights, see fitlm
event_sample = 750;
window = event_sample-25:event_sample+200; %[-.1 .8]seconds start and end of interesting, to be analyzed, samples
count = 1;

alpha = .05;

%% plot r^2s for ERPs

% what to plot
metric = 'mocap'; % 'erp';
channel = 'vel'; %['channel_' num2str(65)];
channel_name = 'FCz';
plot_sample_start_end = -250:250;
models = {'erp_sample ~ congruency * haptics + trial_nr + direction + sequence', ...
    'erp_sample ~ velocity * haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ after_mismatch * haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ congruency * haptics + trial_nr + direction + sequence'};
model = models{3};

% paths
load_p = ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/' ...
    metric '/' bemobil_config.study_filename(1:end-6) '/'];

% shift due to EEG age of sample?
age_of_sample = 0;

% prepare plot
cols = brewermap(size(channels_of_int,2), 'Accent');
figure('Renderer', 'painters', 'Position', [10 10 450 300]);
hold on;

% load and plot
load([load_p channel '/' model '/res_' model '_robust-' num2str(robustfit) '.mat']);
data = res.r2;
%x_plot = (-25-age_of_sample:size(data,2)-26-age_of_sample) / 250;
% x_plot = x_plot * 1000; % seconds to ms
x_plot = plot_sample_start_end / 250;
ylim([0 max(mean(data,1))+.05]);
l = plot(x_plot, mean(data,1));
l.LineWidth = 3;
l.Color = cols(1,:);

% title({'Modelfit of channels of model:',model});
xlabel('time in ms')
ylabel('R^2')
l2 = line([0 0], [0 max(ylim)]);
l2.LineStyle = '-';
l2.Color = 'k';
l2.LineWidth = 3;

% format further
% legend({channel_name});
grid on
set(gca,'FontSize',20)
% tightfig;

%% Channels & Clusters: make figure with mean results across all participants, mark significant pvals and

% what to plot
channel = 65;
channel_name = 'FCz';
model = models{2};

sensor = 'cluster'; % channel
cs = clusters_of_int; % channels_of_int

coeffs_to_plot = [2:4, 7:8];
nr_of_plots = 2 + size(coeffs_to_plot,2);
subplots = [2, size(coeffs_to_plot,2)];

this_ts = ts_all(4);

for chan = cs
    
    % subplot ix
    s_ix = 1;

    % best R^2: this_ts(6) 
    load([load_p sensor '_' num2str(chan) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event.mat']);

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
        load([load_p sensor '_' num2str(chan) '/ttest_' coeff_names{coeff} '/H0/tfce_H0_one_sample_ttest_parameter_1.mat']); % bootstraps tfce
        load([load_p sensor '_' num2str(chan) '/ttest_' coeff_names{coeff} '/tfce/tfce_one_sample_ttest_parameter_1.mat']); % true tfce
        for i = 1:size(tfce_H0_one_sample,3)
            max_i(i) = max(tfce_H0_one_sample(1,event_0_ix:end,i));
        end
        thresh = prctile(max_i, (1-alpha)*100);
        sig_ix = find(tfce_one_sample>thresh);
        
        % plot
        ploterp_lg(data, sig_ix, event_0_ix, 1, coeffs_to_plot_names(coeff), '', [], cols);
    end

    tightfig;
    
    saveas(gcf, [load_p sensor '_' num2str(chan) '/res_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event.png'], 'png')
    close(gcf);

end

%% Channels & Clusters: make figure with regression IPQ results, mark significant pvals and

cols = brewermap(2, 'Spectral');

sensor = 'cluster'; % channel
cs = clusters_of_int; % channels_of_int

coeff_names =  {'mean', 'immersion_1', 'vel', 'trial', 'sequence', 'immersion_1_vel'};
subplots = [2, 6]; % top coefficient, below R^2

for chan = cs
    
    % subplot ix
    s_ix = 1;
    % make figure
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1800 600]);
        
    for coeff = coeff_names

        subplot(subplots(1),subplots(2),s_ix);
        s_ix = s_ix + 1;
        load([load_p sensor '_' num2str(chan) '/regress_' coeff{1} '/Betas.mat']);
        data = squeeze(Betas(1,:,2));
        
        % make tfce thresh sig mask
        load([load_p sensor '_' num2str(chan) '/regress_' coeff{1} '/H0/tfce_H0_Covariate_effect_1.mat']); % bootstraps tfce
        load([load_p sensor '_' num2str(chan) '/regress_' coeff{1} '/TFCE/tfce_Covariate_effect_1.mat']); % true tfce
        for i = 1:size(tfce_H0_score,3)
            max_i(i) = max(tfce_H0_score(event_0_ix:end,2,i));
        end
        thresh = prctile(max_i, (1-alpha)*100);
        sig_ix = find(tfce_score>thresh);
        ploterp_lg(data, sig_ix, event_0_ix, 1, ['IPQ: ' coeff{1}], '', [], cols);

        % plot r^2
        subplot(subplots(1),subplots(2),s_ix + (subplots(2)-1));
        load([load_p sensor '_' num2str(chan) '/regress_' coeff{1} '/R2.mat']);
        data = squeeze(R2(1,:,1));
        ploterp_lg(data, [], event_0_ix, 1, 'R^2', '', [0 1], cols);
    end

    tightfig;
    
    saveas(gcf, [load_p sensor '_' num2str(chan) '/ipq_res.png'], 'png')
    close(gcf);

end

%% plot r^2s for many clusters and one vel timepoint

cols = brewermap(size(clusters_of_int,2), 'Spectral');
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 800 400]);
ylim([0 .3]);
hold on;

ix = 1;
for chan = clusters_of_int
    
    load([load_p 'cluster_' num2str(chan) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event.mat']);
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
