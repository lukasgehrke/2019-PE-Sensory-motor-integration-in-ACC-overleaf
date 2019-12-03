%% plot result 1:

event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);
robustfit = 1;

% shift due to EEG age of sample
mocap_aos = .011; % age of sample mocap samples = 11 ms
mocap_aos_samples = ceil(mocap_aos * 250);
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);
robustfit = 0;

% try out several ts prior to event
seconds_before_event = .3;
samples_before_event = seconds_before_event * 250;
ts_of_ints = (event_onset-samples_before_event):25:event_onset+50;

eeg_age_of_sample_samples = floor(eeg_age_of_sample * 250);
plot_s_after_event = .5; % plot 0 to half a second after event
plot_win = event_onset:event_onset+(plot_s_after_event * 250);
event_0_ix = (event_onset+eeg_age_of_sample_samples-plot_win(1)) / 250;

% load
load_p = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/erp/cluster_';
model = 'erp_sample ~ immersion * vel + trial + direction + sequence';

ts_all = (event_onset - ts_of_ints) / 250;

% select best one
% this_ts = this_ts(end);


%% plot results

%% make figure with mean results across all participants, average pvals and

cols = brewermap(2, 'Spectral');
% select subjects out of clusters of int
clusters_of_int = [3, 7, 9, 24, 25, 28, 30, 33, 34];
% clusters
% 3: right parietal
% 7: right motor?
% 24: right SMA
% 25: left parietal
% 28: interesting
% 33: ACC

this_ts = ts_all(4);

load_p = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/erp/cluster_';

for c = clusters_of_int

    % best R^2: this_ts(6) 
    load([load_p num2str(c) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(c) '.mat']);

    % do fdr if anything is significant at the uncorrected .05 level
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 400 1200]);

    % plot mean ERP -> sum all betas
    subplot(5,1,1);
    data = sum(res.betas(:,plot_win,:),3);
    ploterp(data, event_0_ix, 1, 'mean amplitude', '', 0, cols);

    % plot r^2
    subplot(5,1,2);
    data = res.r2(:,plot_win);
    ploterp(data, event_0_ix, 1, 'adjusted R^2', '', 1, cols);

    % plot haptics coefficient
    subplot(5,1,3);
    data = res.betas(:,plot_win,2);
    ploterp(data, event_0_ix, 1, 'immersion coeff.', '', 0, cols);

    % plot velocity coefficient
    subplot(5,1,4);
    data = res.betas(:,plot_win,3);
    ploterp(data, event_0_ix, 1, 'velocity coeff.', '', 0, cols);

    % plot haptics * velocity coefficient (= interaction effect of haptic immersion and velocity)
    subplot(5,1,5);
    data = res.betas(:,plot_win,4);
    ploterp(data, event_0_ix, 1, 'vel. x immersion coeff.', 'time in ms', 0, cols);

    tightfig;
    
    saveas(gcf, [load_p num2str(c) '/res_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(c) '.png'], 'png')
    close(gcf);

end

%% plot r^2s for many clusters and one vel timepoint
cols = brewermap(size(clusters_of_int,2), 'Spectral');
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 400 600]);
ylim([0 .6]);
hold on;

ix = 1;
for c = clusters_of_int
    
    load([load_p num2str(c) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(c) '.mat']);
    data = res.r2(:,plot_win);
    
    l = plot(mean(data,1));
    l.LineWidth = 3;
    l.Color = cols(ix,:);
    ix = ix + 1;
    
%     subplot(1,4,ix);
%     ix = ix + 1;
%     ploterp(data, event_0_ix, 1, 'adjusted R^2', 'time in ms', 1, cols);
    
end
title(model);
legend(string(clusters_of_int));

% format further
grid on
set(gca,'FontSize',20)

%% plot r^2s for many timepoints and one cluster
cols = brewermap(size(clusters_of_int,2), 'Spectral');

for c = clusters_of_int
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 400 600]);
    ylim([0 .6]);
    hold on;
    ix = 1;

    for this_ts = ts_all

        load([load_p num2str(c) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(c) '.mat']);
        data = res.r2(:,plot_win);

        l = plot(mean(data,1));
        l.LineWidth = 3;
        l.Color = cols(ix,:);
        ix = ix + 1;

    end
    title(string(c));
    legend(string(ts_all));
    % format further
    grid on
    set(gca,'FontSize',20)
end

%% where are effects plot mean pvalues per main effect for all clusters

cols = brewermap(size(clusters_of_int,2), 'Spectral');
% figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 400 600]);
figure;
ylim([-.5 .5]);
hold on;
main_effect = '';
main_effect_ix = 8;

%     {'(Intercept)'     }
%     {'immersion_1'     }
%     {'vel'             }
%     {'trial'           }
%     {'direction_middle'}
%     {'direction_right' }
%     {'sequence'        }
%     {'immersion_1:vel' }

ix = 1;
for c = clusters_of_int
    
    load([load_p num2str(c) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(c) '.mat']);
    data = squeeze(res.p(:,plot_win,main_effect_ix));
    
    l = plot(mean(data,1));
    l.LineWidth = 3;
    l.Color = cols(ix,:);
    ix = ix + 1;
    
%     subplot(1,4,ix);
%     ix = ix + 1;
%     ploterp(data, event_0_ix, 1, 'adjusted R^2', 'time in ms', 1, cols);
    
end
title(['main effect: ' main_effect]);
legend(string(clusters_of_int));

% % format further
% l=hline(.05);
% l.Color = 'k';
% l.LineWidth = 2;
% l.LineStyle = ':';

grid on
set(gca,'FontSize',20)





%% plot r^2s for all clusters and the immersion X vel model
load_p = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/cluster_';
model = 'erp_sample ~ predictor_immersion * predictor_vel';
robustfit = 0;
this_ts = 0.0320;

ix = 1;
clusters_of_int = 1:30;

figure; hold on;

for c = clusters_of_int
    
    load([load_p num2str(c) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(c) '.mat']);
    data = res.r2(:,plot_win);
    
    plot(mean(data,1))
%     subplot(1,4,ix);
%     ix = ix + 1;
%     ploterp(data, event_0_ix, 1, 'adjusted R^2', 'time in ms', 1, cols);
    
end
title(model);
legend();