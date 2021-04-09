
%% plot cluster scalp map and dipoles, dipole location

all_clusters = [10, 6, 4, 11, 7];
cluster = all_clusters(5);

shuffled_baseline = 0;
matched_trial_count = 1;
time_window_for_analysis = [-100, 1000];
model = {'ersp_sample ~ oddball*haptics + base' ,...
    'ersp_sample ~ haptics*velocity_at_impact + diff_at + base'}; % 'oddball ~ ersp_sample*haptics' 
log_regression = [0];

load(fullfile('/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/derivatives/results', ...
    num2str(cluster), [model{1,2} '_base-shuffled-0_matched-trial-count-1_log-regression-' num2str(log_regression) '.mat']))
% disp(['cluster: ', num2str(cluster), ', location: ', num2str(STUDY.cluster(cluster).dipole.posxyz)]);

% results windows
t_lim = 9:48; % (700ms post event)
freq_lim = 48; % (40 Hz)

% exclude participants not in cluster
stat_fields = {'betas', 't', 'p'};
rm_subjects = find(fit.betas(:,1,1,1)==0);
for stat_field = stat_fields
    fit.(stat_field{1})(rm_subjects,:,:,:) = [];
end

%% group-level statistics
clear tfce
fit.save_path = ['/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/results/cluster_' num2str(cluster)];
fit.alpha = .05;
fit.perm = 1000;
for i = 2:size(fit.predictor_names,2)

    % get betas per predictor
    betas = fit.betas(:,1:freq_lim,t_lim,i);
    betas = permute(betas, [2, 3, 1]);
    zero = zeros(size(betas));

    % permutation t-test
    [fit.stats(i).t_stats, ~, fit.stats(i).betas_p_vals, fit.stats(i).surrogate_data] = statcond({betas zero},...
        'method', 'perm', 'naccu', fit.perm);

    % compute tfce transform of t_maps surrogate data, add max tfce dist
    for j = 1:size(fit.stats(i).surrogate_data,3)
        tfce(j,:,:) = limo_tfce(2,squeeze(fit.stats(i).surrogate_data(:,:,j)),[],0);
        this_max = tfce(j,:,:);
        fit.stats(i).tfce_max_dist(j) = max(this_max(:));
    end

    % threshold true t_map
    [~,~,~,STATS] = ttest(permute(betas, [3, 1, 2]));
    fit.stats(i).tfce_true = limo_tfce(2,squeeze(STATS.tstat),[],0);
    fit.stats(i).tfce_thresh = prctile(fit.stats(i).tfce_max_dist,95);
    fit.stats(i).tfce_sig_mask = fit.stats(i).tfce_true>fit.stats(i).tfce_thresh;
end

%% plot
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1400 500]);
measures = 2:size(fit.betas,4);
for measure = measures
    subplot(1, size(measures,2),measure-1); sgtitle(num2str(cluster));
    to_plot = squeezemean(fit.betas(:,1:freq_lim,t_lim,measure),1);
    p = fit.stats(measure).betas_p_vals;

%     [~, p_fdr_mask] = fdr(p, .05);
%     p = p .* p_fdr_mask;
    p = p .* fit.stats(measure).tfce_sig_mask;

    disp([fit.predictor_names{measure}, ' :', num2str(sum(fit.stats(measure).tfce_sig_mask(:)))]);
    plotersp(fit.times(t_lim), fit.freqs(1:freq_lim), to_plot, p, .05, 'auto', '-', 'frequency (Hz)', 'time (ms)', fit.predictor_names{measure}, 'dB', 1);
end

%% save
print(gcf, fullfile(bemobil_config.study_folder, 'results', num2str(cluster), ['single_trial-model' fit.model '.eps']), '-depsc');
close(gcf);
