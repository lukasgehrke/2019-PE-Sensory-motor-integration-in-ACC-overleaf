%% clear all and load params
clear all;

if ~exist('ALLEEG','var')
	eeglab;
end

% add downloaded analyses code to the path
addpath(genpath('/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing'));
% TODO add to path bemobil_pipeline repository download folder
% TODO add to path custom scripts repository Lukas Gehrke folder

% BIDS data download folder
bemobil_config.BIDS_folder = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/ds003552';
% Results output folder -> external drive
bemobil_config.study_folder = fullfile('/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error', 'derivatives');

% init
config_processing_pe;
subjects = 1:19;

%% load study

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 1, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

if isempty(STUDY)
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    [STUDY ALLEEG] = pop_loadstudy('filename', ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_' bemobil_config.study_filename], 'filepath', bemobil_config.study_folder);
    CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];    
    eeglab redraw
end

%% plot cluster scalp map and dipoles, dipole location

all_clusters = [10, 6, 4, 11, 9];
shuffled_baseline = 0;
matched_trial_count = 1;
models = {...
    'ersp_sample ~ oddball*haptics + base',...
    'ersp_sample ~ haptics*velocity_at_impact + diff_at + base',...
    'ersp_sample ~ diff_at*haptics + base',... 
    'ersp_sample ~ diff_at + base',...
    }; 
log_regression = [0, 0, 0, 0];

% results windows: time_window_for_analysis = [-700, 1400];
start_t = -50; % -100
end_t = 1000; % 1400
ceil_freq = 40;

for i = 1:numel(all_clusters)
    cluster = all_clusters(i);
    
    for j = 1:numel(models)
        
        disp(['cluster: ', num2str(cluster), ', location: ', num2str(STUDY.cluster(cluster).dipole.posxyz)]);
        disp(['model: ', models{j}]);
        
        %% load data
        load(fullfile(bemobil_config.study_folder, 'results', ...
            ['cluster_ROI_' ...
            num2str(bemobil_config.STUDY_cluster_ROI_talairach(i).x) '_' ...
            num2str(bemobil_config.STUDY_cluster_ROI_talairach(i).y) '_' ...
            num2str(bemobil_config.STUDY_cluster_ROI_talairach(i).z)], ...        
            num2str(cluster), ...
            [models{j} '_base-shuffled-' num2str(shuffled_baseline) '_matched-trial-count-' num2str(matched_trial_count) ...
            '_log-regression-' num2str(log_regression(j)) '.mat'])); 

        start_t_ix = min(find(fit.times>=start_t));
        end_t_ix = min(find(fit.times>=end_t));
        t_lim = start_t_ix:end_t_ix;
        
        freq_lim = min(find(fit.freqs>ceil_freq));

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
        for k = 2:size(fit.predictor_names,2)

            % get betas per predictor
            betas = fit.betas(:,1:freq_lim,t_lim,k);
            betas = permute(betas, [2, 3, 1]);
            zero = zeros(size(betas));

            % permutation t-test
            [fit.stats(k).t_stats, ~, fit.stats(k).betas_p_vals, fit.stats(k).surrogate_data] = statcond({betas zero},...
                'method', 'perm', 'naccu', fit.perm);

            % compute tfce transform of t_maps surrogate data, add max tfce dist
            for l = 1:size(fit.stats(k).surrogate_data,3)
                tfce(l,:,:) = limo_tfce(2,squeeze(fit.stats(k).surrogate_data(:,:,l)),[],0);
                this_max = tfce(l,:,:);
                fit.stats(k).tfce_max_dist(l) = max(this_max(:));
            end

            % threshold true t_map
            [~,~,~,STATS] = ttest(permute(betas, [3, 1, 2]));
            fit.stats(k).tfce_true = limo_tfce(2,squeeze(STATS.tstat),[],0);
            fit.stats(k).tfce_thresh = prctile(fit.stats(k).tfce_max_dist,95);
            fit.stats(k).tfce_sig_mask = fit.stats(k).tfce_true>fit.stats(k).tfce_thresh;
        end

        %% plot
        figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1400 500]); normal;
        measures = 2:size(fit.betas,4);
        for measure = measures
            subplot(1, size(measures,2),measure-1); sgtitle(num2str(cluster));
            to_plot = squeezemean(fit.betas(:,1:freq_lim,t_lim,measure),1);
            
%             p = fit.stats(measure).betas_p_vals;
%             [~, p_fdr_mask] = fdr(p, .05);
%             p = p .* p_fdr_mask;
%             p = p .* fit.stats(measure).tfce_sig_mask;
            
            p = fit.stats(measure).tfce_sig_mask;

            disp([fit.predictor_names{measure}, ' :', num2str(sum(fit.stats(measure).tfce_sig_mask(:)))]);
            plotersp(fit.times(t_lim), fit.freqs(1:freq_lim), to_plot, p, [], 'auto', '-', 'frequency (Hz)', 'time (ms)', fit.predictor_names{measure}, 'dB', 1);
        end

        %% save
        
        out_folder = fullfile(bemobil_config.study_folder, 'results', ...
            ['cluster_ROI_' ...
            num2str(bemobil_config.STUDY_cluster_ROI_talairach(i).x) '_' ...
            num2str(bemobil_config.STUDY_cluster_ROI_talairach(i).y) '_' ...
            num2str(bemobil_config.STUDY_cluster_ROI_talairach(i).z)], ...        
            num2str(cluster), [num2str(start_t) '_' num2str(end_t)]);
        
        if ~isfolder(out_folder)
            mkdir(out_folder);
        end
        
        save_name = fullfile(out_folder, ...
            [models{j} ...
            '_base-shuffled-' num2str(shuffled_baseline) ...
            '_matched-trial-count-' num2str(matched_trial_count) ...
            '_log-regression-' num2str(log_regression(j)) ...
            '_p_thresh-tfce']);
        
        print(gcf, save_name, '-depsc');
        savefig(gcf, save_name);
        close(gcf);
   
    end
end
