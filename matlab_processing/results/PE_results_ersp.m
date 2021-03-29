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
STUDY_sets = cellfun(@str2num, {STUDY.datasetinfo.subject});

%% compute movement predictors

for subject = subjects

    ALLEEG(subject).etc.analysis.design.reaction_time = ...
        (ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample - ALLEEG(subject).etc.analysis.design.spawn_event_sample) / ALLEEG(subject).srate;
    
    event_sample_ix = abs(bemobil_config.epoching.event_epochs_boundaries(1)) * ALLEEG(subject).srate; % epoched [-3 2] seconds = 1250 samples
    ALLEEG(subject).etc.analysis.design.action_time = ...
        (abs(event_sample_ix) - ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample) / ALLEEG(subject).srate;
    
    ALLEEG(subject).etc.analysis.design.time_to_reach_vel_peak = ...
        ALLEEG(subject).etc.analysis.design.movements.reach_max_vel_ix - ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample;
    
    ALLEEG(subject).etc.analysis.design.peak_vel_to_contact = ...
        event_sample_ix - ALLEEG(subject).etc.analysis.design.movements.reach_max_vel_ix;
    
    ALLEEG(subject).etc.analysis.design.peak_vel_to_retract = ...
        ALLEEG(subject).etc.analysis.design.movements.retract_onset_sample - ALLEEG(subject).etc.analysis.design.movements.reach_max_vel_ix;
    
    ALLEEG(subject).etc.analysis.design.full_outward_movement = ...
        ALLEEG(subject).etc.analysis.design.movements.retract_onset_sample - ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample;
end

%% [] ERSP statistics async only
% ersp ~ vel_at_col * haptics + rt + base
% velocity: does it impact multisensory integration during spatio-temporal binding
% haptics: same as velocity, does object rigidity further perturb spatio-temporal binding
% rt: is there a task clock? self initiated movement timing?
% base: what activity at baseline impacts activity at spatio-temporal binding event -> that part of the activity is not processing related

% load clustering solution
cluster_path = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error';
clustering = load(fullfile(cluster_path, 'derivatives', 'cluster_ROI_0_9_39.mat'));
STUDY.cluster = clustering.STUDY.cluster;
clusters = [11, 14];

for cluster = clusters
    
    % single-trial model fitting
    base_shuffled = 1;
    % zero = 750;
    fit.model = 'ersp_sample ~ oddball * haptics + velocity + out_move + base';

    % get matching datasets from EEGLAB Study struct
    unique_setindices = unique(STUDY.cluster(cluster).sets);
    unique_subjects = STUDY_sets(unique_setindices);
    all_setindices = STUDY.cluster(cluster).sets;
    all_sets = STUDY_sets(all_setindices);
    all_comps = STUDY.cluster(cluster).comps;

    count = 1;
    for subject = unique_subjects

        % select components
        this_sets = find(all_sets==subject);
        comps = all_comps(this_sets);

        % predictors movement
        out_move = [mean(diff(ALLEEG(subject).etc.analysis.design.full_outward_movement)), diff(ALLEEG(subject).etc.analysis.design.full_outward_movement)]' / ALLEEG(subject).srate; % from peak vel to contac

        % predictors task
        oddball = ALLEEG(subject).etc.analysis.design.oddball';
        post_error = ["false"; oddball(1:end-1)];
        post_error = double(post_error=='true');
        isitime = ALLEEG(subject).etc.analysis.design.isitime';
        sequence = ALLEEG(subject).etc.analysis.design.sequence';
        trial_number = ALLEEG(subject).etc.analysis.design.trial_number';
        haptics = double(ALLEEG(subject).etc.analysis.design.haptics)';
        direction = categorical(ALLEEG(subject).etc.analysis.design.direction');
        pID = repmat(subject,size(direction,1),1);

        reg_t = table(post_error, isitime, sequence, haptics, trial_number, direction, oddball, out_move, pID);
        reg_t(ALLEEG(subject).etc.analysis.design.bad_touch_epochs,:)= [];

    %     % match trial count
    %     match_ixs = find(reg_t.post_error==0);
    %     mismatch_ixs = find(reg_t.post_error==1);
    %     match_ixs = randsample(match_ixs, numel(mismatch_ixs));
    %     reg_t = reg_t(union(match_ixs, mismatch_ixs),:);

        % load movement
        load(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_motion.mat']));
        motion.mag_vel(:, ALLEEG(subject).etc.analysis.design.bad_touch_epochs) = [];
        sample_time = (-3:1/ALLEEG(subject).srate:2) * 1000;
        sample_time(end) = [];

        % load ersp
        load(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_ersp.mat']));
        % remove ICs identical to Study IC selection
        good_comps_ori_ix = find(ALLEEG(subject).etc.ic_classification.ICLabel.classifications(:,1) > bemobil_config.lda.brain_threshold);
        ersp.tf_event_raw_power = ersp.tf_event_raw_power(good_comps_ori_ix,:,:,:);
        ersp.tf_base_raw_power = ersp.tf_base_raw_power(good_comps_ori_ix,:,:,:);

        % find nearest element
        [~, t1_ix] = min(abs(ersp.tf_event_times-(bemobil_config.timewarp_anchors(1))));
        ixs = t1_ix:size(ersp.tf_event_times,2);

        % fitlm for each time frequency pixel
        tic
        disp(['now fitting data for subject: ' num2str(subject) ' and model: ' fit.model '...']);
        for t = ixs

            % add matching velocity for this frame
            [~, ix] = min(abs(sample_time - ersp.tf_event_times(t)));
            velocity = motion.mag_vel(ix,:)';

            for f = 1:size(ersp.tf_event_freqs,2)

                % add ersp and baseline sample to design matrix
                if size(comps,2) > 1
                    ersp_sample = squeezemean(ersp.tf_event_raw_power(comps,f,t,:),1);
                    base = squeezemean(ersp.tf_base_raw_power(comps,f,:),1);
                else
                    ersp_sample = squeeze(ersp.tf_event_raw_power(comps,f,t,:));
                    base = squeeze(ersp.tf_base_raw_power(comps,f,:));
                end

                % random to get impact of baseline differnece in haptic
                % condition
                if base_shuffled
                    base = base(randperm(length(base)));
                end

                reg_t = addvars(reg_t, ersp_sample, base, velocity);

                % fit model and save
                mdl = fitlm(reg_t, fit.model);
                fit.betas(subject,f,t-ixs(1)+1,:) = mdl.Coefficients.Estimate;
                fit.t(subject,f,t-ixs(1)+1,:) = mdl.Coefficients.tStat;
                fit.p(subject,f,t-ixs(1)+1,:) = mdl.Coefficients.pValue;
                fit.r2(subject,f,t-ixs(1)+1,:) = mdl.Rsquared.Ordinary;
                fit.adj_r2(subject,f,t-ixs(1)+1,:) = mdl.Rsquared.Adjusted;

                reg_t = removevars(reg_t, {'ersp_sample', 'base', 'velocity'});
            end
        end
        toc
        clear reg_t

    end

    % group-level statistics: settings
    fit.predictor_names = string(mdl.CoefficientNames);
    fit.times = ersp.tf_event_times(ixs);
    fit.freqs = ersp.tf_event_freqs;
    fit.save_path = ['/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/results/cluster_' num2str(cluster)];
    fit.alpha = .05;
    fit.perm = 1000;
    for i = 2:size(fit.predictor_names,2)

        % get betas per predictor
        betas = fit.betas(:,:,:,i);
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

    % save results
    if ~isfolder(fit.save_path)
        mkdir(fit.save_path)
    end
    save(fullfile(fit.save_path, [fit.model '_' num2str(base_shuffled) '_.mat']), 'fit'); clear fit;

end

%% plot
normal; % plot normal window, not docked
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1400 300]);

measures = 1:7;
c = 1;
for measure = measures
%     subplot(1,size(measures,2),measure);
    to_plot = squeezemean(fit.betas(:,:,:,measure),1);
    p = fit.stats(measure).betas_p_vals;
    disp([fit.predictor_names{measure}, ' :', num2str(sum(fit.stats(measure).tfce_sig_mask(:)))]);
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 500 300]);
    plotersp(fit.times, fit.freqs, to_plot, p, .05, 'auto', '-', 'frequency (Hz)', 'time (ms)', fit.predictor_names{measure}, 'dB', 1);
end


% save plot
%print(gcf, [res.save_path 'st_betas_' num2str(measure) '.eps'], '-depsc');
%close(gcf);

% extract stats

% 1. alpha baseline post event
effect = 2;
t = 48; % 200 ms
fit.times(t)
f = 21; % 9 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 2. alpha rt post event
effect = 5;
t = 52; % 250 ms
fit.times(t)
f = 26; % 12 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 3. alpha rt pre event
effect = 5;
t = 16; % -200 ms
fit.times(t)
f = 16; % 12 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 4. alpha haptics post event
effect = 3;
t = 46; % 180 ms
fit.times(t)
f = 11; % 5 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 5. alpha haptics post event
effect = 3;
t = 46; % 180 ms
fit.times(t)
f = 26; % 12 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 6. alpha velocity pre event
effect = 4;
t = 16; % 180 ms
fit.times(t)
f = 26; % 12 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 7. theta interaction post event
effect = 6;
t = 46; % 180 ms
fit.times(t)
f = 6; % 12 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

