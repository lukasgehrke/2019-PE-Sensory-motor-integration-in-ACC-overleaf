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
    
    % no effect and correctly calculated
    ALLEEG(subject).etc.analysis.design.reaction_time = ...
        (ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample - ALLEEG(subject).etc.analysis.design.spawn_event_sample) / ALLEEG(subject).srate;
    
    % 
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

% compute these measures in the results scripts    
%     EEG.etc.analysis.design.reaction_time = (EEG.etc.analysis.design.movements.movement_onset_sample - EEG.etc.analysis.design.spawn_event_sample) / EEG.srate;
%     EEG.etc.analysis.design.action_time = (abs(event_sample_ix) - EEG.etc.analysis.design.movements.movement_onset_sample) / EEG.srate;

%% build full design matrices

all_subjects_reg_t = table();

for subject = subjects
    
    % predictors: change from previous trial
    rt = [mean(diff(ALLEEG(subject).etc.analysis.design.reaction_time)), diff(ALLEEG(subject).etc.analysis.design.reaction_time)]'; % spawn to movement onset
    at = [mean(diff(ALLEEG(subject).etc.analysis.design.action_time)), diff(ALLEEG(subject).etc.analysis.design.action_time)]'; % from movement onset to touch
    acc = [mean(diff(ALLEEG(subject).etc.analysis.design.time_to_reach_vel_peak)), diff(ALLEEG(subject).etc.analysis.design.time_to_reach_vel_peak)]' / ALLEEG(subject).srate; % from movement onset to peak vel
    decel = [mean(diff(ALLEEG(subject).etc.analysis.design.peak_vel_to_retract)), diff(ALLEEG(subject).etc.analysis.design.peak_vel_to_retract)]' / ALLEEG(subject).srate; % from peak vel to contac
    out_move = [mean(diff(ALLEEG(subject).etc.analysis.design.full_outward_movement)), diff(ALLEEG(subject).etc.analysis.design.full_outward_movement)]' / ALLEEG(subject).srate; % from peak vel to contac
    
    % predictors task
    oddball = ALLEEG(subject).etc.analysis.design.oddball';
    post_error = ["false"; oddball(1:end-1)];
    post_error = double(post_error=='true');
    isitime = ALLEEG(subject).etc.analysis.design.isitime';
    sequence = ALLEEG(subject).etc.analysis.design.sequence';
    trial_number = ALLEEG(subject).etc.analysis.design.trial_number';
    haptics = ALLEEG(subject).etc.analysis.design.haptics';
    direction = categorical(ALLEEG(subject).etc.analysis.design.direction');
    pID = repmat(subject,size(direction,1),1);
    
    reg_t = table(post_error, isitime, sequence, haptics, trial_number, direction, oddball, rt, at, acc, decel, out_move, pID);
    reg_t(ALLEEG(subject).etc.analysis.design.bad_touch_epochs,:)= [];
    
    % match trial count
    match_ixs = find(reg_t.post_error==0);
    mismatch_ixs = find(reg_t.post_error==1);
    match_ixs = randsample(match_ixs, numel(mismatch_ixs));
    reg_t = reg_t(union(match_ixs, mismatch_ixs),:);
    
    all_subjects_reg_t = [all_subjects_reg_t; reg_t]; 
end

%% descriptives behavior

modelfit_full = fitlme(all_subjects_reg_t, 'out_move ~ post_error + (1|pID)')
modelfit_null = fitlme(all_subjects_reg_t, 'out_move ~ 1 + (1|pID)') 

compare(modelfit_null, modelfit_full)

%% fit models predicting behavior

% loop over different statistical results and fits
dv = 'post_error';
models = {' ~ out_move + (1|pID)' ,...
%     ' ~ acc + out_move + (1|pID)' ,...
%     ' ~ at + (1|pID)' ,...
%     ' ~ rt + (1|pID)' ,...
%     ' ~ decel + (1|pID)' ,...
    };

% odds of the trial being post_error given movement parameters and EEG
% signature
for model = models
    
    modelfit = fitglme(all_subjects_reg_t, [dv, model{1}], 'Distribution','binomial'); % + acc + decel    
    ypred = predict(modelfit);
    ypred(ypred>=.5) = 1;
    ypred(ypred<.5) = 0;
    acc = sum(ypred==all_subjects_reg_t.(dv)) / size(all_subjects_reg_t.(dv),1);
    disp([model, ', accuracy: ' num2str(acc)])
    modelfit
    
end

%% interpreting log odds

coefs = exp(modelfit.Coefficients.Estimate);
vals = -.3:0.001:.3;
c = 1;
for v = vals
    fit(c) = exp(coefs(1) + coefs(2)*v) / (1+(exp(coefs(1)*coefs(2)*v)));
    c = c+1;
end
scatter(vals, fit)

% for interpreation see: https://stackoverflow.com/questions/41384075/r-calculate-and-interpret-odds-ratio-in-logistic-regression

%% inspect full table
head(all_subjects_reg_t,3)

%% [] grand average velocity sync/async, aligned at movement onset

fit.model = 'vel ~ haptics*oddball';
for subject = subjects
    
    % predictors task
    onset = ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample';
%     onset = ALLEEG(subject).etc.analysis.design.spawn_event_sample';
    oddball = ALLEEG(subject).etc.analysis.design.oddball';
    post_error = ["false"; oddball(1:end-1)];
    post_error = double(post_error=='true');
    isitime = ALLEEG(subject).etc.analysis.design.isitime';
    sequence = ALLEEG(subject).etc.analysis.design.sequence';
    trial_number = ALLEEG(subject).etc.analysis.design.trial_number';
    haptics = ALLEEG(subject).etc.analysis.design.haptics';
    direction = categorical(ALLEEG(subject).etc.analysis.design.direction');
    pID = repmat(subject,size(direction,1),1);
    
    reg_t = table(post_error, isitime, sequence, haptics, trial_number, direction, oddball, onset, pID);
    reg_t(ALLEEG(subject).etc.analysis.design.bad_touch_epochs,:)= [];
    
    % match trial count
    match_ixs = find(reg_t.post_error==0);
    mismatch_ixs = find(reg_t.post_error==1);
    match_ixs = randsample(match_ixs, numel(mismatch_ixs));
%     reg_t = reg_t(union(match_ixs, mismatch_ixs),:);
    
    % load mocap
    out_folder = fullfile(bemobil_config.study_folder, 'data');
    load(fullfile(out_folder, ['sub-', sprintf('%03d', subject), '_motion.mat'])) % var motion
    motion.mag_vel(:,ALLEEG(subject).etc.analysis.design.bad_touch_epochs) = [];
%     motion.mag_vel = motion.mag_vel(:,union(match_ixs, mismatch_ixs));

    for i = 1:size(motion.mag_vel,2)
        tmp(:,i) = motion.mag_vel(reg_t.onset(i):reg_t.onset(i)+499,i); 
    end
    
    for i = 1:size(tmp,1)
        vel = tmp(i,:)';
        reg_t = addvars(reg_t, vel);
        
        % fit model
        mdl = fitlm(reg_t, fit.model);
        fit.estimates(subject,i,:) = mdl.Coefficients.Estimate;
        
        reg_t = removevars(reg_t, 'vel');
    end
    mean_onset(subject) = ceil(mean(reg_t.onset));
    clear reg_t tmp
end
fit.predictor_names = string(mdl.CoefficientNames);

% statistics
fit.save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/results';
fit.alpha = .05;
fit.perm = 1000;
for i = 2:size(fit.estimates,3)
    
    % get betas per predictor
    betas = fit.estimates(:,:,i);
    betas = permute(betas, [2, 1]);
    zero = zeros(size(betas));

    % permutation t-test
    [~, ~, fit.stats(i).betas_p_vals, fit.stats(i).surrogate_data] = statcond({betas zero},...
        'method', 'perm', 'naccu', fit.perm);

    % compute tfce transform of t_maps surrogate data, add max tfce
    % dist
    for j = 1:size(fit.stats(i).surrogate_data,2)
        tfce(j,:) = limo_tfce(1,fit.stats(i).surrogate_data(:,j)',[],0);
        this_max = tfce(j,:);
        fit.stats(i).tfce_max_dist(j) = max(this_max(:));
    end

    % threshold true t_map
    [~,~,~,STATS] = ttest(permute(betas, [2,1]));
    fit.stats(i).perm_t = STATS;
    fit.stats(i).tfce_true = limo_tfce(1,STATS.tstat,[],0);
    fit.stats(i).tfce_thresh = prctile(fit.stats(i).tfce_max_dist,95);
    fit.stats(i).tfce_sig_mask = fit.stats(i).tfce_true>fit.stats(i).tfce_thresh;
end

% resave res
save(fullfile(fit.save_path, ['behavior_' fit.model '.mat']), 'fit');

%% plot:
%match with times in velocity epoch | %res.t1 = -2444; % -2444 (ersp start) % res.tend = 1440; % 1440 (ersp end)

times = 1000*(bemobil_config.epoching.event_epochs_boundaries(1):(1/ALLEEG(subject).srate):bemobil_config.epoching.event_epochs_boundaries(end));
mean_onsets_times = (mean(mean_onset) - 750) / ALLEEG(subject).srate * 1000;
[~, first_time] = min(abs(times - mean_onsets_times));
end_time = (mean(mean_onset) -750 + 500) / ALLEEG(subject).srate * 1000;
[~, last_time] = min(abs(times-end_time));
t_ixs = times(first_time:last_time);

% prepare plot
xline_zero = 750 - ceil(mean(mean_onset));
normal; % plot normal window, not docked
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 960 100]);

% plot condition 1
colors = brewermap(5, 'Spectral');
colors1 = colors(2, :);
ploterp_lg(sync, [], [], xline_zero, 1, 'ERV m/s', '', [0 .8], colors1, '-');
hold on
% plot condition 2
colors2 = colors(5, :);
ploterp_lg(async, [], [], xline_zero, 1, '', '', [0 .8], colors2, '-.');
% add legend
legend('sync.','async.');

save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/vel_erp_sync_async/';
mkdir(save_path)
print(gcf, [save_path 'vel.eps'], '-depsc');
close(gcf);
clear sync async
