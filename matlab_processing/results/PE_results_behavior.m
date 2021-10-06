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
    ALLEEG(subject).etc.analysis.design.action_time = ...
        (abs(ALLEEG(subject).etc.analysis.design.movements.reach_off_sample) - ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample) / ALLEEG(subject).srate;
    
    ALLEEG(subject).etc.analysis.design.time_to_reach_vel_peak = ...
        ALLEEG(subject).etc.analysis.design.movements.reach_max_vel_ix - ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample;
    
    event_sample_ix = abs(bemobil_config.epoching.event_epochs_boundaries(1)) * ALLEEG(subject).srate; % epoched [-3 2] seconds = 1250 samples
    ALLEEG(subject).etc.analysis.design.peak_vel_to_contact = ...
        event_sample_ix - ALLEEG(subject).etc.analysis.design.movements.reach_max_vel_ix;
    
    ALLEEG(subject).etc.analysis.design.peak_vel_to_stop = ...
        ALLEEG(subject).etc.analysis.design.movements.reach_off_sample - ALLEEG(subject).etc.analysis.design.movements.reach_max_vel_ix;
end

% compute these measures in the results scripts    
%     EEG.etc.analysis.design.reaction_time = (EEG.etc.analysis.design.movements.movement_onset_sample - EEG.etc.analysis.design.spawn_event_sample) / EEG.srate;
%     EEG.etc.analysis.design.action_time = (abs(event_sample_ix) - EEG.etc.analysis.design.movements.movement_onset_sample) / EEG.srate;

%% build full design matrices

all_subjects_reg_t = table();

for subject = subjects
    
    % movement predictors
    at = ALLEEG(subject).etc.analysis.design.action_time';
    rt = ALLEEG(subject).etc.analysis.design.reaction_time';
    ct = rt+at;
    
    peak_vel_reach = ALLEEG(subject).etc.analysis.design.movements.reach_max_vel';
    vel_event = ALLEEG(subject).etc.analysis.motion.mag_vel(event_sample_ix,:)';
    acc = ALLEEG(subject).etc.analysis.design.time_to_reach_vel_peak'/ ALLEEG(subject).srate;
    decel = ALLEEG(subject).etc.analysis.design.peak_vel_to_contact' / ALLEEG(subject).srate;
    decel_stop = ALLEEG(subject).etc.analysis.design.peak_vel_to_stop' / ALLEEG(subject).srate;
    
    % movement predictors: change from previous trial
    diff_rt = [diff(ALLEEG(subject).etc.analysis.design.reaction_time), mean(diff(ALLEEG(subject).etc.analysis.design.reaction_time))]'; % spawn to movement onset
    diff_at = [diff(ALLEEG(subject).etc.analysis.design.action_time), mean(diff(ALLEEG(subject).etc.analysis.design.action_time))]'; % from movement onset to touch
    diff_ct = diff_rt + diff_at;
    diff_acc = [diff(ALLEEG(subject).etc.analysis.design.time_to_reach_vel_peak), mean(diff(ALLEEG(subject).etc.analysis.design.time_to_reach_vel_peak))]' / ALLEEG(subject).srate; % from movement onset to peak vel
    diff_decel = [diff(ALLEEG(subject).etc.analysis.design.peak_vel_to_contact), mean(diff(ALLEEG(subject).etc.analysis.design.peak_vel_to_contact))]' / ALLEEG(subject).srate; % from peak vel to contac
    diff_decel_stop = [diff(ALLEEG(subject).etc.analysis.design.peak_vel_to_stop), mean(diff(ALLEEG(subject).etc.analysis.design.peak_vel_to_stop))]' / ALLEEG(subject).srate; % from peak vel to contac
    
    diff2_at = [0; 0; at(3:end) - at(1:end-2)];
    
    % predictors task
    oddball = double((ALLEEG(subject).etc.analysis.design.oddball=='true'))';
    pre_error = [oddball(2:end); 0] * 2;
    post_error = [0; oddball(1:end-1)] * 3;
    pre_odd_post = sum([pre_error, oddball, post_error],2);
    pre_odd_post(pre_odd_post==5)=3;
    
    isitime = ALLEEG(subject).etc.analysis.design.isitime';
    sequence = ALLEEG(subject).etc.analysis.design.sequence';
    trial_number = ALLEEG(subject).etc.analysis.design.trial_number';
    haptics = double(ALLEEG(subject).etc.analysis.design.haptics)';
    direction = categorical(ALLEEG(subject).etc.analysis.design.direction');
    pID = repmat(subject,size(direction,1),1);
    
    reg_t = table(pID, isitime, sequence, haptics, trial_number, direction, oddball, ...
        post_error, pre_error, pre_odd_post, ...
        at, rt, ct, peak_vel_reach, vel_event, acc, decel, decel_stop, ...
        diff_at, diff_rt, diff_ct, diff_acc, diff_decel, diff_decel_stop, ...
        diff2_at);
    
    reg_t(ALLEEG(subject).etc.analysis.design.bad_touch_epochs,:)= [];
    all_subjects_reg_t = [all_subjects_reg_t; reg_t]; 
end

%% inspect full table
head(all_subjects_reg_t,10)

%% action time differs between oddball and no oddball trials

% remove preerror trials
dmatrix = all_subjects_reg_t;
dmatrix.pre_error = [dmatrix.oddball(2:end); 0];
dmatrix(dmatrix.pre_error==1,:)=[];

% % match trial count
% match_ixs = find(dmatrix.oddball==0);
% mismatch_ixs = find(dmatrix.oddball==1);
% match_ixs = randsample(match_ixs, numel(mismatch_ixs));
% dmatrix = dmatrix(union(match_ixs, mismatch_ixs),:);

% save this dmatrix where results where reported from
% save(fullfile(bemobil_config.study_folder, 'dmatrix'), 'dmatrix');

% seq_1 = find(dmatrix.sequence==1);
% match_ixs_seq_1 = intersect(match_ixs, seq_1);
% mismatch_ixs = randsample(mismatch_ixs, numel(match_ixs_seq_1));
% 
% dmatrix = dmatrix(union(match_ixs_seq_1, mismatch_ixs),:);

% {'at', 'rt', 'acc', 'decel', 'decel_stop', ...
%         'diff_at', 'diff_rt', 'diff_acc', 'diff_decel', 'diff_decel_stop'}

% for dv = {'diff_at'}
% 
%     modelfit_full = fitlme(dmatrix, [dv{1} ' ~ oddball*sequence + (1|pID)']);
%     modelfit_null = fitlme(dmatrix, [dv{1} ' ~ 1 + (1|pID)']);
%     compare(modelfit_null, modelfit_full)
%     
%     disp([dv{1} ' t_stat: ' num2str(modelfit_full.Coefficients.tStat(3))]);
% end

% summary stats and raincloud plot
summary_diff_at = groupsummary(dmatrix,{'pID','oddball'},'mean','at');
a = summary_diff_at(summary_diff_at.oddball==0,:);
b = summary_diff_at(summary_diff_at.oddball==1,:);
to_plot = [a.mean_at, b.mean_at];
mean(to_plot)
std(to_plot)
% ans =
% 
%     0.7418    0.6940
% 
% 
% ans =
% 
%     0.1495    0.1532

% summary stats and raincloud plot
summary_diff_at = groupsummary(dmatrix,{'pID','oddball'},'mean','diff_at');
a = summary_diff_at(summary_diff_at.oddball==0,:);
b = summary_diff_at(summary_diff_at.oddball==1,:);
to_plot = [a.mean_diff_at, b.mean_diff_at];
mean(to_plot)
std(to_plot)
% ans =
% 
%     0.0007    0.0468
% 
% 
% ans =
% 
%     0.0153    0.0362

colors = brewermap(5, 'Spectral');
colors1 = colors(2, :);
colors2 = colors(5, :);

fig_position = [200 200 600 400]; % coordinates for figures
alpha = .4;
fig_dir = '/Users/lukasgehrke/Documents/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/behavior/';% fullfile(bemobil_config.study_folder, 'results', 'behavior');
fig_name = 'rate_of_change_action_time';

f7 = figure('Position', fig_position);
h1 = raincloud_plot(to_plot(:,1), 'box_on', 1, 'color', colors1, 'alpha', alpha,...
     'box_dodge', 1, 'box_dodge_amount', .15, 'dot_dodge_amount', .15,...
     'box_col_match', 0);
h2 = raincloud_plot(to_plot(:,2), 'box_on', 1, 'color', colors2, 'alpha', alpha,...
     'box_dodge', 1, 'box_dodge_amount', .35, 'dot_dodge_amount', .35, 'box_col_match', 0);
l = legend([h1{1} h2{1}], {'Match to Match Trial', 'Following Mismatch Trial'});
l.FontSize = 20;

h1{2}.SizeData = 40;
h1{2}.MarkerEdgeColor = 'k';
h1{2}.MarkerEdgeAlpha = alpha;
h1{3}.FaceColor = [colors1, alpha];

h2{2}.SizeData = 40;
h2{2}.MarkerEdgeColor = 'k';
h2{2}.MarkerEdgeAlpha = alpha;
h2{3}.FaceColor = [colors2, alpha];

t = title(['Rate of Change in Action Time']);
t.FontSize = 20;

xlabel('change in action time (s)')
ylabel('density')

set(gca,'XLim', [-.1 .2], 'YLim', [-35 45]);
set(gca,'FontSize',22)
box off
grid on

% save
if ~exist(fig_dir)
    mkdir(fig_dir);
end
print(f7, fullfile(fig_dir, [fig_name, '_all_trials_.eps']), '-depsc');
close(gcf);

%% model rate of change in action time

% modelfit_full = fitlme(dmatrix, 'diff_at ~ haptics + sequence + (1|pID)')
% modelfit_full = fitlme(dmatrix, 'diff_at ~ oddball*haptics + (1|pID)')
% 
% summary_diff_at = groupsummary(dmatrix,{'pID','oddball','haptics'},'mean','diff_at');
% modelfit = fitlme(summary_diff_at, 'mean_diff_at ~ oddball*haptics + (1|pID)')

% modelfit = fitlme(dmatrix, 'at ~ oddball*sequence + (1|pID)')

modelfit = fitlme(dmatrix, 'diff_at ~ oddball + (1|pID)')
modelfit_oddball = fitlme(dmatrix, 'diff_at ~ 1 + (1|pID)');
compare(modelfit_oddball, modelfit)

% why this model?
% to check whether the introduction of vibrotactile feedback altered
% participants behavioral adaptation
% -> it did not -> maybe because vibrotactile feedback renders the sense of
% surface touching and hence does not impact the timing of expected resistive, force feedback
 
%% cross validate across participants

k = 10;
indices = crossvalind('Kfold',dmatrix.oddball,k);
for i = 1:k
    test = indices==i;
    train = ~test;
    
    modelfit = fitglme(dmatrix(train,:), 'oddball ~ diff_at + (1|pID)');
%     modelfit = fitglme(dmatrix(train,:), 'oddball ~ diff_at*sequence + (1|pID)', 'Distribution','binomial');
%     modelfit = fitglme(dmatrix(train,:), 'oddball ~ diff_at + haptics + (1|pID)', 'Distribution','binomial');
%     modelfit = fitlme(dmatrix(train,:), 'diff_at ~ vel_event + (1|pID)');
    
    ypred = predict(modelfit, dmatrix(test,:));
    ypred(ypred>=.5) = 1;
    ypred(ypred<.5) = 0;
    accuracy(i) = sum(ypred==dmatrix(test,:).oddball) / size(dmatrix(test,:).oddball,1);
    
    pconf = simulateChance(round(sum(test)/2) * [1 1], .05);
    sim_chance(i) = pconf(3);

    disp(['accuracy: ' num2str(accuracy(i))])
end

[H,P,CI,STATS] = ttest(accuracy,sim_chance)

%% cv within participants

k = 5;
for subject = subjects
    dmatrix_s = dmatrix(dmatrix.pID==subject,:);
    
    for i = 1:k
    
        indices = crossvalind('Kfold',dmatrix_s.oddball,k);
        test = indices==i;
        train = ~test;
        
        modelfit = fitglme(dmatrix_s(train,:), 'oddball ~ diff_at + (1|pID)', 'Distribution','binomial'); 
        ypred = predict(modelfit, dmatrix_s(test,:));
        
        ypred(ypred>=.5) = 1;
        ypred(ypred<.5) = 0;        

        acc(i) = sum(ypred==dmatrix_s(test,:).oddball) / size(dmatrix_s(test,:).oddball,1);
        pconf = simulateChance(round(sum(test)/2) * [1 1], .05);
        sim_chance(i) = pconf(3);
%         disp(['accuracy: ' num2str(acc(i))])

%         modelfit = fitglme(dmatrix_s, 'oddball ~ diff_at + (1|pID)', 'Distribution','binomial');
%         ypred = predict(modelfit, dmatrix_s);
%         acc(subject) = sum(ypred==dmatrix_s.oddball) / size(dmatrix_s.oddball,1);
%         pconf = simulateChance(round(size(dmatrix_s,1)/2) * [1 1], .05);
%         sim_chance(subject) = pconf(3);
%         disp(['accuracy: ' num2str(acc)])

    end
    
    accuracies(subject) = mean(acc);
    chances(subject) = mean(sim_chance);
    
end

mean(accuracies)
std(accuracies)

mean(chances)
std(chances)

[H,P,CI,STATS] = ttest(accuracies,chances)

%% interpreting log odds

modelfit = fitglme(dmatrix, 'oddball ~ diff_at + (1|pID)', 'Distribution','binomial');

coefs = exp(modelfit.Coefficients.Estimate);
vals = -.3:0.001:.3;
c = 1;
for v = vals
    fit(c) = exp(coefs(1) + coefs(2)*v) / (1+(exp(coefs(1)*coefs(2)*v)));
    c = c+1;
end
scatter(vals, fit)

% for interpreation see: https://stackoverflow.com/questions/41384075/r-calculate-and-interpret-odds-ratio-in-logistic-regression

%% plot vel profiles with movement onset and offset markers main effect oddball:
%match with times in velocity epoch | %res.t1 = -2444; % -2444 (ersp start) % res.tend = 1440; % 1440 (ersp end)

match = [];
mismatch = [];
buffer = 100;

for subject = subjects

    clear move
    
    tmp_move = ALLEEG(subject).etc.analysis.motion.mag_vel;
    bad = ALLEEG(subject).etc.analysis.design.bad_touch_epochs;
    tmp_move(:,bad) = [];
    move = zeros(size(tmp_move));
    
    for i = 1:size(tmp_move,2)
        spawns = ALLEEG(subject).etc.analysis.design.spawn_event_sample;
        spawns(bad) = [];
        move(1:1251-(spawns(i)-buffer),i) = tmp_move(spawns(i)-buffer:end,i);
    end
    
    % split by condition, remove bad epochs
    oddball = ALLEEG(subject).etc.analysis.design.oddball;
    oddball(bad) = [];
    match_ix = oddball=='false';
    mismatch_ix = oddball=='true';
    
    direction = ALLEEG(subject).etc.analysis.design.direction;
    direction(bad) = [];
    direction_left_ix = direction=='left';
    
    match(subject,:) = squeezemean(move(:,match_ix),2);
    match_left(subject,:) = squeezemean(move(:,match_ix&direction_left_ix),2);
    
    mismatch(subject,:) = squeezemean(move(:,mismatch_ix),2);
    mismatch_left(subject,:) = squeezemean(move(:,mismatch_ix&direction_left_ix),2);
    
    spawns = ALLEEG(subject).etc.analysis.design.spawn_event_sample;
    spawns(bad) = [];
    match_touch(subject) = 750 - ceil(mean(spawns(match_ix)));
    mismatch_touch(subject) = 750 - ceil(mean(spawns(mismatch_ix)));
    
    all_spawns(subject) = ceil(mean(ALLEEG(subject).etc.analysis.design.spawn_event_sample));
    
    reach_on(subject) = ceil(mean(ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample));
%     mismatch_reach_on(subject) = ceil(mean(ALLEEG(subject).etc.analysis.design.movements.reach_onset_sample(mismatch_ix)));
    
    match_reach_off(subject) = ceil(mean(ALLEEG(subject).etc.analysis.design.movements.reach_off_sample(match_ix)));
    mismatch_reach_off(subject) = ceil(mean(ALLEEG(subject).etc.analysis.design.movements.reach_off_sample(mismatch_ix)));
    
end

% match(isnan(match)) = 0;

% % align at reach onset
% match_win = ceil(mean(match_reach_on))-100:ceil(mean(match_reach_on))+500;
% match = match(:,match_win);
% 
% mismatch_win = ceil(mean(mismatch_reach_on))-100:ceil(mean(mismatch_reach_on))+500;
% mismatch = mismatch(:,mismatch_win);

%% add lines
spawns = ceil(mean(all_spawns));
reach_onset = ceil(mean(reach_on));
reach_onset_marker = (reach_onset - spawns) / ALLEEG(1).srate;

% must be per condition
reach_off = ceil(mean(match_reach_off));
reach_off_marker = (reach_off - spawns) / ALLEEG(1).srate;
mis_reach_off = ceil(mean(mismatch_reach_off));
mis_reach_off_marker = (mis_reach_off - spawns) / ALLEEG(1).srate;

% touch
match_touch = ceil(mean(match_touch)) / ALLEEG(1).srate;
mismatch_touch = ceil(mean(mismatch_touch)) / ALLEEG(1).srate;

match = match(:,1:end-375);
mismatch = mismatch(:,1:end-375);

%% make plot
normal;

xline_zero = buffer;
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 600 200]);

% plot condition 1
colors = brewermap(5, 'Spectral');
colors1 = colors(2, :);
ploterp_lg(match, [], [], xline_zero, 1, 'ERV m/s', 'time (s)', [0 .8], colors1, '-');
hold on

% plot condition 2
colors2 = colors(5, :);
ploterp_lg(mismatch, [], [], xline_zero, 1, '', '', [0 .8], colors2, '-.');
legend('','match','mismatch','');

markers = [reach_onset_marker, reach_off_marker, match_touch, mismatch_touch];
markers_label = {'on', 'off', 'match', 'mismatch'};
for i = 1:numel(markers)
    [l, h] = vline(markers(i),'.',markers_label{i});
    
    if i<= 2
        l.LineStyle = '-.';
    else
        l.LineStyle = ':';
    end
    l.Color = '#636978';
    l.LineWidth = 3;
    h.FontSize = 24;
end

save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/behavior/';
mkdir(save_path)
print(gcf, [save_path 'vel.eps'], '-depsc');
close(gcf);

%% descriptives

mean(match_touch/ ALLEEG(1).srate)
std(match_touch/ALLEEG(1).srate)

mean(mismatch_touch/ ALLEEG(1).srate)
std(mismatch_touch/ALLEEG(1).srate)

modelfit = fitlme(dmatrix, 'at ~ oddball*haptics + (1|pID)')
summary_at = groupsummary(dmatrix,{'oddball'},'mean','at')
summary_at = groupsummary(dmatrix,{'oddball'},'std','at')

modelfit = fitlme(dmatrix, 'diff_at ~ oddball*haptics + (1|pID)')
summary_at = groupsummary(dmatrix,{'oddball'},'mean','diff_at')
summary_at = groupsummary(dmatrix,{'oddball'},'std','diff_at')



%% [do i still need that? I dont think so...] grand average velocity sync/async, aligned at movement onset

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
