%% clear all and load params
close all; clear

PE_config;

%% load study : 2nd Level inference

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 1, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

% load IMT_v1 EEGLAB study struct, keeping at most 1 dataset in memory
input_path_STUDY = [bemobil_config.study_folder bemobil_config.study_level];
if isempty(STUDY)
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    [STUDY ALLEEG] = pop_loadstudy('filename', bemobil_config.study_filename, 'filepath', input_path_STUDY);
    CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
    
    eeglab redraw
end
STUDY_sets = cellfun(@str2num, {STUDY.datasetinfo.subject});

% select subjects out of clusters of int
clusters_of_int = [3, 7, 9, 24, 25, 28, 30, 33, 34];
% clusters
% 3: right parietal
% 7: right motor?
% 24: right SMA
% 25: left parietal
% 28: interesting
% 33: ACC

channels_of_int = [5, 25, 65];
% channels
% 5: Fz
% 25: Pz
% 65: FCz

%% FINAL result 0.1: Histograms of reaction times mismatch and match trials

all_match_rt = [];
all_mismatch_rt = []; 

% load data
for s = ALLEEG
        
    % calculate reaction times (time between box:spawned and box:touched)
    % of mismatch and match
    rt_all = str2double({s.epoch.eventreaction_time});
    condition = categorical({s.epoch.eventnormal_or_conflict});
    conflict_trials = condition=="conflict";
    rt_mismatch = rt_all(conflict_trials); 
    rt_match = rt_all(~conflict_trials);
        
    all_mismatch_rt = [all_mismatch_rt rt_mismatch];
    all_match_rt = [all_match_rt rt_match];
end

% print means
disp(["mean reaction time matching feedback: " mean(all_match_rt)]);
disp(["mean reaction time mismatching feedback: " mean(all_mismatch_rt)]);
disp(["difference between matching and mismatching feedback: " mean(all_match_rt) - mean(all_mismatch_rt)]);

% plot
map = brewermap(2,'Set1'); 
figure('Renderer', 'painters', 'Position', [10 10 450 300])
grid on;
hold on;
h1 = histfit(all_match_rt, 50);
h2 = histfit(all_mismatch_rt, 50);

h1(1).FaceColor = map(1,:);
h1(1).EdgeColor = 'none';
h1(1).FaceAlpha = .5;
%h1(2).Color = map(1,:);
%set(get(get(h1(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
delete(h1(2))

h2(1).FaceColor = map(2,:);
h2(1).EdgeColor = 'none';
h2(1).FaceAlpha = .7;
%h2(2).Color = map(2,:);
%set(get(get(h2(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
delete(h2(2))

l = line([mean(all_match_rt) mean(all_match_rt)], [0 500]);
l.Color = map(1,:);
l.LineWidth = 3;

l2 = line([mean(all_mismatch_rt) mean(all_mismatch_rt)], [0 500]);
l2.Color = map(2,:);
l2.LineWidth = 3;

%format plot
set(gca,'FontSize',20)
box off
l = legend('match', 'mismatch',...
    'location','northeast');
legend boxoff
xlabel('seconds')
ylabel('frequency')

%% FINAL result 0.2: histogram velocity profile / mean velocity curve with confidence interval

buffertime = .5;
c = 1;
for s = ALLEEG
    
    rt_all = str2double({s.epoch.eventreaction_time});
    condition = categorical({s.epoch.eventnormal_or_conflict});
    conflict_trials = condition=="conflict";
    rt_mismatch = rt_all(conflict_trials); %mean(a(conflict_trials));
    rt = rt_mismatch * s.srate;
    
    buffer = buffertime * s.srate;
    event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * s.srate);

    event_start = event_onset - rt;
    event_end = event_start + rt + buffer;
    event_wins = floor([event_start; event_end]');

    measure = s.etc.analysis.mocap.mag_vel(:,conflict_trials);
    
    % get windows
    for i = 1:size(event_wins,1)
        data(event_wins(i,1):event_wins(i,2),:) = measure(event_wins(i,1):event_wins(i,2),:);
    end
    
    all_dat(c,:,:) = squeeze(nanmean(data,2));
    c = c+1;
    
    clear data
end

% plot mean velocity profile with 2*sd
start = 550;
vel = all_dat(:,start:end);
mean_vel = mean(vel,1);
% Calculate Standard Error Of The Mean
SEM_A = std(vel', [], 2)./ sqrt(size(vel',2));
% 95% Confidence Intervals
CI95 = bsxfun(@plus, mean(vel',2), bsxfun(@times, [-1  1]*1.96, SEM_A));
upper = CI95(:,2)';
lower = CI95(:,1)';

% prepare labels
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * s.srate);
xtlabels = start:size(all_dat,2);
xtlabels = xtlabels / s.srate;
xtlabels = xtlabels - xtlabels(event_onset-start);

% make plot
map = brewermap(2,'Set1'); 
figure('Renderer', 'painters', 'Position', [10 10 450 300])
p = patch([xtlabels fliplr(xtlabels)], [lower fliplr(upper)], 'g');
p.FaceColor = map(1,:);
p.FaceAlpha = .5;
p.EdgeColor = 'none';
hold on;
l = plot(xtlabels, mean_vel);
l.MarkerFaceColor = map(2,:);
l.LineWidth = 3;

% line at 0
l2 = line([0 0], [0 .8]);
l2.LineStyle = '-';
l2.Color = 'k';
l2.LineWidth = 2;

% % more lines at sampled velocities
% cols = brewermap(size(ts_of_ints,2), 'Spectral');
% for i = 1:size(ts_of_ints,2)
%     ts = (ts_of_ints(i)-event_onset)/250;
%     l = line([ts ts], [0 .8]);
%     l.LineStyle = '-.';
%     l.Color = cols(end-i+1,:);
%     l.LineWidth = 2;
% end
% legend('95% confidence interval', 'hand velocity', 'collision', 'sampled velocities');

legend('95% CI', 'velocity');
legend boxoff
axis tight
% grid on
set(gca,'FontSize',20)
ylabel('magnitude of velocity');
xlabel('seconds');

%% FINAL result 0.3: plot 3D trajectory colorcoded by hand velocity (and erp amplitude (erpimage2D))

% 1: make hand velocity at each x,y for each subject, then plot average
% across subjects and smooth

% 2: make erp amplitude (erpimage2D) at each (x,y) for each subject, then
% plot average across subjects and smooth it -> this is important and cool

buffertime = 0;
all_dat = NaN(1,4);

for subject = subjects(1)

    s = ALLEEG(subject);
    input_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder ...
        bemobil_config.filename_prefix num2str(subject)];
    mocap = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
        bemobil_config.merged_filename_mocap], 'filepath', input_filepath);

    % get reaction times mismatch trials to extract windows
    rt_all = str2double({s.epoch.eventreaction_time});
    condition = categorical({s.epoch.eventnormal_or_conflict});
    conflict_trials = condition=="conflict";
    rt_mismatch = rt_all(conflict_trials); %mean(a(conflict_trials));
    rt = rt_mismatch * s.srate;

    buffer = buffertime * s.srate;
    event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * s.srate);

    event_start = event_onset - rt;
    event_end = event_start + rt + buffer;
    event_wins = floor([event_start; event_end]');

    x = squeeze(s.etc.analysis.mocap.x(1,:,conflict_trials));
    y = squeeze(s.etc.analysis.mocap.y(1,:,conflict_trials));
    z = squeeze(s.etc.analysis.mocap.z(1,:,conflict_trials));
    vel = s.etc.analysis.mocap.mag_vel(:,conflict_trials);

    % get windows
    data = NaN(size(x,1),size(x,2),4);
    for i = 1:size(event_wins,1)
        data(event_wins(i,1):event_wins(i,2),:,1) = x(event_wins(i,1):event_wins(i,2),:);
        data(event_wins(i,1):event_wins(i,2),:,2) = y(event_wins(i,1):event_wins(i,2),:);
        data(event_wins(i,1):event_wins(i,2),:,3) = z(event_wins(i,1):event_wins(i,2),:);
        data(event_wins(i,1):event_wins(i,2),:,4) = vel(event_wins(i,1):event_wins(i,2),:);
    end

    % standardize offset between subjects
    data(:,:,1) = data(:,:,1) - data(event_wins(1),:,1);
    data(:,:,2) = data(:,:,2) - data(event_wins(1),:,2);
    data(:,:,3) = data(:,:,3) - data(event_wins(1),:,3);

    % reshape to long format concatenating trials as rows
    data_long = reshape(data,[],size(data,3),1);

    % remove noisy trials
    dev = 3*nanstd(data_long(:,4));
    rem_ix = find(data_long(:,4)>dev);
    data_long(rem_ix,:) = [];
    all_dat = vertcat(all_dat, data_long);
end

% plot of param on lines
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 450 300]);
% % restrict colorbar
% max_lim = max(all_dat(:,4));
% min_lim = min(all_dat(:,4));  
% lim = min(abs([max_lim, min_lim])) / 2;
% all_dat(:,4) = min( max(all_dat(:,4), -1*lim), lim); % scale to lim
% plot
fig_erpimage2d = patch([all_dat(:,1)' nan],[all_dat(:,3)' nan],[all_dat(:,2)' nan],[all_dat(:,4)' nan],...
    'LineWidth', 2, 'FaceColor','none','EdgeColor','interp'); grid on;
view(3);
%ylim([-.05, .3]);
%xlim([-.2 .2]);
set(gca,'FontSize',20)
cbar; %title('velocity')
set(gca,'FontSize',20)
% clear data
clear all_dat

%% FINAl discussion results: single trial regression velocity ERP

% Questions:
% 1. do participants slow down more in mismatch trials with haptics?
% significant effect that this participant slowed down more in haptic
% mismatch trials compared to just visual feedback

% 2. Are trials after mismatch trials differently initiated? this is 
% interesting but not much there, must check across subjects

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

% get betas from single subject level of design:
models = {'vel_erp_sample ~ haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ after_mismatch * haptics + trial_nr + direction + was_sequence'};
robustfit = 0; % fit robust regression with squared weights, see fitlm
event_sample = 750;
window = event_sample-250:event_sample+250; %[-1 1]seconds start and end of interesting, to be analyzed, samples
count = 1;

for model = models
    model = model{1};

    % outpath
    save_fpath = [bemobil_config.study_folder bemobil_config.study_level ...
        'analyses/mocap/' bemobil_config.study_filename(1:end-6) ...
        '/vel/' model];
    if ~exist(save_fpath, 'dir')
        mkdir(save_fpath);
    end        

    % load data
    count = 1;
    for s = ALLEEG
        disp(['Now running analysis for velocity and subject: ' num2str(count+1)]);

        %DESIGN make continuous and dummy coded predictors
        congruency = s.etc.epoching.oddball';
        haptics = s.etc.epoching.haptics';
        trial_nr = s.etc.epoching.trial_number';
        direction = categorical(s.etc.epoching.direction)';
        sequence = s.etc.epoching.sequence';
        after_mismatch = [0; congruency(1:end-1)];
        was_sequence = [0 s.etc.epoching.sequence(1:end-1)]';
        if strcmp(model, models{1})
            sequence = sequence(congruency);
            direction = direction(congruency);
            trial_nr = trial_nr(congruency);
            haptics = haptics(congruency);
        end

        tic
        for sample = window

            vel_erp_sample = s.etc.analysis.mocap.mag_vel(sample,:)';
            
            % design matrix per sample
            if strcmp(model, models{1})
                vel_erp_sample = vel_erp_sample(congruency);                
                design = table(vel_erp_sample, haptics, trial_nr, direction, sequence);
            else
                design = table(vel_erp_sample, haptics, trial_nr, direction, sequence, after_mismatch, was_sequence);
            end
            
            if robustfit
                mdl = fitlm(design, model, 'RobustOpts', 'on');
            else
                mdl = fitlm(design, model);
            end

            ix = sample-window(1)+1;
            res.betas(count,ix,:) = mdl.Coefficients.Estimate;
            res.t(count,ix,:) = mdl.Coefficients.tStat;
            res.p(count,ix,:) = mdl.Coefficients.pValue;
            res.r2(count,ix,:) = mdl.Rsquared.Ordinary;
            res.adj_r2(count,ix,:) = mdl.Rsquared.Adjusted;
        end
        toc
        count = count + 1;
    end

    % add parameter names
    res.model = model;
    res.parameter_names = string(mdl.CoefficientNames)';
    %save([save_fpath '/res_' model '_robust-' num2str(robustfit) '.mat'], 'res');

    %LIMO ttests
    % settings
    sig_alpha = .05;
    save_info.robustfit = 0;
    save_info.model = model;
    save_info.load_p = save_fpath;
    save_info.parameters = res.parameter_names;

    % main effects: run limo ttest against 0
    for i = 1:size(save_info.parameters,1)
        save_info.parameter = save_info.parameters{i};
        PE_limo(save_info, res, 0, [], [], [], [], []);

        % load LIMO output: save mean value and sig. mask to res and resave res
        save_name = regexprep(save_info.parameter, ':' , '_');
        save_name = regexprep(save_name, '(' , '');
        save_name = regexprep(save_name, ')' , '');
        load([save_info.load_p '/ttest_' save_name '/one_sample_ttest_parameter_1.mat']);
        res.ttest.(save_name).mean_value = squeeze(one_sample(1,:,1)); % mean betas in this case

        % mcc
        % load bootstrap results
        load([save_info.load_p '/ttest_' save_name '/H0/tfce_H0_one_sample_ttest_parameter_1.mat']);
        % get max dist of tfce
        for j = 1:size(tfce_H0_one_sample,3)
            this_tfce = squeeze(tfce_H0_one_sample(1,:,j));
            max_dist(j) = max(this_tfce(:));
        end
        % threshold
        thresh = prctile(max_dist, (1-sig_alpha)*100);
        % load true result
        load([save_info.load_p '/ttest_' save_name '/tfce/tfce_one_sample_ttest_parameter_1.mat']);
        % threshold true data with bootstrap prctile thresh
        res.ttest.(save_name).sig_mask = find(tfce_one_sample>thresh);    
    end
    
    % save res
    save([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '.mat'], 'res');

    % clear results struct
    clear res
end
