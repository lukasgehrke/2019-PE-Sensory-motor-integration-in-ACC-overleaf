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

%% result 0.0: plot cluster blobs and talairach coordinates

% save dipole location
for cluster = clusters_of_int
    loc(cluster,1:3) = STUDY.cluster(cluster).dipole.posxyz;
    loc(cluster,4) = size(unique(STUDY.cluster(cluster).sets),2);
    loc(cluster,5) = cluster;
end
rem = sum(loc,2) == 0;
loc(rem, :) = [];
save('/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/dipole_locs.mat', 'loc');
clear loc

%% plot clusters dipoleclusters
colors = brewermap(size(clusters_of_int,2),'Set1');
std_dipoleclusters(STUDY, ALLEEG, 'clusters', 24,...
    'centroid', 'add',...
    'projlines', 'on',...
    'viewnum', 4,...
    'colors', colors);

%% makotos NIMAs Blobs
voxelSize      = 4;
FWHM           = 10;
blobOrVoxelIdx = 1;
uniformAlpha   = .4;
optionStr = '''separateColorList'',[0.8941 0.1020 0.1098; 0.2157 0.4941 0.7216; 0.3020 0.6863 0.2902; 0.5961 0.3059 0.6392]';

% plot NIMAs Blobs
% Obtain cluster dipoles.
clusterDipoles = std_readdipoles(STUDY, ALLEEG, clusters_of_int);

% Obtain cluster dipole locations.
clusterDipoleLocations = cell(1,length(clusters_of_int));
for clsIdx = 1:length(clusters_of_int)
    currentClsDip = clusterDipoles{clsIdx};
    
    % Dual dipoles are counted as two.
    dipList = [];
    for dipIdx = 1:length(currentClsDip)
        currentDipList = round(currentClsDip(dipIdx).posxyz);
        dipList = cat(1, dipList, currentDipList);
    end
    
    % Store as cluster dipole locations.
    clusterDipoleLocations{1,clsIdx} = dipList;
end

% Parse optional input to construct string-value pairs if it is not empty.
if isempty(optionStr)
    optionalInput = {};
else
    commaIdx = strfind(optionStr, ',');
    for optionalInputIdx = 1:length(commaIdx)+1
        
        if optionalInputIdx == 1;
            currentStrings = optionStr(1:commaIdx(1)-1);
        elseif optionalInputIdx == length(commaIdx)+1
            currentStrings = optionStr(commaIdx(end):length(optionStr));
        else
            currentStrings = optionStr(commaIdx(optionalInputIdx-1)+1:commaIdx(optionalInputIdx)-1);
        end
        currentStrings = strtrim(currentStrings);
        
        if mod(optionalInputIdx,2) == 0 % Odd numbers are strings, even numbers are values.
            optionalInput{optionalInputIdx} = str2num(currentStrings);
        else
            optionalInput{optionalInputIdx} = currentStrings(2:end-1);
        end
    end
end
nimasImagesfromMpA(clusterDipoleLocations, 'voxelSize', voxelSize,...
                                           'FWHM', FWHM,...
                                           'blobOrVoxel', blobOrVoxelIdx,...
                                           'uniformAlpha', uniformAlpha,...
                                           optionalInput);

%% result 0.1: Histograms of reaction times mismatch and match trials

all_match_rt = [];
all_mismatch_rt = []; 

for s = ALLEEG
    
    all_mismatch_rt = [all_mismatch_rt s.etc.epoching.latency_diff_mismatch];
    all_match_rt = [all_match_rt s.etc.epoching.latency_diff_match];
    
end

% covert to time
all_mismatch_rt = all_mismatch_rt/s.srate;
all_match_rt =  all_match_rt/s.srate;

map = brewermap(2,'Set1'); 
all_match_rt(all_match_rt>2) = [];

figure;
grid on;
hold on;
h1 = histfit(all_match_rt, 50);
h2 = histfit(all_mismatch_rt, 50);

h1(1).FaceColor = map(1,:);
h1(1).EdgeColor = 'none';
h1(1).FaceAlpha = .5;
h1(2).Color = map(1,:);

h2(1).FaceColor = map(2,:);
h2(1).EdgeColor = 'none';
h2(1).FaceAlpha = .7;
h2(2).Color = map(2,:);

mean(all_match_rt)
l = line([mean(all_match_rt) mean(all_match_rt)], [0 500]);
l.Color = map(1,:);
l.LineWidth = 2;

mean(all_mismatch_rt)
l2 = line([mean(all_mismatch_rt) mean(all_mismatch_rt)], [0 500]);
l2.Color = map(2,:);
l2.LineWidth = 2;

mean(all_match_rt) - mean(all_mismatch_rt)

%format plot
set(gca,'FontSize',20)
box off
l = legend('match', 'fit',...
    'mismatch', 'fit',...
    'location','northeast');
legend boxoff
xlabel('seconds')
ylabel('frequency')

%% result 0.2: histogram velocity profile / mean velocity curve with confidence interval

cond = 'vibro';
trials = 'good_vibro';
buffertime = .5;
c = 1;
for s = ALLEEG
    
    rt = s.etc.epoching.latency_diff_mismatch(s.etc.epoching.(trials));
    buffer = buffertime * s.srate;
    event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * s.srate);

    event_start = event_onset - rt;
    event_end = event_start + rt + buffer;
    event_wins = floor([event_start; event_end]');

    x = s.etc.analysis.mocap.(cond).x;
    y = s.etc.analysis.mocap.(cond).y;
    z = s.etc.analysis.mocap.(cond).z;
    measure = s.etc.analysis.mocap.(cond).mag_vel;
    
    % get windows
    for i = 1:size(event_wins,1)
        data(event_wins(i,1):event_wins(i,2),:,1) = x(event_wins(i,1):event_wins(i,2),:);
        data(event_wins(i,1):event_wins(i,2),:,2) = y(event_wins(i,1):event_wins(i,2),:);
        data(event_wins(i,1):event_wins(i,2),:,3) = z(event_wins(i,1):event_wins(i,2),:);
        data(event_wins(i,1):event_wins(i,2),:,4) = measure(event_wins(i,1):event_wins(i,2),:);
    end
    
    % standardize offset between subjects
    data(:,:,1) = data(:,:,1) - data(event_wins(1),:,1);
    data(:,:,2) = data(:,:,2) - data(event_wins(1),:,2);
    data(:,:,3) = data(:,:,3) - data(event_wins(1),:,3);
    
    all_dat(c,:,:) = squeeze(nanmean(data,2));
    c = c+1;
    
    clear data
end

% plot mean velocity profile with 2*sd
start = 380;
vel = all_dat(:,start:end,4);
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
figure;
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
l2.LineWidth = 8;

% more lines at sampled velocities
cols = brewermap(size(ts_of_ints,2), 'Spectral');
for i = 1:size(ts_of_ints,2)
    ts = (ts_of_ints(i)-event_onset)/250;
    l = line([ts ts], [0 .8]);
    l.LineStyle = '-.';
    l.Color = cols(end-i+1,:);
    l.LineWidth = 2;
end

legend('95% confidence interval', 'hand movement', 'collision', 'sampled velocities');
legend boxoff
axis tight
% grid on
set(gca,'FontSize',20)
ylabel('magnitude of velocity');
xlabel('seconds');

%% result 0.3: plot 3D trajectory colorcoded by hand velocity and erp amplitude (erpimage2D)

% 1: make hand velocity at each x,y for each subject, then plot average
% across subjects and smooth

% 2: make erp amplitude (erpimage2D) at each (x,y) for each subject, then
% plot average across subjects and smooth it -> this is important and cool

cond = 'vibro';
trials = 'good_vibro';
chan = 25;
buffertime = 0;
clusters_of_int = 1:size(STUDY.cluster,2);

for cluster = clusters_of_int

    % get matching datasets from EEGLAB Study struct
    unique_setindices = unique(STUDY.cluster(cluster).sets);
    unique_subjects = STUDY_sets(unique_setindices);
    all_setindices = STUDY.cluster(cluster).sets;
    all_sets = STUDY_sets(all_setindices);
    all_comps = STUDY.cluster(cluster).comps;

    c = 1;
    all_dat = NaN(1,4);

    for subject = unique_setindices(1:6)

        s = ALLEEG(subject);

        rt = s.etc.epoching.latency_diff_mismatch(s.etc.epoching.(trials));
        buffer = buffertime * s.srate;
        event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * s.srate);

        event_start = event_onset - rt;
        event_end = event_start + rt + buffer;
        event_wins = floor([event_start; event_end]');

        x = s.etc.analysis.mocap.(cond).x;
        y = s.etc.analysis.mocap.(cond).y;
        z = s.etc.analysis.mocap.(cond).z;
    %     measure = s.etc.analysis.mocap.(cond).mag_vel;

        % average ERP if more than 1 comp
        compos = all_comps(all_sets==subject+1);
        if compos > 1
            s.etc.analysis.erp.base_corrected.visual.comps(compos(1),:,:) = ...
                mean(s.etc.analysis.erp.base_corrected.visual.comps(compos,:,:),1);
            s.etc.analysis.erp.base_corrected.vibro.comps(compos(1),:,:) = ...
                mean(s.etc.analysis.erp.base_corrected.vibro.comps(compos,:,:),1);
        end
        measure = squeeze(s.etc.analysis.erp.base_corrected.(cond).comps(compos(1),:,:));

        % get windows
        data = NaN(size(x,1), size(x,2),4);
        for i = 1:size(event_wins,1)
            data(event_wins(i,1):event_wins(i,2),:,1) = x(event_wins(i,1):event_wins(i,2),:);
            data(event_wins(i,1):event_wins(i,2),:,2) = y(event_wins(i,1):event_wins(i,2),:);
            data(event_wins(i,1):event_wins(i,2),:,3) = z(event_wins(i,1):event_wins(i,2),:);
            data(event_wins(i,1):event_wins(i,2),:,4) = measure(event_wins(i,1):event_wins(i,2),:);
        end

        % standardize offset between subjects
        data(:,:,1) = data(:,:,1) - data(event_wins(1),:,1);
        data(:,:,2) = data(:,:,2) - data(event_wins(1),:,2);
        data(:,:,3) = data(:,:,3) - data(event_wins(1),:,3);

        % reshape to long formant concatenating trials as rows
        data_long = reshape(data,[],size(data,3),1);

        % remove noisy trials
        dev = 3*nanstd(data_long(:,4));
        rem_ix = find(data_long(:,4)>dev);
        data_long(rem_ix,:) = [];

        all_dat = vertcat(all_dat, data_long);
        c = c+1;
    end

    % plot of param on lines
    figure('visible','off', 'Renderer', 'painters', 'Position', [10 10 900 900]);
    % restrict colorbar
    max_lim = max(all_dat(:,4));
    min_lim = min(all_dat(:,4));  
    lim = min(abs([max_lim, min_lim])) / 2;
    all_dat(:,4) = min( max(all_dat(:,4), -1*lim), lim); % scale to lim
    % plot
    fig_erpimage2d = patch([all_dat(:,1)' nan],[all_dat(:,3)' nan],[all_dat(:,2)' nan],[all_dat(:,4)' nan],...
        'LineWidth', 2, 'FaceColor','none','EdgeColor','interp'); grid on;
%     view(3);
    ylim([-.05, .3]);
    xlim([-.2 .2]);
    set(gca,'FontSize',20)
%     axis tight
    cbar;
    set(gca,'FontSize',20)
    % save
    saveas(gcf, ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/plots/brainXtrajectory/' num2str(cluster) '.png'])
    % clear data
    clear all_dat

end

%% compute result 1: main effect velocity components & channels

% 3: effect of velocity on ERP per subject and condition, i.e. pERP, then
% average betas across subjects for each condition

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

mocap_aos = .011; % age of sample mocap samples = 11 ms
mocap_aos_samples = ceil(mocap_aos * 250);
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);
robustfit = 0;

% try out several ts prior to event
seconds_before_event = .3;
samples_before_event = seconds_before_event * 250;
ts_of_ints = (event_onset-samples_before_event):25:event_onset+50;
% ts_of_ints = ts_of_ints - mocap_aos_samples;

% select best tf_of_ints
% ts_of_ints = ts_of_ints(end);
clusters_of_int = [3, 7, 9, 24, 25, 28, 30, 33, 34];

for ts = ts_of_ints
    % components
    for cluster = clusters_of_int

        disp(['Now running analysis for cluster: ' num2str(cluster)]);
        tic

        % outpath
        save_fpath = [bemobil_config.study_folder bemobil_config.study_level 'analyses/cluster_' num2str(cluster)];
        if ~exist(save_fpath, 'dir')
            mkdir(save_fpath);
        end        

        %% get matching datasets from EEGLAB Study struct
        unique_setindices = unique(STUDY.cluster(cluster).sets);
        unique_subjects = STUDY_sets(unique_setindices);
        all_setindices = STUDY.cluster(cluster).sets;
        all_sets = STUDY_sets(all_setindices);
        all_comps = STUDY.cluster(cluster).comps;

        % load IC data
        count = 1;
        for subject = unique_subjects

            % select EEG dataset
            [~, ix] = find(subject==subjects);
            s_eeg = ALLEEG(ix);

            % loop through all vels and accs at different time points before the
            % event
            model = 'erp_sample ~ immersion * vel + trial + direction + sequence';

            %DESIGN make continuous and dummy coded predictors
            vel_vis = s_eeg.etc.analysis.mocap.visual.mag_vel(ts,:)'; % correct for age-of-sample
            vel_vibro = s_eeg.etc.analysis.mocap.vibro.mag_vel(ts,:)';

            vel = zscore([vel_vis; vel_vibro]);
            immersion = categorical([zeros(size(vel_vis,1),1); ones(size(vel_vibro,1),1)]);
            trial = zscore([s_eeg.etc.epoching.visual_tr_num, s_eeg.etc.epoching.vibro_tr_num]');
            direction = categorical([s_eeg.etc.epoching.visual_dir, s_eeg.etc.epoching.vibro_dir]');
            sequence = zscore([s_eeg.etc.epoching.visual_match_seq, s_eeg.etc.epoching.vibro_match_seq]');
            
            % old predictor_time = [1:size(vel_vis), 1:size(vel_vibro)]';

            % average ERP if more than 1 comp
            compos = all_comps(all_sets==subject);
            if compos > 1
                s_eeg.etc.analysis.erp.base_corrected.visual.comps(compos(1),:,:) = ...
                    mean(s_eeg.etc.analysis.erp.base_corrected.visual.comps(compos,:,:),1);
                s_eeg.etc.analysis.erp.base_corrected.vibro.comps(compos(1),:,:) = ...
                    mean(s_eeg.etc.analysis.erp.base_corrected.vibro.comps(compos,:,:),1);
            end

            % now fit linear model for each component
            % after averaging take one IC per subject in cluster
            disp(['running lm for subject ' num2str(subject) ' and comp ' num2str(compos(1))]);
            for sample = 1:size(s_eeg.etc.analysis.erp.base_corrected.visual.comps,2)
                erp_sample_vis = squeeze(s_eeg.etc.analysis.erp.base_corrected.visual.comps(compos(1),sample,:));
                erp_sample_vibro = squeeze(s_eeg.etc.analysis.erp.base_corrected.vibro.comps(compos(1),sample,:));
                erp_sample = [erp_sample_vis; erp_sample_vibro];

                design = table(erp_sample, immersion, vel, trial, direction, sequence);
                if robustfit
                    mdl = fitlm(design, model, 'RobustOpts', 'on');
                else
                    mdl = fitlm(design, model);
                end

                res.betas(count,sample,:) = mdl.Coefficients.Estimate;
                res.t(count,sample,:) = mdl.Coefficients.tStat;
                res.p(count,sample,:) = mdl.Coefficients.pValue;
                res.r2(count,sample,:) = mdl.Rsquared.Ordinary;
                res.adj_r2(count,sample,:) = mdl.Rsquared.Adjusted;
            end
            count = count + 1;
        end

        %TO SAVE: statistics and design info
        % add parameter names
        res.timepoint_before_touch = ts;
        res.event_onset = event_onset;
        res.model = model;
        res.parameter_names = mdl.CoefficientNames;
        this_ts = (event_onset - ts) / 250;
        save([save_fpath '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(cluster) '.mat'], 'res');
        clear res
        disp(['fitting took: ' num2str(toc/250) ' minutes']);

    end
end

%% channels

mocap_aos = .011; % age of sample mocap samples = 11 ms
mocap_aos_samples = ceil(mocap_aos * 250);
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);
robustfit = 1;

% try out several ts prior to event
seconds_before_event = .5;
samples_before_event = seconds_before_event * 250;
ts_of_ints = event_onset-samples_before_event:20:event_onset;
ts_of_ints = ts_of_ints - mocap_aos_samples;

% best tf_of_ints
ts_of_ints = ts_of_ints(end);

for ts = ts_of_ints
    for chan = channels_of_int
    
        disp(['Now running analysis for channel: ' num2str(chan)]);

        % outpath
        save_fpath = [bemobil_config.study_folder bemobil_config.study_level 'analyses/channel_' num2str(chan)];
        if ~exist(save_fpath, 'dir')
            mkdir(save_fpath);
        end        

        % load data
        count = 1;
        for subject = subjects

            % select EEG dataset
            s_eeg = ALLEEG(count);

            % loop through all vels and accs at different time points before the
            % event
            tic
            model = 'erp_sample ~ predictor_immersion * predictor_vel';

            %DESIGN make continuous and dummy coded predictors
            vel_vis = s_eeg.etc.analysis.mocap.visual.mag_vel(ts,:)';
            vel_vibro = s_eeg.etc.analysis.mocap.vibro.mag_vel(ts,:)';

            predictor_vel = [vel_vis; vel_vibro];
            predictor_immersion = [zeros(size(vel_vis,1),1); ones(size(vel_vibro,1),1)];

            % now fit linear model for each component
            disp(['running lm for subject ' num2str(subject) ' and chan ' num2str(chan)]);
            for sample = 1:size(s_eeg.etc.analysis.erp.base_corrected.visual.chans,2)
                erp_sample_vis = squeeze(s_eeg.etc.analysis.erp.base_corrected.visual.chans(chan,sample,:));
                erp_sample_vibro = squeeze(s_eeg.etc.analysis.erp.base_corrected.vibro.chans(chan,sample,:));
                erp_sample = [erp_sample_vis; erp_sample_vibro];

                design = table(erp_sample, predictor_immersion, predictor_vel);
                if robustfit
                    mdl = fitlm(design, model, 'RobustOpts', 'on');
                else
                    mdl = fitlm(design, model);
                end

                res.betas(count,sample,:) = mdl.Coefficients.Estimate;
                res.t(count,sample,:) = mdl.Coefficients.tStat;
                res.p(count,sample,:) = mdl.Coefficients.pValue;
                res.r2(count,sample,:) = mdl.Rsquared.Adjusted;
            end
            toc
            count = count + 1;
        end

        %TO SAVE: statistics and design info
        % add parameter names
        res.timepoint_before_touch = ts;
        res.event_onset = event_onset;
        res.model = model;
        res.parameter_names = mdl.CoefficientNames;
        this_ts = (event_onset - ts) / 250;
        save([save_fpath '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_channel-' num2str(chan) '.mat'], 'res');
        clear res
    end
end

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
load_p = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/cluster_';
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

load_p = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/cluster_';

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