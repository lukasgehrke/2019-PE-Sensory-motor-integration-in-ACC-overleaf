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
clusters_of_int = [3, 24, 25, 28, 33];
% clusters
% 3: right parietal
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

%% result 0.1: plot 3D trajectory colorcoded by hand velocity and erp amplitude (erpimage2D)

% 1: make hand velocity at each x,y for each subject, then plot average
% across subjects and smooth

% 2: make erp amplitude (erpimage2D) at each (x,y) for each subject, then
% plot average across subjects and smooth it -> this is important and cool

cond = 'vibro';
trials = 'good_vibro';
chan = 25;
comp = 5;
buffertime = 0;

c = 1;
all_dat = NaN(1,4);

for s = ALLEEG
    
    rt = s.etc.epoching.latency_diff(s.etc.epoching.(trials));
    buffer = buffertime * s.srate;
    event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * s.srate);

    event_start = event_onset - rt;
    event_end = event_start + rt + buffer;
    event_wins = floor([event_start; event_end]');

    x = s.etc.analysis.mocap.(cond).x;
    y = s.etc.analysis.mocap.(cond).y;
    z = s.etc.analysis.mocap.(cond).z;
%     measure = s.etc.analysis.mocap.(cond).mag_vel;
    measure = squeeze(s.etc.analysis.erp.base_corrected.(cond).chans(chan,:,:));
%     measure = squeeze(s.etc.analysis.erp.base_corrected.(cond).comps(comp,:,:));

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
    
    all_dat = vertcat(all_dat, data_long);
    c = c+1;
end

% plot of param on lines
figure;
fig_erpimage2d = patch([all_dat(:,1)' nan],[all_dat(:,2)' nan],[all_dat(:,3)' nan],[all_dat(:,4)' nan],...
    'LineWidth', 2, 'FaceColor','none','EdgeColor','interp'); grid on;
view(3);
cbar;

% % plot mean
% test = squeeze(mean(data,2));
% figure;
% plot(test(:,1),test(:,2))

% % make heat map
% edges.x = [.25, .8];
% edges.y = [.25, .8];
% x = test(:,1)';
% y = test(:,3)';
% c = test(:,4)';
% resolution_stepsize = .01;
% gauss_kernel_size = 1.2;
% [ image ] = make_map(edges, x, y, c, resolution_stepsize, gauss_kernel_size);
% 
% % plot
% lims = (max(abs(measure(:)))/10) * [-1 1];
% fig_heat_erp = figure;imagesc(image.image_c, lims);
% cbarax = cbar;
% cbarax.YLim = [0 cbarax.YLim(2)];

%% result 1: main effect velocity components & channels

% 3: effect of velocity on ERP per subject and condition, i.e. pERP, then
% average betas across subjects for each condition

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

vel_ts = 11; % age of sample mocap samples = 11 ms
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);
ts_of_int = event_onset - vel_ts;
robustfit = 1;

for cluster = clusters_of_int
    
    disp(['Now running analysis for cluster: ' num2str(cluster)]);
    
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
        tic
        model = 'erp_sample ~ predictor_immersion * predictor_vel';
        
        %DESIGN make continuous and dummy coded predictors
        vel_vis = s_eeg.etc.analysis.mocap.visual.mag_vel(ts_of_int-vel_ts,:)'; % correct for age-of-sample
        vel_vibro = s_eeg.etc.analysis.mocap.vibro.mag_vel(ts_of_int-vel_ts,:)';
        
        predictor_vel = [vel_vis; vel_vibro];
        predictor_immersion = [zeros(size(vel_vis,1),1); ones(size(vel_vibro,1),1)];

        % average ERP if more than 1 comp
        comps = all_comps(all_sets==subject);
        if comps > 1
            s_eeg.etc.analysis.erp.base_corrected.visual.comps(comps(1),:,:) = ...
                mean(s_eeg.etc.analysis.erp.base_corrected.visual.comps(comps,:,:),1);
            s_eeg.etc.analysis.erp.base_corrected.vibro.comps(comps(1),:,:) = ...
                mean(s_eeg.etc.analysis.erp.base_corrected.vibro.comps(comps,:,:),1);
        end
        
        % now fit linear model for each component
        % after averaging take one IC per subject in cluster
        disp(['running lm for subject ' num2str(subject) ' and comp ' num2str(comps(1))]);
        for sample = 1:size(s_eeg.etc.analysis.erp.base_corrected.visual.comps,2)
            erp_sample_vis = squeeze(s_eeg.etc.analysis.erp.base_corrected.visual.comps(comps(1),sample,:));
            erp_sample_vibro = squeeze(s_eeg.etc.analysis.erp.base_corrected.vibro.comps(comps(1),sample,:));
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
    res.timepoint_before_touch = ts_of_int;
    res.event_onset = event_onset;
    res.model = model;
    res.parameter_names = mdl.CoefficientNames;
    save([save_fpath '/res_' model '_robust_' num2str(robustfit) '_comp_' num2str(comps(1))], 'res');
    
end

for chan = channels_of_int
    
    disp(['Now running analysis for channel: ' num2str(chan)]);
    
    % outpath
    save_fpath = [bemobil_config.study_folder bemobil_config.study_level 'analyses/channel_' num2str(chan)];
    if ~exist(save_fpath, 'dir')
        mkdir(save_fpath);
    end        
    
    % load IC data
    count = 1;
    for subject = subjects
        
        % select EEG dataset
        s_eeg = ALLEEG(count);
        
        % loop through all vels and accs at different time points before the
        % event
        tic
        model = 'erp_sample ~ predictor_immersion * predictor_vel';
        
        %DESIGN make continuous and dummy coded predictors
        vel_vis = s_eeg.etc.analysis.mocap.visual.mag_vel(ts_of_int-vel_ts,:)';
        vel_vibro = s_eeg.etc.analysis.mocap.vibro.mag_vel(ts_of_int-vel_ts,:)';
        
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
    res.timepoint_before_touch = ts_of_int;
    res.event_onset = event_onset;
    res.model = model;
    res.parameter_names = mdl.CoefficientNames;
    save([save_fpath '/res_' model '_robust_' num2str(robustfit) '_chan_' num2str(chan)], 'res');
    
end

%% result 2: interaction effect of haptic immersion and velocity

% t-test regression estimate agains zero using LIMO robust statistics
% plot 3 & 4 on the same ERP plot with added CIs and marking significant 
% samples

%% plotting results

% shift due to EEG age of sample
eeg_age_of_sample = .066; % ms age of sample eeg data
eeg_age_of_sample_samples = floor(eeg_age_of_sample * 250);
