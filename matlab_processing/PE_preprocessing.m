%% clear all and load params
close all; clear

PE_config;

%% processing loop

if ~exist('ALLEEG','var')
	eeglab;
	runmobilab;
end

for subject = subjects
    
    % change fnames for s3
    if subject == 3
        ori_fnames = bemobil_config.filenames;
        bemobil_config.filenames = bemobil_config.filenames_s3;
    end
	
	% load xdf files and process them with mobilab, export to eeglab, split MoBI and merge all conditions for EEG
	[ALLEEG, EEG_merged, CURRENTSET] = bemobil_process_all_mobilab(subject, bemobil_config, ALLEEG, CURRENTSET, mobilab, 1);
    
	% finally start the complete processing pipeline including AMICA
	[ALLEEG, EEG_AMICA_final, CURRENTSET] = bemobil_process_all_AMICA(ALLEEG, EEG_merged, CURRENTSET, subject, bemobil_config);
    
    % create a merged mocap files
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    input_filepath = [bemobil_config_folder bemobil_config.raw_EEGLAB_data_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
    for filename = bemobil_config.filenames
        EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
            filename{1} '_MOCAP.set' ], 'filepath', input_filepath);
        EEG = pop_resample(EEG, bemobil_config.resample_freq);
        [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'study',0);
    end
    % and merge, saves merged file to single subject analysis folder
    [ALLEEG, mocap, CURRENTSET] = bemobil_merge(ALLEEG,EEG,CURRENTSET,1:length(ALLEEG),...
        [bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.merged_filename_mocap], output_filepath);
    
    % change back fnames after processing s3
    if subject == 3
        bemobil_config.filenames = ori_fnames;
    end
	
end

%% event and baseline epoching, clean epoch indices (autorej function), mocap data

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

for subject = subjects
    
	disp(['Subject #' num2str(subject)]);
    
    % filepaths
	input_filepath = [bemobil_config_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	
    %EEG: load data
	EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.copy_weights_interpolate_avRef_filename], 'filepath', input_filepath);
	[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'study',0);
    %MOCAP: load data
    mocap = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.merged_filename_mocap], 'filepath', input_filepath);
	[ALLEEG, mocap, CURRENTSET] = pop_newset(ALLEEG, mocap, 0,'study',0);
    
    %EEG: filter for ERP analysis
    EEG = pop_eegfiltnew(EEG, bemobil_config.filter_plot_low, bemobil_config.filter_plot_high);
	
    %BOTH
    % epoching
    % parse events from urevent structure 
    EEG = parse_events_PE(EEG);
    mocap = parse_events_PE(mocap);
    touch_ixs = find(strcmp({EEG.event.type}, 'box:touched'));
    % check data consistency
    if size(touch_ixs,2) ~= 600
        warning('Not exactly 600 events in dataset!')
    end
    if ~(size(EEG.times,2)==size(mocap.times,2))
        error('mocap and eeg of different length of data!');
    end
    if ~(size(EEG.event,2)==size(mocap.event,2))
        error('mocap and eeg have differing number of events!');
    end
    
    %TO SAVE: find indeces of mismatch trials in both visual and visual+vibro condition
    box_touched_events = EEG.event(touch_ixs);
    EEG.etc.epoching.mismatch = find(strcmp({box_touched_events.normal_or_conflict}, 'conflict'));
            
    %BOTH: epoch around box:touched event
    % duplicate EEG set for epoching twice (on events and baseline)
    oriEEG = EEG;
    mocap.event = EEG.event;
    EEG = pop_epoch( EEG, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    mocap = pop_epoch( mocap, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
        
    %EEG: find ~10% noisiest epoch indices by searching for large amplitude
    % fluctuations on the channel level using eeglab auto_rej function
    % search only mismatch epochs
    EEG = pop_select(EEG, 'trial', EEG.etc.epoching.mismatch);
    [~, EEG.etc.epoching.rmepochs] = pop_autorej(EEG, 'maxrej', [1],'nogui','on','eegplot','off');
    % ignore rejected epochs
    ep_conds = cellfun(@(v)v(1), {EEG.epoch.eventcondition}); % extract epoch condition
    %TO SAVE: find indeces of visual and vibro trials after epoch rejection
    EEG.etc.epoching.visual = find(strcmp(ep_conds, 'visual'));
    EEG.etc.epoching.visual(find(ismember(EEG.etc.epoching.visual, EEG.etc.epoching.rmepochs))) = [];
    EEG.etc.epoching.vibro = find(strcmp(ep_conds, 'vibro'));
    EEG.etc.epoching.vibro(find(ismember(EEG.etc.epoching.vibro, EEG.etc.epoching.rmepochs))) = [];
            
    %EEG: epochs around box:spawned event, [-1 0], use isiTime (time from starting trial to box:spawned) 600 to 900 as
    % baseline interval
    EEG_base = pop_epoch( oriEEG, bemobil_config.epoching.base_epoching_event, bemobil_config.epoching.base_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    if size(EEG_base.data,3) ~= 600
        warning('Not exactly 600 events in dataset')
    end
    % manually handle missing data
    switch subject
        case 14
            % the baseline for the first trial in visual condition is missing
            % copy baseline win from subsequent trial
            EEG_base.data(:,:,301:600) = EEG_base.data(:,:,end-299:end);
            EEG_base.data(:,:,300) = EEG_base.data(:,:,301); % copy baseline for one trial from the succeeding trial            
            EEG_base.icaact(:,:,301:600) = EEG_base.icaact(:,:,end-299:end);
            EEG_base.icaact(:,:,300) = EEG_base.icaact(:,:,301); % copy baseline for one trial from the succeeding trial            
    end
    
    %EEG: Baseline corrected ERPs of mismatch trials
    EEG_base.data = EEG_base.data(:,:,EEG.etc.epoching.mismatch);
    EEG_base.icaact = EEG_base.icaact(:,:,EEG.etc.epoching.mismatch);
    % get baseline windows in epochs
    base_start_ix = bemobil_config.epoching.base_win(1) * EEG_base.srate;
    base_end_ix = bemobil_config.epoching.base_win(2) * EEG_base.srate;
    % baseline chans
    EEG.etc.epoching.baseline_chans_visual = mean(EEG_base.data(:, base_start_ix:base_end_ix, EEG.etc.epoching.visual), 2);
    EEG.etc.epoching.baseline_chans_vibro = mean(EEG_base.data(:, base_start_ix:base_end_ix, EEG.etc.epoching.vibro), 2);
    % baseline comps
    EEG.etc.epoching.baseline_comps_visual = mean(EEG_base.icaact(:, base_start_ix:base_end_ix, EEG.etc.epoching.visual), 2);
    EEG.etc.epoching.baseline_comps_vibro = mean(EEG_base.icaact(:, base_start_ix:base_end_ix, EEG.etc.epoching.vibro), 2);
    % baseline corrected ERPs
    EEG.etc.analysis.erp.base_corrected_erps_chans_visual(:,:,:) = EEG.data(:,:,EEG.etc.epoching.visual) - EEG.etc.epoching.baseline_chans_visual;
    EEG.etc.analysis.erp.base_corrected_erps_chans_vibro(:,:,:) = EEG.data(:,:,EEG.etc.epoching.vibro) - EEG.etc.epoching.baseline_chans_vibro;
    EEG.etc.analysis.erp.base_corrected_erps_comps_visual(:,:,:) = EEG.icaact(:,:,EEG.etc.epoching.visual) - EEG.etc.epoching.baseline_comps_visual;
    EEG.etc.analysis.erp.base_corrected_erps_comps_vibro(:,:,:) = EEG.icaact(:,:,EEG.etc.epoching.vibro) - EEG.etc.epoching.baseline_comps_vibro;
    
    %MOCAP: Velocity at time points before events of interest
    % select clean epochs within all mismatch epochs
    mocap = pop_select(mocap, 'trial', EEG.etc.epoching.mismatch);
    mocap_vis = pop_select(mocap, 'trial', EEG.etc.epoching.visual);
    mocap_vibro = pop_select(mocap, 'trial', EEG.etc.epoching.vibro);
    % calculate magnitude of velocity in 3D at different time points
    for t = 1:size(bemobil_config.epoching.vel_ts,2)
        EEG.etc.analysis.mocap.visual.win(t).mag_vel = squeeze(sqrt(mocap_vis.data(7,bemobil_config.epoching.vel_ts(t),:).^2 +...
            mocap_vis.data(8,bemobil_config.epoching.vel_ts(t),:).^2 +...
            mocap_vis.data(9,bemobil_config.epoching.vel_ts(t),:).^2));
        EEG.etc.analysis.mocap.visual.win(t).mag_acc = squeeze(sqrt(mocap_vis.data(13,bemobil_config.epoching.vel_ts(t),:).^2 +...
            mocap_vis.data(14,bemobil_config.epoching.vel_ts(t),:).^2 +...
            mocap_vis.data(15,bemobil_config.epoching.vel_ts(t),:).^2));
        
        EEG.etc.analysis.mocap.vibro.win(t).mag_vel = squeeze(sqrt(mocap_vibro.data(7,bemobil_config.epoching.vel_ts(t),:).^2 +...
            mocap_vibro.data(8,bemobil_config.epoching.vel_ts(t),:).^2 +...
            mocap_vibro.data(9,bemobil_config.epoching.vel_ts(t),:).^2));
        EEG.etc.analysis.mocap.vibro.win(t).mag_acc = squeeze(sqrt(mocap_vibro.data(13,bemobil_config.epoching.vel_ts(t),:).^2 +...
            mocap_vibro.data(14,bemobil_config.epoching.vel_ts(t),:).^2 +...
            mocap_vibro.data(15,bemobil_config.epoching.vel_ts(t),:).^2));
    end

    % save epoched datasets with clean trial indices of the 2x2 study design
    pop_saveset(EEG, 'filename', [bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.epochs_filename] , 'filepath', output_filepath)
end

%% 1st Level summary statistic ERP (all channels and comps): betas from fit lm at each point of the ERP post event
% -> effect of hand velocity when box is touched on subsequent feedback processing

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

for subject = subjects
	
	disp(['Subject #' num2str(subject)]);
	
	input_filepath = [bemobil_config_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	
	EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.epochs_filename], 'filepath', input_filepath);

    % loop through all vels and accs at different time points before the
    % event
    model = 'erp_sample ~ predictor_immersion * predictor_vel';
    for w = 1:size(EEG.etc.analysis.mocap.visual.win,2)
        %DESIGN make continuous and dummy coded predictors
        vel_vis = EEG.etc.analysis.mocap.visual.win(w).mag_vel;
        vel_vibro = EEG.etc.analysis.mocap.vibro.win(w).mag_vel;
        predictor_vel = [vel_vis; vel_vibro];
        predictor_immersion = [zeros(size(vel_vis,1),1); ones(size(vel_vibro,1),1)];

        % now fit linear model for each condition and both channels and
        % components
        for chan = 1:size(EEG.etc.analysis.erp.base_corrected_erps_chans_visual,1)
    %         h = waitbar(0, ['Now fitting LM for channel: ' EEG.chanlocs(chan).labels]);
    %         waitbar(chan/size(EEG.data,1),h)

            for sample = 1:size(EEG.etc.analysis.erp.base_corrected_erps_chans_visual,2)

                erp_sample_vis = squeeze(EEG.etc.analysis.erp.base_corrected_erps_chans_visual(chan,sample,:));
                erp_sample_vibro = squeeze(EEG.etc.analysis.erp.base_corrected_erps_chans_vibro(chan,sample,:));
                erp_sample = [erp_sample_vis; erp_sample_vibro];

                design = table(erp_sample, predictor_immersion, predictor_vel);

                mdl = fitlm(design, model);
                EEG.etc.analysis.statistics.chans.win(w).betas(chan,sample,:) = mdl.Coefficients.Estimate;
            end

    %         close(h)
        end
        for comp = 1:size(EEG.etc.analysis.erp.base_corrected_erps_comps_visual,1)
    %         h = waitbar(0, ['Now fitting LM for component: ' num2str(comp)]);
    %         waitbar(comp/size(EEG.icaact,1),h)

            for sample = 1:size(EEG.etc.analysis.erp.base_corrected_erps_comps_visual,2)

                erp_sample_vis = squeeze(EEG.etc.analysis.erp.base_corrected_erps_comps_visual(comp,sample,:));
                erp_sample_vibro = squeeze(EEG.etc.analysis.erp.base_corrected_erps_comps_vibro(comp,sample,:));
                erp_sample = [erp_sample_vis; erp_sample_vibro];

                design = table(erp_sample, predictor_immersion, predictor_vel);

                mdl = fitlm(design, model);
                EEG.etc.analysis.statistics.comps.win(w).betas(comp,sample,:) = mdl.Coefficients.Estimate;
            end

    %         close(h)
        end
    end
    
    %TO SAVE: statistics and design info
    % add parameter names
    EEG.etc.analysis.statistics.timepoints_before_event_in_s = (bemobil_config.epoching.vel_ts - EEG.srate) / EEG.srate;
    EEG.etc.analysis.statistics.model = model;
    EEG.etc.analysis.statistics.parameter_names = mdl.CoefficientNames;
    
    % save
    pop_saveset(EEG, 'filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
            bemobil_config.epochs_filename], 'filepath', output_filepath);
end

%% build eeglab study and cluster

% cluster using only dipole location
% inspect and select cluster #s to analyze
input_path = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder];
output_path = [bemobil_config.study_folder bemobil_config.study_level];

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 1, 'option_savetwofiles', 1, 'option_saveversion6', 0,...
    'option_single', 1, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1,...
    'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0,...
    'option_checkversion', 1, 'option_chat', 1);
STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];

for subject = subjects
    disp(['Subject #' num2str(subject) ]);
    input_filepath = [input_path bemobil_config.filename_prefix num2str(subject)];
    EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.epochs_filename], 'filepath', input_filepath);
    
    comps_ix{subject} = 1:size(EEG.icaact,1);
    
    EEG = eeg_checkset( EEG );
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
end
eeglab redraw
disp('All study files loaded. Creating STUDY...')
command = cell(0,0);
for set = 1:length(subjects)
    command{end+1} = {'index' set 'subject' num2str(subjects(set)) };
end

% create study
[STUDY ALLEEG] = std_editset( STUDY, ALLEEG, 'name', bemobil_config.study_filename,'commands',command,...
    'updatedat','on','savedat','off','rmclust','on' );
[STUDY ALLEEG] = std_checkset(STUDY, ALLEEG);
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];

% precompute component measures except ERSPs
disp('Precomputing topographies and spectra.')
[STUDY ALLEEG] = std_precomp(STUDY, ALLEEG, 'components','recompute','on','scalp','on','erp', 'on', 'spec','on','specparams',...
    {'specmode' 'fft' 'logtrials' 'off'});

% create preclustering array that is used for clustering and save study
[STUDY ALLEEG] = std_preclust(STUDY, ALLEEG, [],...
                         { 'spec'  'npca' 10 'norm' 1 'weight' 1 'freqrange'  [ 3 25 ] } , ...
                         { 'erp'   'npca' 10 'norm' 1 'weight' 2 'timewindow' [ -300 1000 ] } ,...
                         { 'scalp' 'npca' 10 'norm' 1 'weight' 2 'abso' 1 } , ...
                         { 'dipoles'         'norm' 1 'weight' 15 } , ...
                         { 'finaldim' 'npca' 10 });
STUDY.bemobil.clustering.preclustparams = STUDY.cluster.preclust.preclustparams;
                     
% check how many ICs remain and do clustering
c = 1;
for subject = 1:size(STUDY.datasetinfo,2)
    n_clust(c) = size(ALLEEG(subject).icaact,1);
    c = c+1;
end
bemobil_config.STUDY_n_clust = round(bemobil_config.IC_percentage * mean(n_clust));

path_clustering_solutions = [bemobil_config.STUDY_filepath_clustering_solutions bemobil_config.study_filename(1:end-6) '\' ...
    [num2str(bemobil_config.STUDY_cluster_ROI_talairach.x) '_' num2str(bemobil_config.STUDY_cluster_ROI_talairach.y) '_' num2str(bemobil_config.STUDY_cluster_ROI_talairach.z)] '-location_'...
    num2str(bemobil_config.STUDY_n_clust) '-cluster_' num2str(bemobil_config.outlier_sigma) ...
    '-sigma_' num2str(bemobil_config.STUDY_clustering_weights.dipoles) '-dipoles_' num2str(bemobil_config.STUDY_clustering_weights.spectra) '-spec_'...
    num2str(bemobil_config.STUDY_clustering_weights.scalp_topographies) '-scalp_' num2str(bemobil_config.n_iterations) '-iterations'];

% cluster the components repeatedly and use a region of interest and
% quality measures to find the best fitting solution
[STUDY, ALLEEG, EEG] = bemobil_repeated_clustering_and_evaluation(STUDY, ALLEEG, EEG, bemobil_config.outlier_sigma,...
    bemobil_config.STUDY_n_clust, bemobil_config.n_iterations, bemobil_config.STUDY_cluster_ROI_talairach,...
    bemobil_config.STUDY_quality_measure_weights, 1,...
    1, output_path, bemobil_config.study_filename, [output_path path_clustering_solutions],...
    bemobil_config.filename_clustering_solutions, [output_path path_clustering_solutions], bemobil_config.filename_multivariate_data);

% save study
disp('Saving STUDY...')
mkdir(output_path)
[STUDY EEG] = pop_savestudy( STUDY, EEG, 'filename', bemobil_config.study_filename,'filepath',output_path);
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
eeglab redraw
disp('...done')

%% per cluster: 2nd Level inference

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
clusters_of_int = [24, 14, 9, 21, 22, 28, 35];

%%

% 24 ACC
% 14 parietal

for cluster = 14%[24, 14] %, 22, 35]%clusters_of_int
    
    disp(['Now running analysis for cluster: ' num2str(cluster)]);
    
%     % outpath
%     save_fpath = [cfg.save_fpath '\cluster_' num2str(cluster)];
%     if ~exist(save_fpath, 'dir')
%         mkdir(save_fpath);
%     end        
    
    %% get matching datasets from EEGLAB Study struct
    unique_setindices = unique(STUDY.cluster(cluster).sets);
    unique_subjects = STUDY_sets(unique_setindices);
    all_setindices = STUDY.cluster(cluster).sets;
    all_sets = STUDY_sets(all_setindices);
    all_comps = STUDY.cluster(cluster).comps;
    
    % load IC data
    for subject = unique_subjects
        % get IC(s) per subject in cluster
        IC = all_comps(all_sets==subject);
        
        % select EEG dataset
        [~, ix] = find(subject==subjects);
        this_subject_eeg = ALLEEG(ix);
        
        for win = 1:size(this_subject_eeg.etc.analysis.statistics.comps.win,2)
            res.betas.ic(win,ix,:,:) = mean(this_subject_eeg.etc.analysis.statistics.comps.win(win).betas(IC,:,:),1);
%             res.betas.chan(win,ix,:,:) = this_subject_eeg.etc.analysis.statistics.chans.win(win).betas(25,:,:); % PZ
%             res.betas.chan(win,ix,:,:) = this_subject_eeg.etc.analysis.statistics.chans.win(win).betas(65,:,:); % FCz
            res.betas.chan(win,ix,:,:) = this_subject_eeg.etc.analysis.statistics.chans.win(win).betas(5,:,:); % Fz
        end
        res.parameters = this_subject_eeg.etc.analysis.statistics.parameter_names;
    end
    
    % make average betas/pvals
    for ef = 4%1:4
        figure;
        hold on
        for i = 6 %1:6
%             plot(squeeze(mean(res.betas.ic(i,:,75:end,ef),2)));
            plot(squeeze(mean(res.betas.chan(i,:,75:end,ef),2)));
        end
        % legend
        timepoints = round((bemobil_config.epoching.vel_ts / this_subject_eeg.srate) - 1, 1);
        legend(string(timepoints));
        title([res.parameters{ef} ' ' num2str(cluster)]);
%         ylim([-1 1]*5);
        xticklabels(xticks/this_subject_eeg.srate);
    end
    
    clear res

end

%% mcc and plot average effect on group level

% - paired t-test effect of hand velocity between visual vs visual-vibro
% conditions (is the effect of velocity on the error processing ERP moderated by haptic immersion)






% optional

%% plot brain ICs

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

for subject = subjects
	
	disp(['Subject #' num2str(subject)]);
	
	input_filepath = [bemobil_config_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	
	EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.copy_weights_interpolate_avRef_filename], 'filepath', input_filepath);
	[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'study',0);
	
	% clean now, save files and figs
	[ALLEEG, EEG_cleaned, CURRENTSET, ICs_keep, ICs_throw] = bemobil_clean_with_iclabel( EEG , ALLEEG, CURRENTSET, [1], bemobil_config.brain_threshold,...
		[ bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.single_subject_cleaned_ICA_filename],output_filepath);
	
end

% plot dipoles
pop_dipplot( EEG, ICs_keep,...
	'mri','P:\\Marius\\toolboxes\\eeglab14_1_0b\\plugins\\dipfit2.3\\standard_BEM\\standard_mri.mat','normlen','on');

% save fig
savefig(fullfile(output_filepath,'brain_dipoles'))
