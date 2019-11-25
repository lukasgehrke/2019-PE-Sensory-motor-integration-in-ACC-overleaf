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
    input_filepath = [bemobil_config.study_folder bemobil_config.raw_EEGLAB_data_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
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
    
    %% load data and filter EEG
	disp(['Subject #' num2str(subject)]);
    
    % filepaths
	input_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	
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
	
    %% BOTH: obtaining clean epoch indices
    % get latencies of boundaries
    boundaries_lats = EEG.urevent(find(strcmp({EEG.urevent.type}, 'boundary'))).latency;
    
    % parse events from urevent structure 
    EEG = parse_events_PE(EEG);
    mocap = parse_events_PE(mocap);
    touch_ixs = find(strcmp({EEG.event.type}, 'box:touched'));
    spawn_ixs = find(strcmp({EEG.event.type}, 'box:spawned'));
    
    % get event indices
    mismatch_ixs = find(strcmp({EEG.event.normal_or_conflict}, 'conflict'));
    mis_touch_ixs = intersect(touch_ixs, mismatch_ixs);
    mis_spawn_ixs = intersect(spawn_ixs, mismatch_ixs);
    
    % check data consistency
    if ~isequal(size(mis_touch_ixs), size(mis_spawn_ixs))
        error('Number of touch and spawn events is not the same!')
    end
    if size(mis_touch_ixs,2) ~= 150
        warning('Not exactly 600 events in dataset!')
    end
    if ~(size(EEG.times,2)==size(mocap.times,2))
        error('mocap and eeg of different length of data!');
    end
    if ~(size(EEG.event,2)==size(mocap.event,2))
        error('mocap and eeg have differing number of events!');
    end
    
    % 1. remove all trials that have a distance greater than 2s between 
    % spawn and touch
    latency_diff = (cell2mat({EEG.event(touch_ixs).latency}) - cell2mat({EEG.event(spawn_ixs).latency})) / EEG.srate;
    slow_rt_ix = find(latency_diff>2);
    rm_ixs = slow_rt_ix;
    
    % 2. remove all trials with less than 300 ms of data before
    % box:spawned, this will be used for baseline!
    % check first trial and at boundaries
    if EEG.event(spawn_ixs(1)).latency < abs(bemobil_config.epoching.base_epochs_boundaries(1) * 250)
        % remove first mismatch trial because not enough data before the touch event
        rm_ixs = [rm_ixs, 1];
    end
    for b = boundaries_lats
        dist = cell2mat({EEG.event(mis_spawn_ixs).latency}) - b;
        post_b_ix = find(dist>0, 1, 'first'); % index of post boundary spawn event
        ev_dist = EEG.event(mis_spawn_ixs(post_b_ix)).latency - b;
        
        if ev_dist < abs(bemobil_config.epoching.base_epochs_boundaries(1) * EEG.srate)
            % no data for baseline, remove post boundary event
            rm_ixs = [rm_ixs, post_b_ix];
        end
    end
    
    %TO SAVE: find indeces of mismatch trials in both visual and visual+vibro condition
    EEG.etc.epoching.mismatch_touch_ixs = mis_touch_ixs;
    EEG.etc.epoching.mismatch_spawn_ixs = mis_spawn_ixs;
    EEG.etc.epoching.rm_ixs = rm_ixs;
    EEG.etc.epoching.latency_diff = cell2mat({EEG.event(mis_touch_ixs).latency}) - cell2mat({EEG.event(mis_spawn_ixs).latency});
            
    %% BOTH: epoching
    % duplicate EEG set for epoching twice (on events and baseline)
    baseEEG = EEG;
    
    % overwrite events with clean epoch events
    EEG.event = EEG.event(EEG.etc.epoching.mismatch_touch_ixs); % touches
    mocap.event = EEG.event;
    baseEEG.event = baseEEG.event(EEG.etc.epoching.mismatch_spawn_ixs); % spawns
    
    % ERPs: both EEG and mocap
    EEG = pop_epoch( EEG, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    mocap = pop_epoch( mocap, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
        
    %EEG: find ~10% noisiest epoch indices by searching for large amplitude
    % fluctuations on the channel level using eeglab auto_rej function
    [~, rmepochs] = pop_autorej(EEG, 'maxrej', [1],'nogui','on','eegplot','off');
    EEG.etc.epoching.rm_ixs = [EEG.etc.epoching.rm_ixs, rmepochs];
    
    %TO SAVE: find indeces of visual and vibro trials after epoch rejection
    EEG.etc.epoching.visual = find(strcmp({EEG.event.condition}, 'visual'));
    EEG.etc.epoching.vibro = find(strcmp({EEG.event.condition}, 'vibro'));
    EEG.etc.epoching.good_visual = setdiff(EEG.etc.epoching.visual, EEG.etc.epoching.rm_ixs);
    EEG.etc.epoching.good_vibro = setdiff(EEG.etc.epoching.vibro, EEG.etc.epoching.rm_ixs);
    
    %EEG: epochs around box:spawned event for baseline
    EEG_base = pop_epoch( baseEEG, bemobil_config.epoching.base_epoching_event, bemobil_config.epoching.base_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    
%     % manually handle missing data
%     switch subject
%         case 14
%             % the baseline for the first trial in visual condition is missing
%             % copy baseline win from subsequent trial
%             EEG_base.data(:,:,301:600) = EEG_base.data(:,:,end-299:end);
%             EEG_base.data(:,:,300) = EEG_base.data(:,:,301); % copy baseline for one trial from the succeeding trial            
%             EEG_base.icaact(:,:,301:600) = EEG_base.icaact(:,:,end-299:end);
%             EEG_base.icaact(:,:,300) = EEG_base.icaact(:,:,301); % copy baseline for one trial from the succeeding trial            
%     end
    
    %% EEG: Baseline corrected ERPs of mismatch trials
    
    % select clean epochs within all mismatch epochs
    vis = EEG.etc.epoching.good_visual;
    vib = EEG.etc.epoching.good_vibro;
    
    % baseline chans
    EEG.etc.analysis.erp.baseline.visual.chans = mean(EEG_base.data(:, :, vis), 2);
    EEG.etc.analysis.erp.baseline.vibro.chans = mean(EEG_base.data(:, :, vib), 2);
    
    % baseline comps
    EEG.etc.analysis.erp.baseline.visual.comps = mean(EEG_base.icaact(:, :, vis), 2);
    EEG.etc.analysis.erp.baseline.vibro.comps = mean(EEG_base.icaact(:, :, vib), 2);
    
    % baseline corrected ERPs
    EEG.etc.analysis.erp.base_corrected.visual.chans(:,:,:) = EEG.data(:,:,vis) - EEG.etc.analysis.erp.baseline.visual.chans;
    EEG.etc.analysis.erp.base_corrected.vibro.chans(:,:,:) = EEG.data(:,:,vib) - EEG.etc.analysis.erp.baseline.vibro.chans;
    EEG.etc.analysis.erp.base_corrected.visual.comps(:,:,:) = EEG.icaact(:,:,vis) - EEG.etc.analysis.erp.baseline.visual.comps;
    EEG.etc.analysis.erp.base_corrected.vibro.comps(:,:,:) = EEG.icaact(:,:,vib) - EEG.etc.analysis.erp.baseline.vibro.comps;
    
    %% MOCAP: Velocity at time points before events of interest
    
    % save (x,y) coordinates for each trial
    hand_chans_ix = 1:3;
    EEG.etc.analysis.mocap.visual.x = squeeze(mocap.data(hand_chans_ix(1),:,vis));
    EEG.etc.analysis.mocap.visual.y = squeeze(mocap.data(hand_chans_ix(2),:,vis));
    EEG.etc.analysis.mocap.visual.z = squeeze(mocap.data(hand_chans_ix(3),:,vis));
    EEG.etc.analysis.mocap.vibro.x = squeeze(mocap.data(hand_chans_ix(1),:,vib));
    EEG.etc.analysis.mocap.vibro.y = squeeze(mocap.data(hand_chans_ix(2),:,vib));
    EEG.etc.analysis.mocap.vibro.z = squeeze(mocap.data(hand_chans_ix(3),:,vib));
    
    % save 3D magnitude of velocity and acceleration
    EEG.etc.analysis.mocap.visual.mag_vel = squeeze(sqrt(mocap.data(7,:,vis).^2 +...
            mocap.data(8,:,vis).^2 +...
            mocap.data(9,:,vis).^2));
    EEG.etc.analysis.mocap.visual.mag_acc = squeeze(sqrt(mocap.data(13,:,vis).^2 +...
            mocap.data(14,:,vis).^2 +...
            mocap.data(15,:,vis).^2));
    EEG.etc.analysis.mocap.vibro.mag_vel = squeeze(sqrt(mocap.data(7,:,vib).^2 +...
            mocap.data(8,:,vib).^2 +...
            mocap.data(9,:,vib).^2));
    EEG.etc.analysis.mocap.vibro.mag_acc = squeeze(sqrt(mocap.data(13,:,vib).^2 +...
            mocap.data(14,:,vib).^2 +...
            mocap.data(15,:,vib).^2));

    %% save epoched datasets with clean trial indices of the 2x2 study design
    pop_saveset(EEG, 'filename', [bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.epochs_filename] , 'filepath', output_filepath);
    
    % also save results struct only, smaller and hence faster to load
    res.epoching = EEG.etc.epoching;
    res.erp = EEG.etc.analysis.erp;
    res.mocap = EEG.etc.analysis.mocap;
    
    save([output_filepath '_res'], 'res');
    
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

path_clustering_solutions = [bemobil_config.STUDY_filepath_clustering_solutions bemobil_config.study_filename(1:end-6) '/' ...
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
