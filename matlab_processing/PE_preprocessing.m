%% clear all and load params (DONE)
close all; clear all;

PE_config;

%% processing loop (DONE)

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

%% save all topos for all subjects (DONE)

% to fix elocs of one subject
for s = ALLEEG
    pop_topoplot(s,0,[1:size(s.icasphere,1)]);
    saveas(gcf, ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/topomaps/' s.filename(1:3) '.png']);
    close(gcf);
    close all;
end

%% event and baseline epoching, clean epoch indices (autorej,3 function), mocap data (DONE)

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

for subject = subjects
    
    %% load EEG and mocap data
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
	
    %% obtaining clean events
    
    % parse events from urevent structure 
    EEG = parse_events_PE(EEG);
    mocap = parse_events_PE(mocap);
        
    % change field name of 'condition' so its not a study thing
    oldField = 'condition';
    newField = 'feedback';
    [EEG.event.(newField)] = EEG.event.(oldField);
    EEG.event = rmfield(EEG.event,oldField);
    
    % get event indices
    touch_ixs = find(strcmp({EEG.event.type}, 'box:touched'));
    spawn_ixs = find(strcmp({EEG.event.type}, 'box:spawned'));
        
    %% Design Matrix
    
    design = EEG;
    design.event = design.event(touch_ixs);
    design = pop_epoch( design, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    
    % remove all trials that have a distance greater than 2s between 
    % spawn and touch: participants where slow to react / start the trial
    latency_diff = (cell2mat({EEG.event(touch_ixs).latency}) - cell2mat({EEG.event(spawn_ixs).latency})) / EEG.srate;
    bad_tr_ixs = find(latency_diff>2);
    EEG.etc.analysis.design.rt_spawned_touched = latency_diff;
    EEG.etc.analysis.design.isitime = str2double({design.epoch.eventisiTime});
    
    % factor: oddball
    haptics = string({design.epoch.eventfeedback});
    EEG.etc.analysis.design.haptics = categorical(haptics)=="vibro";
    
    % factor: haptics
    oddball = string({design.epoch.eventnormal_or_conflict});
    EEG.etc.analysis.design.oddball = categorical(oddball)=="conflict";
    % factor: direction
    direction = {design.epoch.eventcube};
    direction = strrep(direction, 'CubeLeft (UnityEngine.GameObject)', 'left');
    direction = strrep(direction, 'CubeRight (UnityEngine.GameObject)', 'right');
    direction = strrep(direction, 'CubeMiddle (UnityEngine.GameObject)', 'middle');
    EEG.etc.analysis.design.direction = string(direction);
    % factor: trial number
    EEG.etc.analysis.design.trial_number = str2double({design.epoch.eventtrial_nr});
    % factor: sequence
    count = 0;
    for i = 1:size(design.epoch,2)
        if ~EEG.etc.analysis.design.oddball(1,i)
            count = count+1;
            EEG.etc.analysis.design.sequence(i) = 0;
        else
            EEG.etc.analysis.design.sequence(i) = count;
            count = 0;
        end
    end
    
    %% separation ERP, ERSP and mocap processing
    
    % copy for ERSP before filtering for ERP to maintain freqs > highcutoff ERP filter
    ERSP_event = EEG;
    ERSP_base = EEG;
    
    % mocap
    mocap.event = EEG.event(touch_ixs);
    mocap = pop_epoch( mocap, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');    
    
    %% processing: ERPs

    % for channel ERPs project out eye component activation time courses
    eyes = find(EEG.etc.ic_classification.ICLabel.classifications(:,3) > bemobil_config.eye_threshold);
    line = find(EEG.etc.ic_classification.ICLabel.classifications(:,5) > bemobil_config.line_threshold);
    ERP = pop_subcomp(EEG, unique([eyes', line']));
    
    % filter
    ERP = pop_eegfiltnew(ERP, bemobil_config.filter_plot_low, bemobil_config.filter_plot_high);
    ERP_event = ERP;
    ERP_base = ERP;
    
    % overwrite events with clean epoch events
    ERP_event.event = EEG.event(touch_ixs); % touches
    ERP_base.event = EEG.event(spawn_ixs); % spawns
    
    % ERPs: both EEG and mocap
    ERP_event = pop_epoch( ERP_event, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    ERP_base = pop_epoch( ERP_base , bemobil_config.epoching.base_epoching_event, bemobil_config.epoching.base_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    
    % channel ERP
    % EEG: find ~10% noisiest epoch indices by searching for large amplitude
    % fluctuations on the channel level using eeglab auto_rej function
    [~, rmepochs] = pop_autorej(ERP_event, 'maxrej', 1, 'nogui','on','eegplot','off');
    EEG.etc.analysis.erp.rm_ixs = sort([bad_tr_ixs, rmepochs]);
    % baseline correct
    base_win_samples = (bemobil_config.epoching.base_win * ERP_base.srate) ...
        + abs(bemobil_config.epoching.base_epochs_boundaries) * ERP_base.srate;
    channel_baseline = mean(ERP_base.data(:,base_win_samples(1):base_win_samples(2),:),2);
    ERP_event.data = ERP_event.data - channel_baseline;
    EEG.etc.analysis.erp.data = ERP_event.data;
       
    %% ERSP comps and chans
    
    ERSP_event.event = EEG.event(touch_ixs); % touches
    ERSP_base.event = EEG.event(spawn_ixs); % spawns
    ERSP_event = pop_epoch( ERSP_event, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    ERSP_base = pop_epoch( ERSP_base, bemobil_config.epoching.base_epoching_event, bemobil_config.epoching.base_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    
    % find bad epochs based on component ERP: actually select good
    % components as they should be most relevant for the cleaning as they
    % are the ones later being analysed
    % about the IClabel threshold: Marius said thats what Luca used, cite!
    comps = find(EEG.etc.ic_classification.ICLabel.classifications(:,1) > bemobil_config.brain_threshold);
    [~, rmepochs] = pop_autorej(ERSP_event, 'electrodes', [], 'icacomps', comps, 'maxrej', 1, 'nogui','on','eegplot','off');
    EEG.etc.analysis.ersp.rm_ixs = sort([bad_tr_ixs, rmepochs]);
    
    for comp = 1:size(ERSP_event.icaact,1) % loop through all components
        tic
        
        % event ersp
        [~,~,~,ersp_times,ersp_freqs,~,~,tfdata] = newtimef(ERSP_event.icaact(comp,:,:),...
            ERSP_event.pnts,...
            [ERSP_event.times(1) ERSP_event.times(end)],...
            EEG.srate,...
            'cycles',fft_options.cycles,...
            'freqs',fft_options.freqrange,...
            'freqscale',fft_options.freqscale,...
            'padratio',fft_options.padratio,...
            'baseline',[NaN],... % no baseline, since that is only a subtraction of the freq values, we do it manually
            'nfreqs',fft_options.n_freqs,...
            'timesout',fft_options.timesout,...
            'plotersp','off',...
            'plotitc','off',...
            'verbose','off');
       
        EEG.etc.analysis.ersp.tf_event_raw_power(comp,:,:,:) = abs(tfdata); %discard phase (complex valued)
        EEG.etc.analysis.ersp.tf_event_times = ersp_times;
        EEG.etc.analysis.ersp.tf_event_freqs = ersp_freqs;
        
        % base ersp
        [~,~,~,ersp_times,ersp_freqs,~,~,tfdata] = newtimef(ERSP_base.icaact(comp,:,:),...
            ERSP_base.pnts,...
            [ERSP_base.times(1) ERSP_base.times(end)],...
            EEG.srate,...
            'cycles',fft_options.cycles,...
            'freqs',fft_options.freqrange,...
            'freqscale',fft_options.freqscale,...
            'padratio',fft_options.padratio,...
            'baseline',[NaN],... % no baseline, since that is only a subtraction of the freq values, we do it manually
            'nfreqs',fft_options.n_freqs,...
            'timesout',fft_options.timesout,...
            'plotersp','off',...
            'plotitc','off',...
            'verbose','off');
        
        % remove leading and trailing samples
        win = bemobil_config.epoching.base_win * 1000;
        EEG.etc.analysis.ersp.tf_base_raw_power(comp,:,:) = squeezemean(abs(tfdata(:,find(ersp_times>win(1),1,'first'):find(ersp_times<=win(2),1,'last'),:)),2);

% test grand mean ersp: everything looks good, 14.02.2020
%         ev = squeezemean(EEG.etc.analysis.ersp.tf_event_raw_power,3);
%         base = squeezemean(EEG.etc.analysis.ersp.tf_base_raw_power,2);
%         ev_db = 10.*log10(ev./base);
%         figure;imagesc(EEG.etc.analysis.ersp.tf_event_times, EEG.etc.analysis.ersp.tf_event_freqs, ev_db, [-1 1]);axis xy;cbar;
  
        toc
    end
    
    %% add_xyz MOCAP: Velocity at time points before events of interest
        
    % save x,y,z of hand
    EEG.etc.analysis.mocap.x = mocap.data(7,:,:);
    EEG.etc.analysis.mocap.y = mocap.data(8,:,:);
    EEG.etc.analysis.mocap.z = mocap.data(9,:,:);
    
    % save 3D magnitude of velocity and acceleration
    EEG.etc.analysis.mocap.mag_vel = squeeze(sqrt(mocap.data(7,:,:).^2 +...
            mocap.data(8,:,:).^2 +...
            mocap.data(9,:,:).^2));
    EEG.etc.analysis.mocap.mag_acc = squeeze(sqrt(mocap.data(13,:,:).^2 +...
            mocap.data(14,:,:).^2 +...
            mocap.data(15,:,:).^2));
    
    %% add back unfiltered epoched data and remove bad ERSP trials for creating ERSP study
    
    % add back raw unfiltered epoch data
    EEG = pop_epoch( EEG, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    EEG = eeg_checkset(EEG);
    
    % remove trials
    bad_trials = zeros(1,size(EEG.epoch,2));
    bad_trials(EEG.etc.analysis.ersp.rm_ixs) = 1;
    EEG = pop_rejepoch(EEG, bad_trials, 0);  
    
    % save
    pop_saveset(EEG, 'filename', [bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.epochs_filename] , 'filepath', output_filepath);
    
end

%% build eeglab study (DONE)

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

disp('Precomputing topographies and spectra.')
[STUDY ALLEEG] = std_precomp(STUDY, ALLEEG, 'components',...
    'recompute','on',...
    'scalp','on','spec','on',...
    'specparams',{'specmode' 'fft' 'logtrials' 'on' 'freqrange' [3 40]});

% save study
disp('Saving STUDY...')
mkdir(output_path)
[STUDY EEG] = pop_savestudy( STUDY, EEG, 'filename', bemobil_config.study_filename,'filepath',output_path);
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
eeglab redraw
disp('...done')

%% override icaersp by own computations (DONE)

input_path = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder];
for subject = subjects  % do it for all subjects
    disp(['Subject: ' num2str(subject)])
    input_filepath = [input_path bemobil_config.filename_prefix num2str(subject)];
    EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
        bemobil_config.epochs_filename], 'filepath', input_filepath);

    % good trials ixs
    good_trials = ones(1,size(EEG.etc.analysis.design.oddball,2));
    good_trials(EEG.etc.analysis.ersp.rm_ixs) = 0;
    good_trials = logical(good_trials);
    
    comps = 1:size(EEG.icaact,1);
    
    % settings
    options = {};
    options = { options{:}  'components' comps 'freqs' fft_options.freqrange 'timewindow' [bemobil_config.epoching.event_win] ...
        'cycles' fft_options.cycles 'padratio' fft_options.padratio 'alpha' fft_options.alpha ...
        'type' 'ersp' 'powbase' NaN };
    parameters_icaersp = { 'cycles', fft_options.cycles, 'padratio', NaN, 'alpha', NaN, 'freqscale', fft_options.freqscale};
    
    %open EEG_rej file for checking number of trials
    trialindices = num2cell(1:EEG.trials);
    all_ersp   = [];
    
    for IC = comps % for each (specified) component
        disp(['IC: ' num2str(IC)])
        
        % load full grand average ERSP
        all_ersp.(['comp' int2str(IC) '_ersp']) = squeezemean(EEG.etc.analysis.ersp.tf_event_raw_power(IC,:,:,good_trials),4);
        all_ersp.(['comp' int2str(IC) '_erspbase']) = squeezemean(EEG.etc.analysis.ersp.tf_base_raw_power(IC,:,good_trials),3);
        all_ersp.(['comp' int2str(IC) '_ersp']) = 10.*log10(all_ersp.(['comp' int2str(IC) '_ersp']) ./ all_ersp.(['comp' int2str(IC) '_erspbase'])');
        all_ersp.(['comp' int2str(IC) '_erspboot']) = [];
        
    end % for every component
    
    % Save ERSP into file
    % -------------------
    all_ersp.freqs      = EEG.etc.analysis.ersp.tf_event_freqs;
    all_ersp.times      = EEG.etc.analysis.ersp.tf_event_times;
    all_ersp.datatype   = 'ERSP';
    all_ersp.datafiles  = bemobil_computeFullFileName( { EEG.filepath }, { EEG.filename });
    all_ersp.datatrials = trialindices;
    all_ersp.parameters = parameters_icaersp;
    %std_savedat( [input_filepath '/' bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.epochs_filename(1:end-4) '.icaersp'], all_ersp);
    std_savedat( [input_filepath '/design1_' num2str(subject) '.icaersp'], all_ersp);
    
end % for every participant
disp('Done.')

%% pre clustering (DONE)

% set fpaths
input_path = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder];
output_path = [bemobil_config.study_folder bemobil_config.study_level];

% determine k for k-means clustering
for i = 1:size(STUDY.datasetinfo,2)
    nr_comps(i) = size(STUDY.datasetinfo(i).comps,2);
end
bemobil_config.STUDY_n_clust = round(bemobil_config.IC_percentage * mean(nr_comps));

% create preclustering array that is used for clustering and save study
% [STUDY ALLEEG] = std_preclust(STUDY, ALLEEG, [],...
%     { 'erp'   'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.erp 'timewindow' bemobil_config.event_win} ,...                         
%     { 'ersp'  'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.ersp 'timewindow' bemobil_config.event_win 'freqrange' [ 3 40] } ,...
%     { 'spec'  'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.spectra 'freqrange'  [ 3 40 ] } , ...
%     { 'scalp' 'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.scalp_topographies 'abso' 1 } , ...
%     { 'dipoles'         'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.dipoles } , ...
%     { 'finaldim' 'npca' 10 });

[STUDY ALLEEG] = std_preclust(STUDY, ALLEEG, [],...
    { 'scalp' 'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.scalp_topographies 'abso' 1 } , ...
    { 'spec'  'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.spectra 'freqrange'  [ 3 40 ] } , ...
    { 'ersp'  'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.ersp 'freqrange' [ 3 40] } ,...
    { 'dipoles'         'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.dipoles } , ...
    { 'finaldim' 'npca' 10 });

% store essential info in STUDY struct for later reading
STUDY.bemobil.clustering.preclustparams = STUDY.cluster.preclust.preclustparams;
STUDY.bemobil.clustering.preclustparams.clustering_weights = bemobil_config.STUDY_clustering_weights;
STUDY.bemobil.clustering.n_clust = bemobil_config.STUDY_n_clust;

% save study
disp('Saving STUDY...')
mkdir(output_path)
[STUDY EEG] = pop_savestudy( STUDY, EEG, 'filename', bemobil_config.study_filename,'filepath',output_path);
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
eeglab redraw
disp('...done')

%% repeated clustering (DONE)

input_path = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder];
output_path = [bemobil_config.study_folder bemobil_config.study_level];
path_clustering_solutions = [bemobil_config.STUDY_filepath_clustering_solutions bemobil_config.study_filename(1:end-6) '/' ...
    [num2str(bemobil_config.STUDY_cluster_ROI_talairach.x) '_' num2str(bemobil_config.STUDY_cluster_ROI_talairach.y) '_' num2str(bemobil_config.STUDY_cluster_ROI_talairach.z)] '-location_'...
    num2str(bemobil_config.STUDY_n_clust) '-cluster_' num2str(bemobil_config.outlier_sigma) ...
    '-sigma_' num2str(bemobil_config.STUDY_clustering_weights.dipoles) '-dipoles_' num2str(bemobil_config.STUDY_clustering_weights.spectra) '-spec_'...
    num2str(bemobil_config.STUDY_clustering_weights.scalp_topographies) '-scalp_' num2str(bemobil_config.STUDY_clustering_weights.erp) '-erp_'...
    num2str(bemobil_config.n_iterations) '-iterations'];

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

disp('Saving STUDY clustering solution')
cluster = {STUDY.cluster};
save(['/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing/'...
    'cluster_ROI_' ...
    num2str(bemobil_config.STUDY_cluster_ROI_talairach.x) '_' ...
    num2str(bemobil_config.STUDY_cluster_ROI_talairach.y) '_' ...
    num2str(bemobil_config.STUDY_cluster_ROI_talairach.z) '.mat']);
disp('...done')

