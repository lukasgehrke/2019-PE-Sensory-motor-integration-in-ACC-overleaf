%% clear all and load params
close all; clear all;

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
	
    %% obtaining clean epoch indices
    
    % parse events from urevent structure 
    EEG = parse_events_PE(EEG);
    mocap = parse_events_PE(mocap);
    
    % get event indices
    touch_ixs = find(strcmp({EEG.event.type}, 'box:touched'));
    spawn_ixs = find(strcmp({EEG.event.type}, 'box:spawned'));
        
    % 1. remove all trials that have a distance greater than 2s between 
    % spawn and touch: participants where slow to react / start the trial
    latency_diff = (cell2mat({EEG.event(touch_ixs).latency}) - cell2mat({EEG.event(spawn_ixs).latency})) / EEG.srate;
    bad_tr_ixs = find(latency_diff>2);
    
    %% epoching
    
    % overwrite events with clean epoch events
    EEG.event = EEG.event(touch_ixs); % touches
    mocap.event = EEG.event;    
    
    % filter for ERP: has to be done before epoching
    % first copy for ERSP
    EEG_ersp = EEG;
    EEG = pop_eegfiltnew(EEG, bemobil_config.filter_plot_low, bemobil_config.filter_plot_high);
        
    % ERPs: both EEG and mocap
    EEG = pop_epoch( EEG, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    EEG_ersp = pop_epoch( EEG_ersp, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    mocap = pop_epoch( mocap, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
        
    % EEG: find ~10% noisiest epoch indices by searching for large amplitude
    % fluctuations on the channel level using eeglab auto_rej function
    [~, rmepochs] = pop_autorej(EEG, 'maxrej', [1],'nogui','on','eegplot','off');
    EEG.etc.epoching.rm_ixs = [bad_tr_ixs, rmepochs];
    
    % remove epochs
    EEG = pop_rejepoch( EEG, EEG.etc.epoching.rm_ixs, 0);
    EEG_ersp = pop_rejepoch( EEG_ersp, EEG.etc.epoching.rm_ixs, 0);
    mocap = pop_rejepoch( mocap, EEG.etc.epoching.rm_ixs, 0);
    
    % create design matrix / simplify event descriptors
    % factor: oddball
    haptics = string({EEG.epoch.eventcondition});
    EEG.etc.epoching.haptics = categorical(haptics)=="vibro";
    
    % factor: haptics
    oddball = string({EEG.epoch.eventnormal_or_conflict});
    EEG.etc.epoching.oddball = categorical(oddball)=="conflict";
    % factor: direction
    direction = {EEG.epoch.eventcube};
    direction = strrep(direction, 'CubeLeft (UnityEngine.GameObject)', 'left');
    direction = strrep(direction, 'CubeRight (UnityEngine.GameObject)', 'right');
    direction = strrep(direction, 'CubeMiddle (UnityEngine.GameObject)', 'middle');
    EEG.etc.epoching.direction = string(direction);
    % factor: trial number
    EEG.etc.epoching.trial_number = str2double({EEG.epoch.eventtrial_nr});
    % factor: sequence
    count = 0;
    for i = 1:size(EEG.epoch,2)
        if ~EEG.etc.epoching.oddball(1,i)
            count = count+1;
            EEG.etc.epoching.sequence(i) = 0;
        else
            EEG.etc.epoching.sequence(i) = count;
            count = 0;
        end
    end
       
    %% ERSP comps and chans
    
    % 1. calculate newtimef for both baseline and event epochs
    zero = 3*EEG.srate; % [-3 2] epoch around event
    ersp_win = (zero-EEG.srate):zero+2*EEG.srate; % [-1 2] around event
    ersp_win(end) = [];
    
    % settings
    fft_options = struct();
    fft_options.cycles = [3 0.5];
    fft_options.freqrange = [3 40];
    fft_options.freqscale = 'log';
    fft_options.n_freqs = 40;
    fft_options.timesout = 120;
               
    % do the timefreq analysis for ERSP, use unfiltered data for ERSP
    % computation
    data_for_ersp = EEG_ersp.icaact(:,ersp_win,:);
    
    for comp = 1:size(data_for_ersp,1)
        tic
        [~,~,~,ersp_times,ersp_freqs,~,~,tfdata] = newtimef(data_for_ersp(comp,:,:),...
            size(data_for_ersp,2),...
            [-1 2]*1000,...
            EEG.srate,...
            'cycles',fft_options.cycles,...
            'freqs',fft_options.freqrange,...
            'freqscale',fft_options.freqscale,...
            'baseline',[NaN],... % no baseline, since that is only a subtraction of the freq values, we do it manually
            'nfreqs',fft_options.n_freqs,...
            'timesout',fft_options.timesout,...
            'plotersp','off',...
            'plotitc','off',...
            'verbose','off');
                    
        % select relevant data
        event_win_times = bemobil_config.epoching.event_win * 1000;
        tfdata = abs(tfdata); % discard phase
        tfdata = tfdata(:,find(ersp_times>event_win_times(1),1,'first'):find(ersp_times<=event_win_times(2),1,'last'),:);
        times_win = ersp_times(:,find(ersp_times>event_win_times(1),1,'first'):find(ersp_times<=event_win_times(2),1,'last'));
        
        % average across basewin
        base_win_times = bemobil_config.epoching.base_win * 1000;
        ersp_base_power_raw = tfdata(:,find(ersp_times>base_win_times(1),1,'first'):find(ersp_times<=base_win_times(2),1,'last'),:);
        
        % save raw power single trials
        EEG.etc.analysis.ersp.tfdata.comp(comp,:,:,:) = tfdata;
        
        % divisive baseline correction with ensemble baseline
        EEG.etc.analysis.ersp.tf_data_baseline.comp(comp,:) = mean(mean(ersp_base_power_raw,3),2);
        
        % apply divisive baseline to single trials, convert to dB and save single trial
        EEG.etc.analysis.ersp.base_corrected_dB.comp(comp,:,:,:) = 10.*log10(tfdata ./ EEG.etc.analysis.ersp.tf_data_baseline.comp(comp,:)');
        
%         % correct plotting of mean ERSP
%         base_corrected_ersp_dB_win = squeeze(mean(EEG.etc.analysis.ersp.tfdata.comp(comp,:,:,:),4));
%         base_corrected_ersp_dB_win = 10.*log10(base_corrected_ersp_dB_win ./ EEG.etc.analysis.ersp.tf_data_baseline.comp(comp,:)');
%         figure;imagesclogy(times_win, ersp_freqs, base_corrected_ersp_dB_win, max(abs(base_corrected_ersp_dB_win(:)))/2 * [-1 1]);axis xy;cbar;
        toc
    end
    
    % save freqs and times
    EEG.etc.analysis.ersp.times = times_win;
    EEG.etc.analysis.ersp.freqs = ersp_freqs;
    
    %% FC comps, sliding window correlation between 2 sources, do later on second level
    % otherwise results matrix gets huge if correlating every comp with
    % every comp
 
    %% ERPs: components and channels
        
    % non-baseline corrected component ERPs
    EEG.etc.analysis.erp.non_baseline_corrected.comps(:,:,:) = EEG.icaact;
    
    % for channel ERPs project out eye component activation time courses
    eyes = find(EEG.etc.ic_classification.ICLabel.classifications(:,3) > bemobil_config.eye_threshold);
    EEG_erp_chan = pop_subcomp(EEG, eyes);
    
    % non-baseline corrected channel ERPs with eyes projected out
    EEG.etc.analysis.erp.non_baseline_corrected.chans(:,:,:) = EEG_erp_chan.data;
    
    %% TODO:add_xyz MOCAP: Velocity at time points before events of interest
        
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
    
    %% save epoched datasets with clean trial indices of the 2x2 study design
    pop_saveset(EEG, 'filename', [bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.epochs_filename] , 'filepath', output_filepath);
    
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

% add dipselect
command(end+1) = {{ 'dipselect' 0.15 }};

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
                         { 'spec'  'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.spectra 'freqrange'  [ 3 80 ] } , ...
                         { 'erp'   'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.erp 'timewindow' [ -200 700 ] } ,...
                         { 'scalp' 'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.scalp_topographies 'abso' 1 } , ...
                         { 'dipoles'         'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.dipoles } , ...
                         { 'finaldim' 'npca' 10 });
STUDY.bemobil.clustering.preclustparams = STUDY.cluster.preclust.preclustparams;
                     
% % check how many ICs remain and do clustering, adapt to removed RV comps
% c = 1;
% for subject = 1:size(STUDY.datasetinfo,2)
%     n_clust(c) = size(ALLEEG(subject).icaact,1);
%     c = c+1;
% end
% bemobil_config.STUDY_n_clust = round(bemobil_config.IC_percentage * mean(n_clust));

%% alternatively load study

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

%% repeated clustering

% .7 of mean IC per subject
bemobil_config.STUDY_n_clust = round(bemobil_config.IC_percentage * (size(STUDY.etc.preclust.preclustdata,1) / size(STUDY.datasetinfo,2)));

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
disp('...done')

%% save all topos for all subjects

% to fix elocs of one subject
for s = ALLEEG
    pop_topoplot(s,0,[1:size(s.icasphere,1)]);
    saveas(gcf, ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/topomaps/' s.filename(1:3) '.png']);
    close(gcf);
    close all;
end