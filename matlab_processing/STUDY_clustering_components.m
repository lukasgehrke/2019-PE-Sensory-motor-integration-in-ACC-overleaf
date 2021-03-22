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

%% remove ICS from epochs files and save

for subject = subjects
    disp(['Subject #' num2str(subject) ]);
    EEG = pop_loadset(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_epochs_box_touched.set']));
    [EEG, bemobil_config] = select_ICs_pe(EEG, bemobil_config);
    pop_saveset(EEG, fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_epochs_box_touched.set']));
end

%% build eeglab study

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 1, 'option_savetwofiles', 1, 'option_saveversion6', 0,...
    'option_single', 1, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1,...
    'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0,...
    'option_checkversion', 1, 'option_chat', 1);
STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];

for subject = subjects
    disp(['Subject #' num2str(subject) ]);
    EEG = pop_loadset(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_epochs_box_touched.set']));
    [EEG, bemobil_config] = select_ICs_pe(EEG, bemobil_config);
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
end
eeglab redraw
disp('All study files loaded. Creating STUDY...')
command = cell(0,0);
for set = 1:length(subjects)
    command{end+1} = {'index' set 'subject' num2str(subjects(set)) };
end

% create study
[STUDY ALLEEG] = std_editset( STUDY, ALLEEG, 'name', ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_' bemobil_config.study_filename],...
    'commands',command, 'updatedat','on','savedat','off','rmclust','on' );
[STUDY ALLEEG] = std_checkset(STUDY, ALLEEG);
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];

disp('Precomputing topographies and spectra.')
[STUDY ALLEEG] = std_precomp(STUDY, ALLEEG, 'components',...
    'recompute','on',...
    'scalp','on','spec','on',...
    'specparams',{'specmode' 'fft' 'logtrials' 'on' 'freqrange' [3 40]});

% save study
disp('Saving STUDY...')
[STUDY EEG] = pop_savestudy( STUDY, EEG, 'filename', ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_' bemobil_config.study_filename],...
    'filepath', bemobil_config.study_folder);
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
eeglab redraw
disp('...done')

%% override icaersp by own single-trial computations

for subject = subjects  % do it for all subjects
    disp(['Subject: ' num2str(subject)])
    
    EEG = pop_loadset(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_epochs_box_touched.set']));
    comps = 1:size(EEG.icaact,1);
    load(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_single_trial_dmatrix.mat']));
    good_comps_ori_ix = find(EEG.etc.ic_classification.ICLabel.classifications(:,1) > bemobil_config.lda.brain_threshold);

    % good trials ixs
    good_trials = ones(1,size(EEG.etc.analysis.design.oddball,2));
    good_trials(EEG.etc.analysis.design.bad_touch_epochs) = 0;
    good_trials = logical(good_trials);
    
    % settings
    options = {};
    options = { options{:}  'components' comps 'freqs' fft_options.freqrange 'timewindow' [bemobil_config.epoching.event_epochs_boundaries] ...
        'cycles' fft_options.cycles 'padratio' fft_options.padratio 'alpha' fft_options.alpha ...
        'type' 'ersp' 'powbase' NaN };
    parameters_icaersp = { 'cycles', fft_options.cycles, 'padratio', NaN, 'alpha', NaN, 'freqscale', fft_options.freqscale};
    
    %open EEG_rej file for checking number of trials
    trialindices = num2cell(1:EEG.trials);
    all_ersp   = [];
    
    for IC = comps % for each (specified) component
        disp(['IC: ' num2str(IC)])
        
        % load full grand average ERSP
        all_ersp.(['comp' int2str(IC) '_ersp']) = squeezemean(single_trial_dmatrix.ersp.tf_event_raw_power(good_comps_ori_ix(IC),:,:,good_trials),4);
        all_ersp.(['comp' int2str(IC) '_erspbase']) = squeezemean(single_trial_dmatrix.ersp.tf_base_raw_power(good_comps_ori_ix(IC),:,good_trials),3);
        all_ersp.(['comp' int2str(IC) '_ersp']) = 10.*log10(all_ersp.(['comp' int2str(IC) '_ersp']) ./ all_ersp.(['comp' int2str(IC) '_erspbase'])');
        all_ersp.(['comp' int2str(IC) '_erspboot']) = [];
        
    end % for every component
    
    % Save ERSP into file
    % -------------------
    all_ersp.freqs      = single_trial_dmatrix.ersp.tf_event_freqs;
    all_ersp.times      = single_trial_dmatrix.ersp.tf_event_times;
    all_ersp.datatype   = 'ERSP';
    all_ersp.datafiles  = bemobil_computeFullFileName( { EEG.filepath }, { EEG.filename });
    all_ersp.datatrials = trialindices;
    all_ersp.parameters = parameters_icaersp;
    
    std_savedat(fullfile(bemobil_config.study_folder, 'data', ['design1_', num2str(subject), '.icaersp']), all_ersp);
    
end % for every participant
disp('Done.')

%% pre-clustering

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
[STUDY ALLEEG] = pop_loadstudy('filename', ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_' bemobil_config.study_filename], 'filepath', bemobil_config.study_folder);
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];

% determine k for k-means clustering
for i = 1:size(STUDY.datasetinfo,2)
    nr_comps(i) = size(STUDY.datasetinfo(i).comps,2);
end
bemobil_config.STUDY_n_clust = round(bemobil_config.IC_percentage * median(nr_comps));

% create preclustering array that is used for clustering and save study
[STUDY ALLEEG] = std_preclust(STUDY, ALLEEG, [],...
    { 'scalp' 'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.scalp_topographies 'abso' 1 } , ...
    { 'spec'  'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.spectra 'freqrange'  [ 3 80 ] } , ...
    { 'ersp'  'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.ersp 'freqrange' [ 3 80] } ,...
    { 'dipoles'         'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.dipoles } , ...
    { 'finaldim' 'npca' 10 });

% store essential info in STUDY struct for later reading
STUDY.bemobil.clustering.preclustparams = STUDY.cluster.preclust.preclustparams;
STUDY.bemobil.clustering.preclustparams.clustering_weights = bemobil_config.STUDY_clustering_weights;
STUDY.bemobil.clustering.n_clust = bemobil_config.STUDY_n_clust;

% save study
disp('Saving STUDY...')
[STUDY EEG] = pop_savestudy( STUDY, EEG, 'filename', bemobil_config.study_filename,'filepath', bemobil_config.study_folder);
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
eeglab redraw
disp('...done')

%% repeated clustering to target ROI: 

% [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
% [STUDY ALLEEG] = pop_loadstudy('filename', ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_' bemobil_config.study_filename], 'filepath', bemobil_config.study_folder);
% CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];

% determine k for k-means clustering
for i = 1:size(STUDY.datasetinfo,2)
    nr_comps(i) = size(STUDY.datasetinfo(i).comps,2);
end
bemobil_config.STUDY_n_clust = round(bemobil_config.IC_percentage * median(nr_comps));

path_clustering_solutions = fullfile(bemobil_config.study_folder, bemobil_config.study_filename(1:end-6), [...
    [num2str(bemobil_config.STUDY_cluster_ROI_talairach.x) '_' num2str(bemobil_config.STUDY_cluster_ROI_talairach.y) '_' num2str(bemobil_config.STUDY_cluster_ROI_talairach.z)] '-location_'...
    num2str(bemobil_config.STUDY_n_clust) '-cluster_' num2str(bemobil_config.outlier_sigma) ...
    '-sigma_' num2str(bemobil_config.STUDY_clustering_weights.dipoles) '-dipoles_' num2str(bemobil_config.STUDY_clustering_weights.spectra) '-spec_'...
    num2str(bemobil_config.STUDY_clustering_weights.scalp_topographies) '-scalp_' num2str(bemobil_config.STUDY_clustering_weights.erp) '-erp_'...
    num2str(bemobil_config.n_iterations) '-iterations']);

% cluster the components repeatedly and use a region of interest and
% quality measures to find the best fitting solution
[STUDY, ALLEEG, EEG] = bemobil_repeated_clustering_and_evaluation(STUDY, ALLEEG, EEG, bemobil_config.outlier_sigma,...
    bemobil_config.STUDY_n_clust, bemobil_config.n_iterations, bemobil_config.STUDY_cluster_ROI_talairach,...
    bemobil_config.STUDY_quality_measure_weights, 1,...
    1, bemobil_config.study_folder, bemobil_config.study_filename, path_clustering_solutions,...
    bemobil_config.filename_clustering_solutions, path_clustering_solutions, bemobil_config.filename_multivariate_data);
    
% save study
disp('Saving STUDY...')
[STUDY EEG] = pop_savestudy( STUDY, EEG, 'filename', bemobil_config.study_filename,'filepath',bemobil_config.study_folder);
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
eeglab redraw

disp('Saving STUDY clustering solution')
cluster = {STUDY.cluster};
save(fullfile(bemobil_config.study_folder, ['cluster_ROI_' ...
    num2str(bemobil_config.STUDY_cluster_ROI_talairach.x) '_' ...
    num2str(bemobil_config.STUDY_cluster_ROI_talairach.y) '_' ...
    num2str(bemobil_config.STUDY_cluster_ROI_talairach.z) '.mat']));
disp('...done')
