%% Processing settings

subjects = 2:20;

bemobil_config.study_folder = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/';
bemobil_config.filename_prefix = 's';
bemobil_config.filenames = {'PredError_block_TestVibro' 'PredError_block_TestVisual'};
bemobil_config.filenames_s3 = {'PredError_block_TestVibro_erste100' 'PredError_block_TestVibro_101bis300' 'PredError_block_TestVisual'};

% be as specific as possible (uses regexp)
bemobil_config.unprocessed_data_streams = {'brainvision_rda_bpn-c012'};
bemobil_config.event_streams = {'unity_markers_prederror_BPN-C043'};
bemobil_config.rigidbody_streams = {'rigid_handr_BPN-C043','rigid_head_BPN-C043'};

% enter channels that you did not use at all (e.g. with the MoBI 160 chan layout, only 157 chans are used):
bemobil_config.channels_to_remove = [];

% enter EOG channel names here:
bemobil_config.eog_channels  = {''};
bemobil_config.rename_channels = {'brainvision_rda_bpn-c012_Fp1' 'Fp1';
    'brainvision_rda_bpn-c012_Fp2' 'Fp2';
    'brainvision_rda_bpn-c012_F7' 'F7';
    'brainvision_rda_bpn-c012_F3' 'F3';
    'brainvision_rda_bpn-c012_Fz' 'Fz';
    'brainvision_rda_bpn-c012_F4' 'F4';
    'brainvision_rda_bpn-c012_F8' 'F8';
    'brainvision_rda_bpn-c012_FC5' 'FC5';
    'brainvision_rda_bpn-c012_FC1' 'FC1';
    'brainvision_rda_bpn-c012_FC2' 'FC2';
    'brainvision_rda_bpn-c012_FC6' 'FC6';
    'brainvision_rda_bpn-c012_T7' 'T7';
    'brainvision_rda_bpn-c012_C3' 'C3';
    'brainvision_rda_bpn-c012_Cz' 'Cz';
    'brainvision_rda_bpn-c012_C4' 'C4';
    'brainvision_rda_bpn-c012_T8' 'T8';
    'brainvision_rda_bpn-c012_TP9' 'TP9';
    'brainvision_rda_bpn-c012_CP5' 'CP5';
    'brainvision_rda_bpn-c012_CP1' 'CP1';
    'brainvision_rda_bpn-c012_CP2' 'CP2';
    'brainvision_rda_bpn-c012_CP6' 'CP6';
    'brainvision_rda_bpn-c012_TP10' 'TP10';
    'brainvision_rda_bpn-c012_P7' 'P7';
    'brainvision_rda_bpn-c012_P3' 'P3';
    'brainvision_rda_bpn-c012_Pz' 'Pz';
    'brainvision_rda_bpn-c012_P4' 'P4';
    'brainvision_rda_bpn-c012_P8' 'P8';
    'brainvision_rda_bpn-c012_PO9' 'PO9';
    'brainvision_rda_bpn-c012_O1' 'O1';
    'brainvision_rda_bpn-c012_Oz' 'Oz';
    'brainvision_rda_bpn-c012_O2' 'O2';
    'brainvision_rda_bpn-c012_PO10' 'PO10';
    'brainvision_rda_bpn-c012_AF7' 'AF7';
    'brainvision_rda_bpn-c012_AF3' 'AF3';
    'brainvision_rda_bpn-c012_AF4' 'AF4';
    'brainvision_rda_bpn-c012_AF8' 'AF8';
    'brainvision_rda_bpn-c012_F5' 'F5';
    'brainvision_rda_bpn-c012_F1' 'F1';
    'brainvision_rda_bpn-c012_F2' 'F2';
    'brainvision_rda_bpn-c012_F6' 'F6';
    'brainvision_rda_bpn-c012_FT9' 'FT9';
    'brainvision_rda_bpn-c012_FT7' 'FT7';
    'brainvision_rda_bpn-c012_FC3' 'FC3';
    'brainvision_rda_bpn-c012_FC4' 'FC4';
    'brainvision_rda_bpn-c012_FT8' 'FT8';
    'brainvision_rda_bpn-c012_FT10' 'FT10';
    'brainvision_rda_bpn-c012_C5' 'C5';
    'brainvision_rda_bpn-c012_C1' 'C1';
    'brainvision_rda_bpn-c012_C2' 'C2';
    'brainvision_rda_bpn-c012_C6' 'C6';
    'brainvision_rda_bpn-c012_TP7' 'TP7';
    'brainvision_rda_bpn-c012_CP3' 'CP3';
    'brainvision_rda_bpn-c012_CPz' 'CPz';
    'brainvision_rda_bpn-c012_CP4' 'CP4';
    'brainvision_rda_bpn-c012_TP8' 'TP8';
    'brainvision_rda_bpn-c012_P5' 'P5';
    'brainvision_rda_bpn-c012_P1' 'P1';
    'brainvision_rda_bpn-c012_P2' 'P2';
    'brainvision_rda_bpn-c012_P6' 'P6';
    'brainvision_rda_bpn-c012_PO7' 'PO7';
    'brainvision_rda_bpn-c012_PO3' 'PO3';
    'brainvision_rda_bpn-c012_POz' 'POz';
    'brainvision_rda_bpn-c012_PO4' 'PO4';
    'brainvision_rda_bpn-c012_PO8' 'PO8'}; % 'E65' 'FCz'    
bemobil_config.ref_channel = 'FCz';

% leave this empty if you have standard channel names that should use standard locations:
% the standard .elc file can be found at "M:/BrainVision Stuff/Cap Layouts/standard_MoBI_channel_locations" and must
% be copied into every subject's data folder (where the .xdf files are)
bemobil_config.channel_locations_filename = '';

% for SSD analysis
bemobil_config.frontal_channames = {'Fz','FCz','F1','F2'};
bemobil_config.parietal_channames = {'Pz','P1','P2','P3','P4'};

% Prediction Error (2019) Study parameters and folder structure
bemobil_config.subjects = 2:20;

%%% Filename and folder structure informations. folders will be created automatically!
bemobil_config.folders.study_folder = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/';

%% EEG and MOCAP epochs processing
bemobil_config.epoching.event_epochs_boundaries = [-3  2]; % larger window for ERSP computation
bemobil_config.epoching.event_epoching_event = {'box:touched'}; 

% ERSPs
bemobil_config.epoching.base_epochs_boundaries = [-1  1];
bemobil_config.epoching.base_win = [-.2 0];
bemobil_config.epoching.base_epoching_event = {'box:spawned'}; 

% settings
fft_options = struct();
fft_options.cycles = [3 0.5];
fft_options.padratio = 2;
fft_options.freqrange = [3 80];
fft_options.freqscale = 'log';
fft_options.n_freqs = 60;
fft_options.timesout = 300;
fft_options.alpha = NaN;
fft_options.powbase = NaN;

% ERP filtering
bemobil_config.filter_plot_low = .1;
bemobil_config.filter_plot_high = 15;
bemobil_config.channels_of_int = [5, 14, 25, 65];
bemobil_config.channels_of_int_labels = {'Fz', 'Cz', 'Pz', 'FCz'};
% channels
% 5: Fz
% 14: Cz
% 25: Pz
% 65: FCz

%% Classifier approach: LDA using windowed means from ERPs as features
wnds = [.05 .1;.1 .15; .15 .2;.2 .25; .25 .3; .3 .35; .35 .4; .4 .45];
base_win = [.0 .05];
bemobil_config.lda.targetmarkers = {'normal','conflict'};
% define approach
bemobil_config.lda.approach = {'Windowmeans' ...
    'SignalProcessing', {'Resampling','off', 'BaselineRemoval', base_win,...
        'EpochExtraction',[base_win(1) max(wnds(:))],'SpectralSelection',[0.1 15]},...
    'Prediction', { ...
        'FeatureExtraction', { ...
            'TimeWindows', wnds}, ...
        'MachineLearning', { ...
            'Learner', {'lda' ...
                'Regularizer', 'shrinkage'}}}};

% without baseline
% bemobil_config.lda.approach = {'Windowmeans' ...
%     'SignalProcessing', {'Resampling','off',...
%         'EpochExtraction',[min(wnds(:)) max(wnds(:))],'SpectralSelection',[0.1 15]},...
%     'Prediction', { ...
%         'FeatureExtraction', { ...
%             'TimeWindows', wnds}, ...
%         'MachineLearning', { ...
%             'Learner', {'lda' ...
%                 'Regularizer', 'shrinkage'}}}};
            
% number of cross-validation folds and trial spacing margins for parameter search ('OptimizationScheme'),
% and performance estimates ('EvaluationScheme'). this is for the default chronological/blockwise
% cross-validation scheme; see utl_crossval for other options. performance estimation can also be 
% left out entirely.
bemobil_config.lda.parafolds = 5;
bemobil_config.lda.paramargin = 5;
bemobil_config.lda.evalfolds = 5;
bemobil_config.lda.evalmargin = 5;

%% study parameters
bemobil_config.study_filename = 'PE_lda_ersp.study';

% no double dipping
%bemobil_config.STUDY_clustering_weights = struct('dipoles', 1, 'scalp_topographies', 0, 'spectra', 0, 'erp', 0);
% spot rotation settings
%STUDY_2_clustering_weights = struct('dipoles', 6, 'scalp_topographies', 1, 'spectra', 1, 'ERSPs', 3);
%STUDY_2_quality_measure_weights = [3,-2,-1,-1,-2,-1];

% default: spot rotation
%bemobil_config.STUDY_clustering_weights = struct('dipoles', 6, 'scalp_topographies', 1, 'spectra', 1, 'ersp', 3, 'erp', 0);
bemobil_config.STUDY_clustering_weights = struct('dipoles', 1, 'scalp_topographies', 0, 'spectra', 0, 'ersp', 0, 'erp', 0);

% dipoledensity clusters weighted by LDA
%bemobil_config.STUDY_cluster_ROI_talairach = struct('x', 0, 'y', -35, 'z', 50); 
bemobil_config.STUDY_cluster_ROI_talairach = struct('x', 20, 'y', -65, 'z', 30); % 20 -65 30 Visual Association Area / Cuneus
%0 -40 30 Posterior Cingulate

%     quality_measure_weights         - vector of weights for quality measures. 6 entries: subjects, ICs/subjects, normalized
%                                     spread, mean RV, distance from ROI, mahalanobis distance from median of multivariate
%                                     distribution (put this very high to get the most "normal" solution)
bemobil_config.STUDY_quality_measure_weights = [2,-2,-1,-1,-2,-1];

% calculate how many ICs remain in the STUDY and take 70% of that for the k
% clusters
bemobil_config.IC_percentage = 1;
bemobil_config.outlier_sigma = 3;
bemobil_config.n_iterations = 10000;
bemobil_config.do_clustering = 1;
bemobil_config.do_multivariate_data = 1;
bemobil_config.STUDY_filepath_clustering_solutions = 'clustering_solutions/';
bemobil_config.filename_clustering_solutions = 'solutions';
bemobil_config.filepath_multivariate_data = '';
bemobil_config.filename_multivariate_data = 'multivariate_data';

%% everything from here is according to the general pipeline, changes only recommended if you know the whole structure

% general foldernames and filenames
bemobil_config.raw_data_folder = '0_raw-data/';
bemobil_config.mobilab_data_folder = '1_mobilab-data/';
bemobil_config.raw_EEGLAB_data_folder = '2_basic-EEGLAB/';
bemobil_config.spatial_filters_folder = '3_spatial-filters/';
bemobil_config.spatial_filters_folder_AMICA = '3-1_AMICA/';
bemobil_config.spatial_filters_folder_SSD = '3-2_SSD/';
bemobil_config.single_subject_analysis_folder = '4_single-subject-analysis/';
bemobil_config.study_level = '5_study_level/';

bemobil_config.merged_filename = 'merged.set';
bemobil_config.preprocessed_filename = 'preprocessed.set';
bemobil_config.interpolated_avRef_filename = 'interpolated_avRef.set';
bemobil_config.filtered_filename = 'filtered.set';
bemobil_config.amica_raw_filename_output = 'postAMICA_raw.set';
bemobil_config.amica_chan_no_eye_filename_output = 'preAMICA_no_eyes.set';
bemobil_config.amica_filename_output = 'postAMICA_cleaned.set';
bemobil_config.warped_dipfitted_filename = 'warped_dipfitted.set';
bemobil_config.copy_weights_interpolate_avRef_filename = 'interp_avRef_ICA.set';
bemobil_config.single_subject_cleaned_ICA_filename = 'cleaned_with_ICA.set';
bemobil_config.ssd_frontal_parietal_filename = 'ssd_frontal_parietal.set';
%bemobil_config.epochs_filename = 'epochs.set';
bemobil_config.epochs_filename = 'epochs_new.set';

bemobil_config.merged_filename_mocap = 'merged_mocap.set';

% preprocessing
bemobil_config.mocap_lowpass = 6;
bemobil_config.rigidbody_derivatives = 2;
bemobil_config.resample_freq = 250;

%%% AMICA
% on some PCs AMICA may crash before the first iteration if the number of
% threads and the amount the data does not suit the algorithm. Jason Palmer
% has been informed, but no fix so far. just roll with it. if you see the
% first iteration working there won't be any further crashes. in this case
% just press "close program" or the like and the bemobil_spatial_filter
% algorithm will AUTOMATICALLY reduce the number of threads and start AMICA
% again. this way you will always have the maximum number
% of threads that should be used for AMICA. check in the
% task manager how many threads you have theoretically available and think
% how much computing power you want to devote for AMICA. on the bpn-s1
% server, 12 is half of the capacity and can be used. be sure to check with
% either Ole or your supervisor and also check the CPU usage in the task
% manager before!

% 4 threads are most effective for single subject speed, more threads don't
% really shorten the calculation time much. best efficiency is using just 1
% thread and have as many matlab instances open as possible (limited by the
% CPU usage). Remember your RAM limit in this case.
bemobil_config.filter_lowCutoffFreqAMICA = 1;
bemobil_config.filter_highCutoffFreqAMICA = [];
bemobil_config.max_threads = 4;
bemobil_config.num_models = 1;

% warp electrodemontage and run dipfit
bemobil_config.warping_channel_names = [];
bemobil_config.residualVariance_threshold = 100;
bemobil_config.do_remove_outside_head = 'off';
bemobil_config.number_of_dipoles = 1;

% IC_label
bemobil_config.brain_threshold = .5;

% FHs cleaning
bemobil_config.buffer_length = 0.49;
bemobil_config.automatic_cleaning_threshold_to_keep = 0.82;

% SSD analysis
bemobil_config.ssd_freq_theta = [4 7; % signal band
	1 10; % noise bandbass (outer edges)
	2.5 8.5]; % noise bandstop (inner edges)
bemobil_config.ssd_freq_alpha = [8 13; % signal band
	5 16; % noise bandbass (outer edges)
	6.5 14.5]; % noise bandstop (inner edges)