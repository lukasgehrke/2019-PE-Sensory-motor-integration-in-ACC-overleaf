%% metadata
bemobil_config.task = 'ReachToTouchPredictionError';

% IC_label settings
% -1 uses the popularity classifier, i.e. every IC gets the class with the highest probability. set a specific threshold
% otherwise, i.e. 0.4 (40% brain probability)
bemobil_config.brain_threshold = -1;

%% EEG and MOCAP epochs processing
bemobil_config.epoching.event_epochs_boundaries = [-3  2]; % larger window for ERSP computation
bemobil_config.epoching.event_epoching_event = {'box_touched'}; 

% ERSPs
bemobil_config.epoching.base_epochs_boundaries = [-1  1];
bemobil_config.epoching.base_win = [-.2 0];
bemobil_config.epoching.base_epoching_event = {'box_spawned'}; 

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
% and performance estimates ('EvaluationScheme'). this is for the default chronologicalblockwise
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
bemobil_config.STUDY_cluster_ROI_talairach = struct('x', 0, 'y', -35, 'z', 50); 
%bemobil_config.STUDY_cluster_ROI_talairach = struct('x', 20, 'y', -65, 'z', 30); % 20 -65 30 Visual Association Area  Cuneus
%0 -40 30 Posterior Cingulate

%     quality_measure_weights         - vector of weights for quality measures. 6 entries: subjects, ICssubjects, normalized
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
bemobil_config.STUDY_filepath_clustering_solutions = 'clustering_solutions';
bemobil_config.filename_clustering_solutions = 'solutions';
bemobil_config.filepath_multivariate_data = '';
bemobil_config.filename_multivariate_data = 'multivariate_data';