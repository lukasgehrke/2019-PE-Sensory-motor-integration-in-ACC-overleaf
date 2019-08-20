% Prediction Error (2019) Study parameters and folder structure
cfg.subjects = 2:20;

%%% Filename and folder structure informations. folders will be created automatically!
cfg.folders.study_folder = 'P:\Lukas_Gehrke\studies\Prediction_Error\data\';

%% mocap processing

cfg.epoching.event_epochs_boundaries_mocap = [0 1];

%% EEG processing
cfg.epoching.event_epochs_boundaries = [-.3  1];
cfg.epoching.event_epoching_event = {'box:touched'}; 

cfg.epoching.base_epochs_boundaries = [-2  0];
cfg.epoching.base_epoching_event = {'box:spawned'}; 

% filtering
cfg.filter_plot_low = 1;
cfg.filter_plot_high = 15;

% study parameters
% single subject final datasets and epochs
cfg.study.study_filename = 'IMT2.study';
cfg.study.STUDY_components_to_use = 1:160;
% precluster
cfg.study.STUDY_clustering_weights = struct('dipoles', 6, 'scalp_topographies', 1, 'spectra', 1, 'ERSPs', 3);
cfg.study.STUDY_clustering_freqrange = [3 60];

% ERSPs
% baseline
cfg.ersp.baseline = [-300 -100];
cfg.ersp.n_times = 300;
cfg.ersp.trial_normalization = true;
cfg.ersp.baseline_start_end = cfg.ersp.baseline;

% fft options
cfg.ersp.fft_cycles = [3 0.5];
cfg.ersp.fft_freqrange = [3 100];
cfg.ersp.fft_padratio = 2;
cfg.ersp.fft_freqscale = 'log';
cfg.ersp.fft_alpha = NaN;
cfg.ersp.fft_powbase = NaN;
cfg.ersp.fft_c_type   = 'ersp'; % 'itc' 'both'
cfg.ersp.n_freqs = 98;

% repeated clustering, TODO set Talairach of peak interest
% % RSC Spot Rotation
cfg.study.STUDY_cluster_ROI_talairach = struct('x', 0, 'y', -45, 'z', 10);
% % RSC more posterior
% % MNI ('x', 0, 'y', -55, 'z', 10)
% % https://www.biorxiv.org/content/biorxiv/early/2017/10/09/200576.full.pdf
% % https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5321500/
% STUDY_cluster_ROI_talairach = struct('x', 0, 'y', -55, 'z', 10);

%     quality_measure_weights         - vector of weights for quality measures. 6 entries: subjects, ICs/subjects, normalized
%                                     spread, mean RV, distance from ROI, mahalanobis distance from median of multivariate
%                                     distribution (put this very high to get the most "normal" solution)
% STUDY_quality_measure_weights = [3,-1,-2,-1,-2,-1];
cfg.study.STUDY_quality_measure_weights = [3,-1,-1,-1,-2,-1];

% calculate how many ICs remain in the STUDY and take 70% of that for the k
% clusters
cfg.study.IC_percentage = .7;
cfg.study.outlier_sigma = 3;
cfg.study.n_iterations = 10000;

cfg.study.clustering.do_clustering = 1;
cfg.study.clustering.do_multivariate_data = 1;
cfg.study.clustering.STUDY_filepath_clustering_solutions = 'clustering_solutions\';
cfg.study.clustering.filename_clustering_solutions = 'solutions';
cfg.study.clustering.filepath_multivariate_data = '';
cfg.study.clustering.filename_multivariate_data = 'multivariate_data';

% all clusters with muscle and eyes



