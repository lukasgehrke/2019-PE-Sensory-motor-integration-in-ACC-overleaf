%close all; clear all; clc;

if ~exist('ALLEEG','var')
	eeglab;
end

% add downloaded analyses code to the path
PE_config;

% BIDS data download folder
bemobil_config.study_folder = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/BIDS/processing';
bemobil_config.BIDS_folder = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/BIDS';

%% event and baseline epoching, clean epoch indices (autorej,3 function), mocap data (DONE)

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

for subject = subjects
	
    %% load BIDS (with AMICA results) set
    
    %% parsing event structure
        
    % change field name of 'condition' so its not a study thing
    oldField = 'condition';
    newField = 'feedback';
    [EEG.event.(newField)] = EEG.event.(oldField);
    EEG.event = rmfield(EEG.event,oldField);
    
    % get event indices
    touch_ixs = find(strcmp({EEG.event.trial_type}, 'box:touched'));
    spawn_ixs = find(strcmp({EEG.event.trial_type}, 'box:spawned'));
        
    %% build design matrix
    
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
    
    %% removing ICs and cleaning epochs
    
    no_brain = find(EEG.etc.ic_classification.ICLabel.classifications(:,1) < bemobil_config.brain_threshold);
    EEG = pop_subcomp(EEG, no_brain);
    EEG = eeg_checkset(EEG); % recomp ICA activation
    
    % overwrite events with clean epoch events
    clean_EEG = EEG;
    clean_EEG.event = clean_EEG.event(touch_ixs); % touches
    
    % ERPs: both EEG and mocap
    clean_EEG = pop_epoch( clean_EEG, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    
    % EEG: find ~10% noisiest epoch indices by searching for large amplitude
    % fluctuations on the channel level using eeglab auto_rej function
    clean_EEG.event = []; % for speed up
    [~, rmepochs] = pop_autorej(clean_EEG, 'maxrej', 2, 'nogui','on','eegplot','off');
    EEG.etc.analysis.design.rm_ixs = sort([bad_tr_ixs, rmepochs]); % combine noisy epochs with epochs of long reaction times
    
    %% filtered ERPs
    
    ERP = EEG;
    ERP.event = ERP.event(touch_ixs); % touches
    ERP = pop_eegfiltnew(ERP, bemobil_config.filter_plot_low, bemobil_config.filter_plot_high);
    ERP = pop_epoch( ERP, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    EEG.etc.analysis.filtered_erp.chan = ERP.data(bemobil_config.channels_of_int,:,:); % save space saving only channels of int FZ, PZ, FCz
    EEG.etc.analysis.filtered_erp.comp = ERP.icaact;
    
    %% computing ERSP
    
    ERSP_event = EEG;
    ERSP_event.event = EEG.event(touch_ixs); % touches
    ERSP_event = pop_epoch( ERSP_event, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    
    ERSP_base = EEG;
    ERSP_base.event = EEG.event(spawn_ixs); % spawns
    ERSP_base = pop_epoch( ERSP_base, bemobil_config.epoching.base_epoching_event, bemobil_config.epoching.base_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    
    for comp = 1:size(ERSP_event.icaact,1) % loop through all components
        tic
        
        % event ersp
        [~,~,~,ersp_times_ev,ersp_freqs_ev,~,~,tfdata] = newtimef(ERSP_event.icaact(comp,:,:),...
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
        EEG.etc.analysis.ersp.tf_event_raw_power(comp,:,:,:) = abs(tfdata).^2; %discard phase (complex valued)
        
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
        EEG.etc.analysis.ersp.tf_base_raw_power(comp,:,:) = squeezemean(abs(tfdata(:,find(ersp_times>win(1),1,'first'):find(ersp_times<=win(2),1,'last'),:)).^2,2);

% test grand mean ersp: everything looks good, 14.02.2020
        ev = squeezemean(EEG.etc.analysis.ersp.tf_event_raw_power,3);
        base = squeezemean(EEG.etc.analysis.ersp.tf_base_raw_power,2);
        ev_db = 10.*log10(ev./base);
        figure;imagesc(EEG.etc.analysis.ersp.tf_event_times, EEG.etc.analysis.ersp.tf_event_freqs, ev_db, [-1 1]);axis xy;cbar;
  
        toc
    end
    
    % save times and freqs
    EEG.etc.analysis.ersp.tf_event_times = ersp_times_ev;
    EEG.etc.analysis.ersp.tf_event_freqs = ersp_freqs_ev;
    
    %% add_xyz MOCAP: Velocity at time points before events of interest
    
    % mocap
    mocap.event = EEG.event(touch_ixs);
    mocap = pop_epoch( mocap, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');    
    
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
    
    %% prepare for study, i.e. epoch and save
    
    touch_ixs(EEG.etc.analysis.design.rm_ixs) = []; % remove bad trials
    EEG.event = EEG.event(touch_ixs);
    EEG = pop_epoch( EEG, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
%     
%     % save
    pop_saveset(EEG, 'filename', [bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.epochs_filename] , 'filepath', output_filepath);
    
end

%% run BCI using Matlab2014a (DONE)

bcilab;

patterns_t = [];
weights_t = [];
dipoles_t = [];

for subject = subjects
    
    % load EEG and mocap data
	disp(['Subject #' num2str(subject)]);
    
    % filepaths
	input_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	
    %EEG: load data
    EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.copy_weights_interpolate_avRef_filename], 'filepath', input_filepath);
	epochs = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.epochs_filename], 'filepath', input_filepath);
    
    % remove non brain ICs, same as above but need to do again since
    % version incompatibilities
    %no_brain = find(EEG.etc.ic_classification.ICLabel.classifications(:,1) < bemobil_config.brain_threshold);
    no_brain = find(EEG.etc.ic_classification.ICLabel.classifications(:,1) < .5);
    EEG = pop_subcomp(EEG, no_brain);
    EEG = eeg_checkset(EEG); % recompute icaact
    EEG = parse_events_PE(EEG);
    
    % get event indices
    touch_ixs = find(strcmp({EEG.event.type}, 'box:touched'));
    % select touch events
    EEG.event = EEG.event(touch_ixs);
    % remove bad epochs
    EEG.event(epochs.etc.analysis.design.rm_ixs) = [];
    % make event classes: synchronous and asynchronous
    async_ixs = find(strcmp({EEG.event.normal_or_conflict}, 'conflict'));
    sync_ixs = find(strcmp({EEG.event.normal_or_conflict}, 'normal'));
    % match class size
    sync_ixs = randsample(sync_ixs, size(async_ixs,2));
    % select those events
    EEG.event = EEG.event(union(async_ixs, sync_ixs));
    % copy targetmarkers
    [EEG.event.type] = EEG.event.normal_or_conflict;
    % remove parts on EEG set to speed up bcilab
    EEG.etc = [];
    
    % training the classifier! (assuming you have some data loaded as 'EEG')
    [trainloss, model, stats] = bci_train('Data', EEG,...
        'Approach', bemobil_config.lda.approach,...
        'TargetMarkers', bemobil_config.lda.targetmarkers,...
        'EvaluationScheme', {'chron', bemobil_config.lda.evalfolds, bemobil_config.lda.evalmargin},...
        'OptimizationScheme', {'chron', bemobil_config.lda.parafolds, bemobil_config.lda.paramargin});
    
    disp(['training mis-classification rate: ' num2str(trainloss*100,3) '%'])
    %bci_visualize(model)
    
    % calculate results
    correct(subject) = 100 - trainloss*100;
    chance = simulateChance([size(sync_ixs,2), size(async_ixs,2)], .05);
    chance_level(subject) = chance(3);

    % stats contains some statistics. for example, the classifier accuracy is 1-stats.mcr,
    % and stats.TP, stats.TN, etc. contain the true positive, true negative etc. rates.
    % those figures reflect the mean across folds; stats.per_fold contains the individual values.
    all_stats(subject) = stats;

    % model is the calibrated model, containing i.a. LDA filter weights ...
    ldaweights = model.predictivemodel.model.w;
    % ... which can also be transformed into patterns
    ldapatterns = (reshape(ldaweights, numel(model.featuremodel.chanlocs), [])' * model.featuremodel.cov)';
    
    % getting source dipole weights by projecting LDA weights through ICA unmixing matrix
    weights = (EEG.icaweights * EEG.icasphere) * ldapatterns;
    weights = abs(weights);
    % normalizing across time windows
    weights = weights / sum(weights(:));
    
    % computing control signal for each window
    ldaweights = reshape(model.predictivemodel.model.w, numel(model.featuremodel.chanlocs), []);
    
    % for each window compute control singal
    % first filter 
    EEG = pop_eegfiltnew(EEG, bemobil_config.filter_plot_low, bemobil_config.filter_plot_high);
    for window = 1:size(ldaweights,2) % 3rd dimension in results is window
        windowweights = ldaweights(:, window);
        % rms-normalising ldaweights
        windowweights = windowweights / rms(windowweights); 
        % epoch data synchronous class
        ERP_sync = pop_epoch( EEG, {'normal'}, [-.2 .8], 'newname', 'epochs', 'epochinfo', 'yes');
        control_signal(subject,1,window,:) = mean(bsxfun(@times, mean(ERP_sync.data, 3), windowweights)); %2nd dimension is class
        % epoch data asynchronous class
        ERP_async = pop_epoch( EEG, {'conflict'}, [-.2 .8], 'newname', 'epochs', 'epochinfo', 'yes');
        control_signal(subject,2,window,:) = mean(bsxfun(@times, mean(ERP_async.data, 3), windowweights)); %2nd dimension is class
    end
    
    % save all subjects
    patterns_t = vertcat(patterns_t, ldapatterns);
    weights_t = vertcat(weights_t, weights);
    dipoles_t = vertcat(dipoles_t, get_dipoles(EEG));

    clear EEG tmpEEG
end

correct = correct(subjects);
chance_level = chance_level(subjects);
all_stats = all_stats(subjects);
control_signal = control_signal(subjects,:,:,:);

lda_results = struct('patterns', patterns_t,...
    'weights', weights_t,...
    'dipoles', dipoles_t,...
    'correct', correct,...
    'chance_level', chance_level,...
    'all_stats', all_stats,...
    'control_signal', control_signal);

% ttest between chance and correct (DONE)
[H,P,CI,STATS] = ttest(lda_results.correct,lda_results.chance_level);
% save results
lda_results.ttest.h = H;
lda_results.ttest.p = P;
lda_results.ttest.ci = CI;
lda_results.ttest.stats = STATS;

save([bemobil_config.study_folder bemobil_config.study_level 'lda_results_05_prob_brain_base_removal_0-05.mat'], 'lda_results');

%% build eeglab study (DONE)

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
    good_trials(EEG.etc.analysis.design.rm_ixs) = 0;
    good_trials = logical(good_trials);
    
    comps = 1:size(EEG.icaact,1);
    
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
    %std_savedat( [input_filepath '/design1_' num2str(subject) '.icaersp'], all_ersp);
    std_savedat( [input_filepath '/s' num2str(subject) 'epochs_new.icaersp'], all_ersp);
    
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
bemobil_config.STUDY_n_clust = round(bemobil_config.IC_percentage * median(nr_comps));

% create preclustering array that is used for clustering and save study
[STUDY ALLEEG] = std_preclust(STUDY, ALLEEG, [],...
    { 'scalp' 'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.scalp_topographies 'abso' 1 } , ...
    { 'spec'  'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.spectra 'freqrange'  [ 3 80 ] } , ...
    { 'ersp'  'npca' 10 'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.ersp 'freqrange' [ 3 80] } ,...
    { 'dipoles'         'norm' 1 'weight' bemobil_config.STUDY_clustering_weights.dipoles } , ...
    { 'finaldim' 'npca' 10 });

% % store essential info in STUDY struct for later reading
% STUDY.bemobil.clustering.preclustparams = STUDY.cluster.preclust.preclustparams;
% STUDY.bemobil.clustering.preclustparams.clustering_weights = bemobil_config.STUDY_clustering_weights;
% STUDY.bemobil.clustering.n_clust = bemobil_config.STUDY_n_clust;

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
