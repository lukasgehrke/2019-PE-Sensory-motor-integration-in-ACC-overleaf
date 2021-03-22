% run BCI using Matlab2014a

% add downloaded analyses code to the path
addpath(genpath('P:\Lukas_Gehrke\studies\Prediction_Error\publications\2019-PE-Sensory-motor-integration-in-ACC-overleaf\matlab_processing'));
% TODO add to path bemobil_pipeline repository download folder
% TODO add to path custom scripts repository Lukas Gehrke folder

% add path BCILAB
addpath(genpath('P:\Lukas_Gehrke\studies\Prediction_Error\BCILAB'));

% BIDS data download folder
bemobil_config.BIDS_folder = 'P:\Lukas_Gehrke\studies\Prediction_Error\data\BIDS';
% Results output folder -> external drive
bemobil_config.study_folder = fullfile('P:\Lukas_Gehrke\studies\Prediction_Error\data', 'derivatives');

% init
config_processing_pe;
subjects = 1:19;

%% Run LDA

bcilab;

threshs = [0, .5, .7, .8, .9];
for t = threshs
    
    bemobil_config.lda.brain_threshold = t;
    
    patterns_t = [];
    weights_t = [];
    dipoles_t = [];

    modality = 'eeg';

    for subject = subjects

        %% load BIDS (with AMICA results) set

        EEG = pop_loadset(fullfile(bemobil_config.BIDS_folder, ['sub-', sprintf('%03d', subject)], modality,...
            ['sub-', sprintf('%03d', subject), '_task-', bemobil_config.task, '_', modality '.set']));

        %% make design matrix, exclude training trials and EMS condition, clean up, find bad epochs

        EEG = pe_remove_training_block(EEG);
        EEG.event(find(strcmp({EEG.event.condition}, 'ems'))) = [];

        if subject == 15
            EEG.event(1:min(find(ismember({EEG.event.hedTag}, 'n/a')))) = [];
        end
        
        EEG.event = renamefields(EEG.event, 'trial_type', 'type');
        [EEG.etc.analysis.design, touch_event_ixs] = pe_build_dmatrix(EEG, bemobil_config);
        EEG.etc.analysis.design.bad_touch_epochs = sort([EEG.etc.analysis.design.slow_rt_spawn_touch_events_ixs, pe_clean_epochs(EEG, touch_event_ixs, bemobil_config)]); % combine noisy epochs with epochs of long reaction times
        touch_event_ixs(EEG.etc.analysis.design.bad_touch_epochs) = [];
        EEG.event = EEG.event(touch_event_ixs);

        %% select ICs to project out of channel data

        [EEG, bemobil_config] = select_ICs_pe(EEG, bemobil_config);

        %% select classes and match the number of epochs in classes

        % make event classes: synchronous and asynchronous
        async_ixs = find(strcmp({EEG.event.normal_or_conflict}, 'conflict'));
        sync_ixs = find(strcmp({EEG.event.normal_or_conflict}, 'normal'));

        % match class size
        sync_ixs = randsample(sync_ixs, size(async_ixs,2));
        EEG.event = EEG.event(union(async_ixs, sync_ixs));

        % targetmarkers to type field
        [EEG.event.type] = EEG.event.normal_or_conflict;

    %         EEGnormal = pop_epoch(EEG, {'normal'}, [0 .5]);
    %         EEGconflict = pop_epoch(EEG, {'conflict'}, [0 .5]);
    %         figure;plot(mean(EEGconflict.data(65,:,:),3));hold on;plot(mean(EEGnormal.data(65,:,:),3));

        % training the classifier! (assuming you have some data loaded as 'EEG')
        [trainloss, model, stats] = bci_train('Data', EEG,...
            'Approach', bemobil_config.lda.approach,...
            'TargetMarkers', bemobil_config.lda.targetmarkers,...
            'EvaluationScheme', {'chron', bemobil_config.lda.evalfolds, bemobil_config.lda.evalmargin},...
            'OptimizationScheme', {'chron', bemobil_config.lda.parafolds, bemobil_config.lda.paramargin});

        disp(['training mis-classification rate: ' num2str(trainloss*100,3) '%'])
        %bci_visualize(model)

        %% calculate results
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

        clear EEG

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

    fname = ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_base_removal-' num2str(bemobil_config.lda.approach{3}{4}(1)) '-'  num2str(bemobil_config.lda.approach{3}{4}(2))];
    save(fullfile(bemobil_config.study_folder, ['lda_results-' fname '.mat']), 'lda_results');

end

%% inspect results

% add downloaded analyses code to the path
addpath(genpath('/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing'));

% Results output folder -> external drive
bemobil_config.study_folder = fullfile('/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error', 'derivatives');

% init
config_processing_pe;
subjects = 1:19;

% load
for t = [0, .5, .7, .8, .9]

    bemobil_config.lda.brain_threshold = t;
    fname = ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_base_removal-' num2str(bemobil_config.lda.approach{3}{4}(1)) '-'  num2str(bemobil_config.lda.approach{3}{4}(2))];
    load(fullfile(bemobil_config.study_folder, ['lda_results-' fname '.mat']));
    
    disp(['tstat: ', num2str(lda_results.ttest.stats.tstat), '; class acc.: ', num2str(mean(lda_results.correct)), '; thresh: ' num2str(t), ' number of dipoles: ', num2str(size(lda_results.dipoles,1))])

end

bemobil_config.lda.brain_threshold = .7;
fname = ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_base_removal-' num2str(bemobil_config.lda.approach{3}{4}(1)) '-'  num2str(bemobil_config.lda.approach{3}{4}(2))];
load(fullfile(bemobil_config.study_folder, ['lda_results-' fname '.mat']));

% plot all dipoles and save for supplements
plot_weighteddipoledensity(lda_results.dipoles)

plot_weighteddipoledensity(lda_results.dipoles,mean(lda_results.weights,2));
plot_weighteddipoledensity(lda_results.dipoles,mean(lda_results.weights(:,1:2),2));

plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,1));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,2));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,3));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,4)); % posterior, precuneus
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,5));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,6));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,7));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,8));
    
%% plot patterns, sample EEG chanlocs have to be loaded (DONE & PLOTTED)

save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/lda/';
mkdir(save_path);
normal;

%ixs = 1:19;
%ixs(7) = [];
%figure;topoplot(mean(normalize(reshape(lda_results.patterns(ixs,i), 65, 19), 1), 2), locs)

% s8 is bad for window 6

%figure;
%for j = 1:19 % for all subjects, save for supplements
for i = 6 %1:8 
%for j = 1:19 % for all subjects, save for supplements
    
    % plot
    mean_pattern = lda_results.patterns(:,i);
    mean_pattern = reshape(mean_pattern, [65,19]);
    mean_pattern(:,7) = [];
    %mean_pattern = mean(mean_pattern,2);
    %figure;topoplot(mean_pattern(:,j),locs);
    figure;topoplot(mean(normalize(mean_pattern, 1), 2), locs);
    
    %subplot(4,6,i);
    %mean_pattern = mean_pattern(:,ixs);
    %topoplot(mean_pattern,locs);
    %topoplot(mean(normalize(mean_pattern, 1), 2), locs);
        
    % plot with normalization
    %figure;topoplot(mean(normalize(lda_results.patterns(:,i), 65, 19), 1), 2), locs);
    
    % save with window name
    print(gcf, [save_path '0' num2str(brain_prob)' fname '_lda_pattern_win_' num2str(i) '.eps'], '-depsc');     
    % close figure
    close(gcf);
end
%end

%% plot weighted dipoles at all timepoints (DONE & PLOTTED)

save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/lda/';
mkdir(save_path);
normal;

for i = 1:8
    
    % plot weighted dipoles
    plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,i));
    % save with window name
    print(gcf, [save_path '0' num2str(brain_prob)' fname '_win_' num2str(i) '.eps'], '-depsc');     
    % close figure
    close(gcf);
end

%% plot control signal ERP style: mark 2 classes (DONE & PLOTTED)
    
save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/lda/';
mkdir(save_path);
normal;
for i = 1:8 
    
    % prepare plot
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 300 200]);
    title('Control Signal 6th Window');

    % plot condition 1
    colors = brewermap(5, 'Spectral');
    colors1 = colors(2, :);
    sync = squeeze(lda_results.control_signal(:,1,i,:));
    ploterp_lg(sync, [], [], 50, 1, 'norm. \muV', '', '', colors1, '-');
    hold on

    % plot condition 2
    colors2 = colors(5, :);
    async = squeeze(lda_results.control_signal(:,2,i,:));
    ploterp_lg(async, [], [], 50, 1, 'norm. \muV', '', '', colors2, '-.');
    
    % save with window name
    print(gcf, [save_path '0' num2str(brain_prob)' fname 'lda_control_signal_win_' num2str(i) '.eps'], '-depsc');     
    % close figure
    close(gcf);
end

