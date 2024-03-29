% run BCI using Matlab2014a

% add downloaded analyses code to the path
addpath(genpath('/Users/lukasgehrke/Documents/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing'));
% TODO add to path bemobil_pipeline repository download folder
% TODO add to path custom scripts repository Lukas Gehrke folder

% add path BCILAB
addpath('/Volumes/work/studies/Prediction_Error/BCILAB');

% BIDS data download folder
bemobil_config.BIDS_folder = '/Volumes/work/studies/Prediction_Error/data/DFA/';
% Results output folder -> external drive
bemobil_config.study_folder = fullfile('/Volumes/work/studies/Prediction_Error/data', 'derivatives');

% init
config_processing_pe;
subjects = 1:19;

%% Run LDA

bcilab;
%%
threshs = .7 %[0, .5, .7, .8, .9];
for t = threshs
    
    bemobil_config.lda.brain_threshold = t;
    
    patterns_t = [];
    weights_t = [];
    dipoles_t = [];

    modality = 'eeg';

    for subject = subjects

        %% load BIDS (with AMICA results) set

%         EEG = pop_loadset(fullfile(bemobil_config.BIDS_folder, ['sub-', sprintf('%03d', subject)], modality,...
%             ['sub-', sprintf('%03d', subject), '_task-', bemobil_config.task, '_', modality '.set']));

        EEG = pop_loadset(fullfile(bemobil_config.BIDS_folder, ['sub-', sprintf('%03d', subject+1) '.set']));

        %% make design matrix, exclude training trials and EMS condition, clean up, find bad epochs

        EEG = pe_remove_training_block(EEG);
        EEG.event(find(strcmp({EEG.event.condition}, 'ems'))) = [];

        if subject == 15
            EEG.event(1:min(find(ismember({EEG.event.hedTag}, 'n/a')))) = [];
        end
        
        
        EEG.event = renamefield(EEG.event, 'trial_type', 'type');
        %         EEG.event = renamefields(EEG.event, 'trial_type', 'type');
        
        
        [EEG.etc.analysis.design, touch_event_ixs] = pe_build_dmatrix(EEG, bemobil_config);
        EEG.etc.analysis.design.bad_touch_epochs = sort([EEG.etc.analysis.design.slow_rt_spawn_touch_events_ixs, pe_clean_epochs(EEG, touch_event_ixs, bemobil_config)]); % combine noisy epochs with epochs of long reaction times
        touch_event_ixs(EEG.etc.analysis.design.bad_touch_epochs) = [];
        EEG.event = EEG.event(touch_event_ixs);

        %% select ICs to project out of channel data

        [EEG, bemobil_config] = select_ICs_pe(EEG, bemobil_config);

        %% select classes and match the number of epochs in classes

%         % make event classes: synchronous and asynchronous
%         async_ixs = find(strcmp({EEG.event.normal_or_conflict}, 'conflict'));
%         sync_ixs = find(strcmp({EEG.event.normal_or_conflict}, 'normal'));
% 
%         % match class size
%         sync_ixs = randsample(sync_ixs, size(async_ixs,2));
%         EEG.event = EEG.event(union(async_ixs, sync_ixs));

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
%     save(fullfile(bemobil_config.study_folder, ['lda_results-' fname '.mat']), 'lda_results');

end

%% load results

bemobil_config.lda.brain_threshold = .7;
fname = ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_base_removal-' num2str(bemobil_config.lda.approach{3}{4}(1)) '-'  num2str(bemobil_config.lda.approach{3}{4}(2))];
load(fullfile(bemobil_config.study_folder, ['lda_results-' fname '.mat']));

save_path = '/Users/lukasgehrke/Documents/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/lda/';
mkdir(save_path);
normal;

%% plot patterns, sample EEG chanlocs have to be loaded (DONE & PLOTTED)

locs = ALLEEG(1).chanlocs;

for i = 1:8 
    mean_pattern = lda_results.patterns(:,i);
    mean_pattern = reshape(mean_pattern, [65,19]);

    % wrong electrode loc CP2s7?
    mean_pattern(:,7) = [];
    
    figure;topoplot(mean(mean_pattern, 2), locs, ...
        'electrodes', 'on'); cbar;
    
%     figure;topoplot(mean(normalize(mean_pattern, 1), 2), locs, ...
%         'electrodes', 'on'); cbar;
    print(gcf, [save_path fname '_lda_pattern_win_' num2str(i) '.eps'], '-depsc');     
    close(gcf);
end

%% plot grand average pattern (DONE & PLOTTED)
mean_pattern = mean(lda_results.patterns,2);
mean_pattern = reshape(mean_pattern, [65,19]);

% wrong electrode loc CP2s7?
mean_pattern(:,7) = [];

figure;topoplot(mean(mean_pattern, 2), locs, ...
    'electrodes', 'on'); cbar;

% figure;topoplot(mean(normalize(mean_pattern, 1), 2), locs, ...
%     'electrodes', 'on'); cbar;
print(gcf, [save_path fname '_lda_mean_pattern.eps'], '-depsc');     
close(gcf);

%% plot weighted dipoles averaged over all timepoints (DONE & PLOTTED)

plot_weighteddipoledensity(lda_results.dipoles,mean(lda_results.weights,2));

%% plot weighted dipoles at all timepoints (DONE & PLOTTED)

for i = 1:8
    plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,i));
    print(gcf, [save_path fname '_dipdensity_win_' num2str(i) '_50-50-11.eps'], '-depsc');     
    close(gcf);
end

%% plot control signal ERP style: mark 2 classes (DONE & PLOTTED)
    
event_zero = 63;

for i = 1:8 
    
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 600 400]);
    title(['Control Signal ' num2str(i) 'th Window']);

    % plot condition 1
    colors = brewermap(5, 'Spectral');
    colors1 = colors(2, :);
    sync = squeeze(lda_results.control_signal(:,1,i,:));
    
    base_sync = mean(sync(:,event_zero-13:event_zero),2);
    sync = sync - base_sync;
    
    ploterp_lg(sync, [], [], event_zero, 1, 'norm. \muV', '', '', colors1, '-');
    hold on

    % plot condition 2
    colors2 = colors(5, :);
    async = squeeze(lda_results.control_signal(:,2,i,:));
    
    base_async = mean(async(:,event_zero-13:event_zero),2);
    async = async - base_async;
    
    ploterp_lg(async, [], [], event_zero, 1, 'norm. \muV', '', '', colors2, '-.');
    legend({'', 'sync', '', '', 'async'});
    
    print(gcf, [save_path fname '_lda_control_signal_win_' num2str(i) '.eps'], '-depsc');
    close(gcf);
end

