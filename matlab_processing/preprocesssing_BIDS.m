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

%% event and baseline epoching, clean epoch indices (autorej,3 function), motion data (DONE)

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

for subject = subjects
    
    %% load BIDS (with AMICA results) set
    modality = 'eeg';
    EEG = pop_loadset(fullfile(bemobil_config.BIDS_folder, ['sub-', sprintf('%03d', subject)], modality, ...
        ['sub-', sprintf('%03d', subject), '_task-', bemobil_config.task, '_', modality, '.set']));
    
    %% clean up, get touch events
    
    EEG = pe_remove_training_block(EEG);
    EEG.event(find(strcmp({EEG.event.condition}, 'ems'))) = [];

    if subject == 15
        EEG.event(1:min(find(ismember({EEG.event.hedTag}, 'n/a')))) = [];
    end
    
    EEG.event = renamefields(EEG.event, 'trial_type', 'type');
    touch_event_ixs = find(strcmp({EEG.event.type}, bemobil_config.epoching.event_epoching_event));
    
    %% measure: motion, Velocity at time points before events of interest
    
    modality = 'motion';
    motion = pop_loadset(fullfile(bemobil_config.BIDS_folder, ['sub-', sprintf('%03d', subject)], modality, ...
        ['sub-', sprintf('%03d', subject), '_task-', bemobil_config.task, '_', modality, '.set']));
    motion.event = renamefields(EEG.event, 'trial_type', 'type');
    motion.event = EEG.event(touch_event_ixs);
    motion = pop_epoch( motion, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');    
    
    % save x,y,z of hand
    EEG.etc.analysis.motion.x = motion.data(7,:,:);
    EEG.etc.analysis.motion.y = motion.data(8,:,:);
    EEG.etc.analysis.motion.z = motion.data(9,:,:);
    
    % save 3D magnitude of velocity and acceleration
    EEG.etc.analysis.motion.mag_vel = squeeze(sqrt(motion.data(7,:,:).^2 +...
            motion.data(8,:,:).^2 +...
            motion.data(9,:,:).^2));
    EEG.etc.analysis.motion.mag_acc = squeeze(sqrt(motion.data(13,:,:).^2 +...
            motion.data(14,:,:).^2 +...
            motion.data(15,:,:).^2));
    
    %% make design matrix, find noisy epochs, detect movements
    
    [EEG.etc.analysis.design, touch_event_ixs] = pe_build_dmatrix(EEG, bemobil_config);
    
    if isfield(EEG.etc.analysis.design, 'bad_trial_order_ixs')
        EEG.etc.analysis.design.bad_touch_epochs = sort(unique([EEG.etc.analysis.design.slow_rt_spawn_touch_events_ixs, ...
            EEG.etc.analysis.design.bad_trial_order_ixs,...
            pe_clean_epochs(EEG, touch_event_ixs, bemobil_config)])); % combine noisy epochs with epochs of long reaction times and bad trial order
    else
        EEG.etc.analysis.design.bad_touch_epochs = sort(unique([EEG.etc.analysis.design.slow_rt_spawn_touch_events_ixs, ...
            pe_clean_epochs(EEG, touch_event_ixs, bemobil_config)])); % combine noisy epochs with epochs of long reaction times
    end

    event_sample_ix = abs(bemobil_config.epoching.event_epochs_boundaries(1)) * EEG.srate; % epoched [-3 2] seconds = 1250 samples
    thresh = .05;
    EEG.etc.analysis.design.movements = pe_get_motion_onset_single_trials(EEG, event_sample_ix, thresh, subject, bemobil_config);
 
    %% measure: filtered ERPs
    
    ERP = EEG;
    ERP.event = EEG.event(touch_event_ixs);
    ERP.event(EEG.etc.analysis.design.bad_touch_epochs) = [];
    ERP = pop_eegfiltnew(ERP, bemobil_config.filter_plot_low, bemobil_config.filter_plot_high);
    ERP = pop_epoch( ERP, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    EEG.etc.analysis.filtered_erp.chan = ERP.data; %(bemobil_config.channels_of_int,:,:); % save space saving only channels of int FZ, PZ, FCz
    EEG.etc.analysis.filtered_erp.comp = ERP.icaact;
   
    %% measure: ERSP
    
    ERSP_event = EEG;
    ERSP_event.event = EEG.event(touch_event_ixs); % touches
    
    ERSP_base = EEG;
    spawn_event_ixs = find(strcmp({EEG.event.type}, bemobil_config.epoching.base_epoching_event));
    ERSP_base.event = EEG.event(spawn_event_ixs); % spawns
    
    % timewarp
    if bemobil_config.timewarp
        tw = zeros(size(ERSP_event.event,2),3);
        tw(:,1) = (ERSP_event.etc.analysis.design.spawn_event_sample - event_sample_ix) / ERSP_event.srate;
        tw(:,2) = (ERSP_event.etc.analysis.design.movements.reach_onset_sample - event_sample_ix) / ERSP_event.srate;
        tw = tw*1000;
        
        ERSP_event.event(EEG.etc.analysis.design.bad_touch_epochs) = [];
        ERSP_base.event(EEG.etc.analysis.design.bad_touch_epochs) = [];
        tw(EEG.etc.analysis.design.bad_touch_epochs,:) = [];
    end
    
    ERSP_event = pop_epoch( ERSP_event, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    ERSP_base = pop_epoch( ERSP_base, bemobil_config.epoching.base_epoching_event, bemobil_config.epoching.base_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    
    for comp = 1:size(ERSP_event.icaact,1)
        tic
        
        % event ersp
        if bemobil_config.timewarp
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
                'verbose','off',...
                'timewarp',tw,...
                'timewarpms', [bemobil_config.timewarp_anchors(1) bemobil_config.timewarp_anchors(1) 0]);
        else
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
        end
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

%         % test grand mean ersp: everything looks good, 02.03.2021
%         ev = squeezemean(EEG.etc.analysis.ersp.tf_event_raw_power,4);
%         base = squeezemean(EEG.etc.analysis.ersp.tf_base_raw_power,3);
%         ev_db = 10.*log10(squeeze(ev(comp,:,:))./base(comp,:)');
%         figure;imagesc(EEG.etc.analysis.ersp.tf_event_times, EEG.etc.analysis.ersp.tf_event_freqs, ev_db, [-1 1]);axis xy;cbar;
  
        toc
    end
    
    % save times and freqs
    EEG.etc.analysis.ersp.tf_event_times = ersp_times_ev;
    EEG.etc.analysis.ersp.tf_event_freqs = ersp_freqs_ev;
    clear tw
    
    %% epoch for EEGLAB study structure and save
    
    out_folder = fullfile(bemobil_config.study_folder, 'data');
    if ~exist(out_folder, 'dir')
        mkdir(out_folder);
    end
    
    % save externally to speed up EEGLAB processing
    filtered_erp = EEG.etc.analysis.filtered_erp;
    save(fullfile(out_folder, ['sub-', sprintf('%03d', subject), '_filtered_erp.mat']), 'filtered_erp', '-v7.3');
    EEG.etc.analysis.filtered_erp = [];
    
    ersp = EEG.etc.analysis.ersp;
    save(fullfile(out_folder, ['sub-', sprintf('%03d', subject), '_ersp.mat']), 'ersp', '-v7.3');
    EEG.etc.analysis.ersp = [];
    
    motion = EEG.etc.analysis.motion;
    save(fullfile(out_folder, ['sub-', sprintf('%03d', subject), '_motion.mat']), 'motion', '-v7.3');
    EEG.etc.analysis.motion = [];
    
    EEG.event = EEG.event(touch_event_ixs);
    epochs = pop_epoch( EEG, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    pop_saveset(epochs, 'filename', ['sub-', sprintf('%03d', subject), '_epochs_', bemobil_config.epoching.event_epoching_event{1} '.set'], 'filepath', out_folder); % epochs for EEGLAB Study
    
end
