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
	
    %% BOTH: obtaining clean epoch indices
    % get latencies of boundaries
    boundaries_lats = EEG.urevent(find(strcmp({EEG.urevent.type}, 'boundary'))).latency;
    
    % parse events from urevent structure 
    EEG = parse_events_PE(EEG);
    mocap = parse_events_PE(mocap);
    
    % get event indices
    touch_ixs = find(strcmp({EEG.event.type}, 'box:touched'));
    spawn_ixs = find(strcmp({EEG.event.type}, 'box:spawned'));
    
    mismatch_ixs = find(strcmp({EEG.event.normal_or_conflict}, 'conflict'));
    match_ixs = find(strcmp({EEG.event.normal_or_conflict}, 'normal'));
    
    touch_match_ixs = intersect(touch_ixs, match_ixs);
    suc_touch_match_ix = []; % succeeding trial
    touch_mismatch_ixs = intersect(touch_ixs, mismatch_ixs);
    
    
    spawn_match_ixs = intersect(spawn_ixs, match_ixs);
    spawn_mismatch_ixs = intersect(spawn_ixs, mismatch_ixs);
    
    suc_spawn_match_ix = [];
    
    suc_spawn_mismatch_ix = [];
    suc_touch_mismatch_ix = [];
    
    % check data consistency
    if ~isequal(size(touch_mismatch_ixs), size(spawn_mismatch_ixs))
        error('Number of touch and spawn events is not the same!')
    end
    if size(touch_mismatch_ixs,2) ~= 150
        warning('Not exactly 600 events in dataset!')
    end
    if ~(size(EEG.times,2)==size(mocap.times,2))
        error('mocap and eeg of different length of data!');
    end
    if ~(size(EEG.event,2)==size(mocap.event,2))
        error('mocap and eeg have differing number of events!');
    end
    
    % 1. remove all trials that have a distance greater than 2s between 
    % spawn and touch
    latency_diff = (cell2mat({EEG.event(touch_ixs).latency}) - cell2mat({EEG.event(spawn_ixs).latency})) / EEG.srate;
    bad_tr_ixs = find(latency_diff>2);
    
    % 2. remove all trials with not enough data before
    % box:spawned, this will be used for baseline!
    % check first trial and at boundaries
    if EEG.event(spawn_ixs(1)).latency < abs(bemobil_config.epoching.base_epochs_boundaries(1) * 250)
        % remove first mismatch trial because not enough data before the touch event
        bad_tr_ixs = [bad_tr_ixs, 1];
    end
    for b = boundaries_lats
        dist = cell2mat({EEG.event(spawn_mismatch_ixs).latency}) - b;
        post_b_ix = find(dist>0, 1, 'first'); % index of post boundary spawn event
        ev_dist = EEG.event(spawn_mismatch_ixs(post_b_ix)).latency - b;
        
        if ev_dist < abs(bemobil_config.epoching.base_epochs_boundaries(1) * EEG.srate)
            % no data for baseline, remove post boundary event
            bad_tr_ixs = [bad_tr_ixs, post_b_ix];
        end
    end
    
    % 3. remove all trials without a succeeding match trial and get
    % succeding trial indices
    % get event indeces of all trials succeeding a mismatch trial
    c = 1;
    b_ix = 1;
    for mi = touch_mismatch_ixs
        missing_suc_tr_ix = touch_match_ixs(touch_match_ixs>mi);
        
        if ~isempty(missing_suc_tr_ix)
            missing_suc_tr_ix = missing_suc_tr_ix(1);
        end
        
        if isempty(missing_suc_tr_ix)
            no_suc_ix(b_ix) = find(touch_mismatch_ixs==mi);
            b_ix = b_ix + 1;
            
        % check if succeeding trial actually the next trial or from the next block
        elseif str2double(EEG.event(mi).trial_nr) == (str2double(EEG.event(missing_suc_tr_ix).trial_nr)-1)
            mis_touch_suc_ixs(c) = missing_suc_tr_ix;
            c = c+1;
        
        % succeeding trial from another block
        else
            no_suc_ix(b_ix) = find(touch_mismatch_ixs==mi);
            b_ix = b_ix + 1;
        end
    end
    bad_tr_ixs = [bad_tr_ixs, touch_mismatch_ixs(no_suc_ix)];
    
    % clean up and save indices
    %TO SAVE: find EEG.event indeces of mismatch trials in both visual and visual+vibro condition
    spawn_match_ix = [];
    touch_match_ix = [];
    suc_spawn_match_ix = [];
    suc_touch_match_ix = [];
    
    spawn_mismatch_ix = [];
    touch_mismatch_ix = [];
    suc_spawn_mismatch_ix = [];
    suc_touch_mismatch_ix = [];
    
    touch_mismatch_ixs(missing_suc_tr_ix) = [];
    spawn_mismatch_ixs(missing_suc_tr_ix) = [];
    
    EEG.etc.epoching.mismatch_touch_ixs = touch_mismatch_ixs;
    EEG.etc.epoching.mismatch_spawn_ixs = spawn_mismatch_ixs;
    EEG.etc.epoching.mismatch_touch_succeeding_ixs = mis_touch_suc_ixs;
    EEG.etc.epoching.rm_ixs = bad_tr_ixs;
    EEG.etc.epoching.latency_diff_mismatch = cell2mat({EEG.event(touch_mismatch_ixs).latency}) - cell2mat({EEG.event(spawn_mismatch_ixs).latency});
    EEG.etc.epoching.latency_diff_match = cell2mat({EEG.event(touch_match_ixs).latency}) - cell2mat({EEG.event(spawn_match_ixs).latency});
    
    %% BOTH: epoching
    % duplicate EEG set for epoching
    %EEG: epochs around box:spawned event for baseline
    
    % copy for ERSP later
    EEG_ersp = EEG;
    EEG_ersp_base = EEG;
    
    % copy for mocap of succeeding trials later
    mocap_suc = mocap;
    
    % filter for ERP
    EEG = pop_eegfiltnew(EEG, bemobil_config.filter_plot_low, bemobil_config.filter_plot_high);
    baseEEG = EEG; % for baseline
    
    % overwrite events with clean epoch events
    EEG.event = EEG.event(EEG.etc.epoching.mismatch_touch_ixs); % touches
    mocap.event = EEG.event;    
    baseEEG.event = baseEEG.event(EEG.etc.epoching.mismatch_spawn_ixs); % spawns
    
    % ERPs: both EEG and mocap
    EEG = pop_epoch( EEG, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    mocap = pop_epoch( mocap, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
        
    %EEG: find ~10% noisiest epoch indices by searching for large amplitude
    % fluctuations on the channel level using eeglab auto_rej function
    [~, rmepochs] = pop_autorej(EEG, 'maxrej', [1],'nogui','on','eegplot','off');
    EEG.etc.epoching.rm_ixs = [EEG.etc.epoching.rm_ixs, rmepochs];
    
    %TO SAVE: find indeces of visual and vibro trials after epoch rejection
    EEG.etc.epoching.visual = find(strcmp({EEG.event.condition}, 'visual'));
    EEG.etc.epoching.vibro = find(strcmp({EEG.event.condition}, 'vibro'));
    EEG.etc.epoching.good_visual = setdiff(EEG.etc.epoching.visual, EEG.etc.epoching.rm_ixs);
    EEG.etc.epoching.good_vibro = setdiff(EEG.etc.epoching.vibro, EEG.etc.epoching.rm_ixs);
    % shorter variable for below epoch selection
    vis = EEG.etc.epoching.good_visual;
    vib = EEG.etc.epoching.good_vibro;
    % add further event descriptors
    % direction
    dir = {EEG.event.cube};
    dir = strrep(dir, 'CubeLeft (UnityEngine.GameObject)', 'left');
    dir = strrep(dir, 'CubeRight (UnityEngine.GameObject)', 'right');
    dir = strrep(dir, 'CubeMiddle (UnityEngine.GameObject)', 'middle');
    EEG.etc.epoching.visual_dir = string(dir(vis));
    EEG.etc.epoching.vibro_dir = string(dir(vib));
    % trial number
    EEG.etc.epoching.visual_tr_num = str2double({EEG.event(vis).trial_nr});
    EEG.etc.epoching.vibro_tr_num = str2double({EEG.event(vib).trial_nr});
    % number of match trials preceding mismatch
    EEG.etc.epoching.visual_match_seq = diff([1, EEG.etc.epoching.visual_tr_num]);
    EEG.etc.epoching.vibro_match_seq = diff([1, EEG.etc.epoching.vibro_tr_num]);
    
    %EEG: epochs around box:spawned event for baseline
    baseEEG = pop_epoch( baseEEG, bemobil_config.epoching.base_epoching_event, bemobil_config.epoching.base_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
        
    % get windows of interest
    zero = EEG.srate; % [-1 1] epoch around event
    base_win_samples = zero + (bemobil_config.epoching.base_win(1) * baseEEG.srate):zero+(bemobil_config.epoching.base_win(2) * baseEEG.srate);
    zero = 3*EEG.srate; % [-3 2] epoch around event    
    event_win_samples = zero + (bemobil_config.epoching.event_win(1) * baseEEG.srate):zero+(bemobil_config.epoching.event_win(2) * baseEEG.srate);
    
    %% ERPs comps: Baseline corrected ERPs of mismatch trials
    
    % baseline comps
    EEG.etc.analysis.erp.baseline.visual.comps = mean(baseEEG.icaact(:, base_win_samples, vis), 2);
    EEG.etc.analysis.erp.baseline.vibro.comps = mean(baseEEG.icaact(:, base_win_samples, vib), 2);
    
    % baseline corrected ERPs
    EEG.etc.analysis.erp.base_corrected.visual.comps(:,:,:) = EEG.icaact(:,event_win_samples,vis) - EEG.etc.analysis.erp.baseline.visual.comps;
    EEG.etc.analysis.erp.base_corrected.vibro.comps(:,:,:) = EEG.icaact(:,event_win_samples,vib) - EEG.etc.analysis.erp.baseline.vibro.comps;
    
    % non-baseline corrected ERPs
    EEG.etc.analysis.erp.non_corrected.visual.comps(:,:,:) = EEG.icaact(:,event_win_samples,vis);
    EEG.etc.analysis.erp.non_corrected.vibro.comps(:,:,:) = EEG.icaact(:,event_win_samples,vib);
    
%     %% ERPs chans 
%     
%     % for channel ERPs project out eye component activation time courses
%     eyes = find(EEG.etc.ic_classification.ICLabel.classifications(:,3) > bemobil_config.eye_threshold);
%     EEG_erp_chan = pop_subcomp(EEG, eyes);
%     baseEEG_erp_chan = pop_subcomp(baseEEG, eyes);
%     
%     % baseline chans
%     EEG.etc.analysis.erp.baseline.visual.chans = mean(baseEEG_erp_chan.data(:, base_win_samples, vis), 2);
%     EEG.etc.analysis.erp.baseline.vibro.chans = mean(baseEEG_erp_chan.data(:, base_win_samples, vib), 2);
%     
%     % base corrected ERPs
%     EEG.etc.analysis.erp.base_corrected.visual.chans(:,:,:) = EEG_erp_chan.data(:,event_win_samples,vis) - EEG.etc.analysis.erp.baseline.visual.chans;
%     EEG.etc.analysis.erp.base_corrected.vibro.chans(:,:,:) = EEG_erp_chan.data(:,event_win_samples,vib) - EEG.etc.analysis.erp.baseline.vibro.chans;
%     
%     % non-baseline corrected ERPs
%     EEG.etc.analysis.erp.non_corrected.visual.chans(:,:,:) = EEG_erp_chan.data(:,event_win_samples,vis);
%     EEG.etc.analysis.erp.non_corrected.vibro.chans(:,:,:) = EEG_erp_chan.data(:,event_win_samples,vib);
%     
%     % ERSP comps and chans
%     
%     copy over events from EEG set
%     EEG_ersp.event = EEG.event;
%     epoching
%     EEG_ersp = pop_epoch( EEG_ersp, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
%         'epochs', 'epochinfo', 'yes');
%     baseEEG_ersp = pop_epoch( EEG_ersp_base, bemobil_config.epoching.base_epoching_event, bemobil_config.epoching.base_epochs_boundaries, 'newname',...
%         'epochs', 'epochinfo', 'yes');
%     
%     1. calculate newtimef for both baseline and event epochs
%     event_win
%     zero = 3*EEG.srate; % [-3 2] epoch around event
%     ersp_win = (zero-EEG.srate):zero+2*EEG.srate; % [-1 2] around event
%     ersp_win(end) = [];
%     
%     settings
%     fft_options = struct();
%     fft_options.cycles = [3 0.5];
%     fft_options.freqrange = [3 80];
%     fft_options.freqscale = 'log';
%     fft_options.n_freqs = 60;
%     fft_options.timesout = 120;
%                
%     % do the timefreq analysis for ERSP
%     data_for_ersp = EEG_ersp.icaact(:,ersp_win,:);
%     data_for_base = baseEEG_ersp.icaact;
%     for comp = 1:size(data_for_ersp,1)
%         tic
%         [~,~,~,ersp_times,ersp_freqs,~,~,tfdata] = newtimef(data_for_ersp(comp,:,:),...
%             size(data_for_ersp,2),...
%             [-1 2]*1000,...
%             EEG.srate,...
%             'cycles',fft_options.cycles,...
%             'freqs',fft_options.freqrange,...
%             'freqscale',fft_options.freqscale,...
%             'baseline',[NaN],... % no baseline, since that is only a subtraction of the freq values, we do it manually
%             'nfreqs',fft_options.n_freqs,...
%             'timesout',fft_options.timesout,...
%             'plotersp','off',...
%             'plotitc','off',...
%             'verbose','off');
%                     
%         % select relevant data
%         event_win_times = bemobil_config.epoching.event_win * 1000;
%         tfdata = abs(tfdata); % discard phase
%         tfdata_all = tfdata;
%         tfdata = tfdata(:,find(ersp_times>event_win_times(1),1,'first'):find(ersp_times<=event_win_times(2),1,'last'),:);
%         times_win = ersp_times(:,find(ersp_times>event_win_times(1),1,'first'):find(ersp_times<=event_win_times(2),1,'last'));
%         
%         % 2. make mean baseline vector across all samples in baseline epoch (ensemble baseline)
%         % power spectrum of baseline segment
%         [~,~,~,base_times,freqs,~,~,basedata] = newtimef(data_for_base(comp,:,:),...
%             size(data_for_base,2),...
%             [-1 1]*1000,...
%             EEG.srate,...
%             'cycles',fft_options.cycles,...
%             'freqs',fft_options.freqrange,...
%             'freqscale',fft_options.freqscale,...
%             'baseline',[NaN],... % no baseline, since that is only a subtraction of the freq values, we do it manually
%             'nfreqs',fft_options.n_freqs,...
%             'timesout',fft_options.timesout,...
%             'plotersp','off',...
%             'plotitc','off',...
%             'verbose','off');
%         
%         % average across basewin
%         basedata = abs(basedata); % discard phase
%         base_win_times = bemobil_config.epoching.base_win * 1000;
%         ersp_base_power_raw = basedata(:,find(base_times>base_win_times(1),1,'first'):find(base_times<=base_win_times(2),1,'last'),:);
%         
%         % save full data
%         EEG.etc.analysis.ersp.tfdata.visual.comp(comp,:,:,:) = tfdata_all(:,:,vis);
%         EEG.etc.analysis.ersp.tfdata.vibro.comp(comp,:,:,:) = tfdata_all(:,:,vib);
%         
%         % divisive baseline correction with ensemble baseline
%         EEG.etc.analysis.ersp.baseline.visual.comp(comp,:) = mean(mean(ersp_base_power_raw(:,:,vis),3),2);
%         EEG.etc.analysis.ersp.baseline.vibro.comp(comp,:) = mean(mean(ersp_base_power_raw(:,:,vib),3),2);
%         
%         % apply divisive baseline, convert to dB and save
%         EEG.etc.analysis.ersp.base_corrected_dB.visual.comp(comp,:,:,:) = 10.*log10(tfdata(:,:,vis) ./ EEG.etc.analysis.ersp.baseline.visual.comp(comp,:)');
%         EEG.etc.analysis.ersp.base_corrected_dB.vibro.comp(comp,:,:,:) = 10.*log10(tfdata(:,:,vib) ./ EEG.etc.analysis.ersp.baseline.vibro.comp(comp,:)');
%         
% %         figure;imagesclogy(times_win, ersp_freqs, base_corrected_ersp_dB_win, max(abs(base_corrected_ersp_dB_win(:)))/2 * [-1 1]);axis xy;cbar;
%         toc 
%     end
% 
%     now do timefreq for chans FZ = 5 and PZ = 25 and FCz = 65
%     data_for_ersp = EEG_ersp.data(:,ersp_win,:);
%     data_for_base = baseEEG_ersp.data;
%     for chan = [5, 25, 65]
%         tic
%         [~,~,~,ersp_times,ersp_freqs,~,~,tfdata] = newtimef(data_for_ersp(chan,:,:),...
%             size(data_for_ersp,2),...
%             [-1 2]*1000,...
%             EEG.srate,...
%             'cycles',fft_options.cycles,...
%             'freqs',fft_options.freqrange,...
%             'freqscale',fft_options.freqscale,...
%             'baseline',[NaN],... % no baseline, since that is only a subtraction of the freq values, we do it manually
%             'nfreqs',fft_options.n_freqs,...
%             'timesout',fft_options.timesout,...
%             'plotersp','off',...
%             'plotitc','off',...
%             'verbose','off');
%                     
%         select relevant data
%         event_win_times = bemobil_config.epoching.event_win * 1000;
%         tfdata = abs(tfdata); % discard phase
%         tfdata_all = tfdata;
%         tfdata = tfdata(:,find(ersp_times>event_win_times(1),1,'first'):find(ersp_times<=event_win_times(2),1,'last'),:);
%         times_win = ersp_times(:,find(ersp_times>event_win_times(1),1,'first'):find(ersp_times<=event_win_times(2),1,'last'));
%         
%         2. make mean baseline vector across all samples in baseline epoch (ensemble baseline)
%         power spectrum of baseline segment
%         [~,~,~,base_times,freqs,~,~,basedata] = newtimef(data_for_base(chan,:,:),...
%             size(data_for_base,2),...
%             [-1 1]*1000,...
%             EEG.srate,...
%             'cycles',fft_options.cycles,...
%             'freqs',fft_options.freqrange,...
%             'freqscale',fft_options.freqscale,...
%             'baseline',[NaN],... % no baseline, since that is only a subtraction of the freq values, we do it manually
%             'nfreqs',fft_options.n_freqs,...
%             'timesout',fft_options.timesout,...
%             'plotersp','off',...
%             'plotitc','off',...
%             'verbose','off');
%         
%         average across basewin
%         basedata = abs(basedata); % discard phase
%         base_win_times = bemobil_config.epoching.base_win * 1000;
%         ersp_base_power_raw = basedata(:,find(base_times>base_win_times(1),1,'first'):find(base_times<=base_win_times(2),1,'last'),:);
%         
%         save full data
%         EEG.etc.analysis.ersp.tfdata.visual.channel(chan,:,:,:) = tfdata_all(:,:,vis);
%         EEG.etc.analysis.ersp.tfdata.vibro.channel(chan,:,:,:) = tfdata_all(:,:,vib);
%         
%         divisive baseline correction with ensemble baseline
%         EEG.etc.analysis.ersp.baseline.visual.channel(chan,:) = mean(mean(ersp_base_power_raw(:,:,vis),3),2);
%         EEG.etc.analysis.ersp.baseline.vibro.channel(chan,:) = mean(mean(ersp_base_power_raw(:,:,vib),3),2);
%         
%         apply divisive baseline, convert to dB and save
%         EEG.etc.analysis.ersp.base_corrected_dB.visual.channel(chan,:,:,:) = 10.*log10(tfdata(:,:,vis) ./ EEG.etc.analysis.ersp.baseline.visual.channel(chan,:)');
%         EEG.etc.analysis.ersp.base_corrected_dB.vibro.channel(chan,:,:,:) = 10.*log10(tfdata(:,:,vib) ./ EEG.etc.analysis.ersp.baseline.vibro.channel(chan,:)');
%         
%         figure;imagesclogy(times_win, ersp_freqs, base_corrected_ersp_dB_win, max(abs(base_corrected_ersp_dB_win(:)))/2 * [-1 1]);axis xy;cbar;
%         toc 
%     end
%     
%     save freqs and times
%     EEG.etc.analysis.ersp.times = times_win;
%     EEG.etc.analysis.ersp.times_all = ersp_times;
%     EEG.etc.analysis.ersp.freqs = ersp_freqs;
    
    %% FC comps, sliding window correlation between 2 sources, do later on second level
    % otherwise results matrix gets huge if correlating every comp with
    % every comp
    
    %% MOCAP: Velocity at time points before events of interest
    
    % save (x,y) coordinates for each vis and vib and their succeeding
    % trials
    hand_chans_ix = 1:3;
    EEG.etc.analysis.mocap.visual.x = squeeze(mocap.data(hand_chans_ix(1),:,vis));
    EEG.etc.analysis.mocap.visual.y = squeeze(mocap.data(hand_chans_ix(2),:,vis));
    EEG.etc.analysis.mocap.visual.z = squeeze(mocap.data(hand_chans_ix(3),:,vis));
    EEG.etc.analysis.mocap.vibro.x = squeeze(mocap.data(hand_chans_ix(1),:,vib));
    EEG.etc.analysis.mocap.vibro.y = squeeze(mocap.data(hand_chans_ix(2),:,vib));
    EEG.etc.analysis.mocap.vibro.z = squeeze(mocap.data(hand_chans_ix(3),:,vib));
    
    % save 3D magnitude of velocity and acceleration
    EEG.etc.analysis.mocap.visual.mag_vel = squeeze(sqrt(mocap.data(7,:,vis).^2 +...
            mocap.data(8,:,vis).^2 +...
            mocap.data(9,:,vis).^2));
    EEG.etc.analysis.mocap.visual.mag_acc = squeeze(sqrt(mocap.data(13,:,vis).^2 +...
            mocap.data(14,:,vis).^2 +...
            mocap.data(15,:,vis).^2));
    EEG.etc.analysis.mocap.vibro.mag_vel = squeeze(sqrt(mocap.data(7,:,vib).^2 +...
            mocap.data(8,:,vib).^2 +...
            mocap.data(9,:,vib).^2));
    EEG.etc.analysis.mocap.vibro.mag_acc = squeeze(sqrt(mocap.data(13,:,vib).^2 +...
            mocap.data(14,:,vib).^2 +...
            mocap.data(15,:,vib).^2));
        
    % save 3D magnitude of velocity and acceleration of trials succeeding
    % vis and vib trials
    mocap_suc.event = mocap_suc.event(EEG.etc.epoching.mismatch_touch_succeeding_ixs);
    mocap_suc = pop_epoch( mocap_suc, bemobil_config.epoching.base_epoching_event, bemobil_config.epoching.base_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');    
    
    EEG.etc.analysis.mocap.visual_suc.mag_vel = squeeze(sqrt(mocap.data(7,:,vis_suc).^2 +...
            mocap.data(8,:,vis_suc).^2 +...
            mocap.data(9,:,vis_suc).^2));
    EEG.etc.analysis.mocap.vibro_suc.mag_vel = squeeze(sqrt(mocap.data(7,:,vib_suc).^2 +...
            mocap.data(8,:,vib_suc).^2 +...
            mocap.data(9,:,vib_suc).^2));

    %% save epoched datasets with clean trial indices of the 2x2 study design
    pop_saveset(EEG, 'filename', [bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.epochs_filename] , 'filepath', output_filepath);
    
%     % also save results struct only, smaller and hence faster to load
%     res.epoching = EEG.etc.epoching;
%     res.erp = EEG.etc.analysis.erp;
%     res.ersp = EEG.etc.analysis.ersp;
%     res.mocap = EEG.etc.analysis.mocap;
%     
%     save([output_filepath '_res'], 'res');

% test ERSP
% win = [-400 -100];
% base = EEG.etc.analysis.ersp.tfdata.visual(:,:,find(EEG.etc.analysis.ersp.times_all>win(1),1,'first'):find(EEG.etc.analysis.ersp.times_all<=win(2),1,'last'),:);
% base = mean(mean(base,4),3);
% d = squeeze(EEG.etc.analysis.ersp.tfdata.visual(16,:,:,:));
% test = d ./ base(16,:)';
% p = 10.*log10(squeeze(mean(test,3)));
% figure; imagesc(p(:,4:56), [-1 1]);axis xy;cbar;
    
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



