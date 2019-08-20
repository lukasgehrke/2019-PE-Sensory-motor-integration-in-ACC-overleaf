%% clear all and load params
close all; clear

PE_params;
PE_config;

% special processing subject 3, different filenames
%bemobil_config.filenames = {'PredError_block_TestVibro_erste100' 'PredError_block_TestVibro_101bis300' 'PredError_block_TestVisual'};
%subjects = 3;

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
    
    % change back fnames after processing s3
    if subject == 3
        bemobil_config.filenames = ori_fnames;
    end
	
end

%% plot brain ICs

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

for subject = subjects
	
	disp(['Subject #' num2str(subject)]);
	
	input_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	
	EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.copy_weights_interpolate_avRef_filename], 'filepath', input_filepath);
	[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'study',0);
	
	% clean now, save files and figs
	[ALLEEG, EEG_cleaned, CURRENTSET, ICs_keep, ICs_throw] = bemobil_clean_with_iclabel( EEG , ALLEEG, CURRENTSET, [1], bemobil_config.brain_threshold,...
		[ bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.single_subject_cleaned_ICA_filename],output_filepath);
	
end

% plot dipoles
pop_dipplot( EEG, ICs_keep,...
	'mri','P:\\Marius\\toolboxes\\eeglab14_1_0b\\plugins\\dipfit2.3\\standard_BEM\\standard_mri.mat','normlen','on');

% save fig
savefig(fullfile(output_filepath,'brain_dipoles'))

%% event and baseline epoching, clean epoch indices (autorej function)

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

for subject = subjects
    
	disp(['Subject #' num2str(subject)]);
	
	input_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	
	EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.copy_weights_interpolate_avRef_filename], 'filepath', input_filepath);
	[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'study',0);
    
    % filter for ERP analysis
    EEG = pop_eegfiltnew(EEG, cfg.filter_plot_low, cfg.filter_plot_high);
	
    % epoching
    % parse events from urevent structure 
    EEG = parse_events_PE(EEG);
    touch_ixs = find(strcmp({EEG.event.type}, 'box:touched'));
    if size(touch_ixs,2) ~= 600
        warning('Not exactly 600 events in dataset')
    end
    
    % find indeces of match and mismatch trials in both visual and
    % visual+vibro condition
    box_touched_events = EEG.event(touch_ixs);
    
    match = find(strcmp({box_touched_events.normal_or_conflict}, 'normal'));
    mismatch = find(strcmp({box_touched_events.normal_or_conflict}, 'conflict'));
    visual = find(strcmp({box_touched_events.condition}, 'visual'));
    vibro = find(strcmp({box_touched_events.condition}, 'vibro'));
    
    match_vis = intersect(match, visual);
    match_vibro = intersect(match, vibro);
    mismatch_vis = intersect(mismatch, visual);
    mismatch_vibro = intersect(mismatch, vibro);
    
    % duplicate EEG set for epoching twice (on events and baseline)
    oriEEG = EEG;
        
    % epoch around box:touched event
    EEG = pop_epoch( EEG, cfg.epoching.event_epoching_event, cfg.epoching.event_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    
    % find ~10% noisiest epoch indices by searching for large amplitude
    % fluctuations on the channel level using eeglab auto_rej function
    [~, EEG.etc.epoching.rmepochs] = pop_autorej(EEG, 'maxrej', [2],'nogui','on','eegplot','off');
    
    % remove rmepochs from event indies of match/mismatch visual/vibro
    % trials
    EEG.etc.epoching.match_vis = setdiff(match_vis, EEG.etc.epoching.rmepochs);
    EEG.etc.epoching.match_vibro = setdiff(match_vibro, EEG.etc.epoching.rmepochs);
    EEG.etc.epoching.mismatch_vis = setdiff(mismatch_vis, EEG.etc.epoching.rmepochs);
    EEG.etc.epoching.mismatch_vibro = setdiff(mismatch_vibro, EEG.etc.epoching.rmepochs);
    
    % epochs around box:spawned event, [-2 0], use isiTime (time from starting trial to box:spawned) +500 to +800 as
    % baseline interval
    EEG_base = pop_epoch( oriEEG, cfg.epoching.base_epoching_event, cfg.epoching.base_epochs_boundaries, 'newname',...
        'epochs', 'epochinfo', 'yes');
    % make vector of baseline time window for each epoch
    base_ixs = find(strcmp({oriEEG.event.type}, 'box:spawned'));
    for i = 1:size(base_ixs,2)
        base_ixs_isitime = str2num(oriEEG.event(base_ixs(i)).isiTime);
        EEG.etc.epoching.base_wins(i,:) = [-base_ixs_isitime+.5 -base_ixs_isitime+.8];
    end
    
    % calculate baseline corrected difference ERPs (mismatch - match) all channels/comps
    % first do single trial baseline correction for all trials
    for ep = 1:size(EEG.epoch,2)
        % extract baseline indices
        EEG.etc.analysis.erp.base_inds = ceil((EEG.etc.epoching.base_wins(ep,:) + abs(cfg.epoching.base_epochs_boundaries(1))) * EEG.srate);
        
        EEG.etc.analysis.erp.baseline_chans = squeeze(mean(EEG_base.data(:,EEG.etc.analysis.erp.base_inds(1):EEG.etc.analysis.erp.base_inds(2),ep),2));
        EEG.etc.analysis.erp.baseline_comps = squeeze(mean(EEG_base.icaact(:,EEG.etc.analysis.erp.base_inds(1):EEG.etc.analysis.erp.base_inds(2),ep),2));
        
        EEG.etc.analysis.erp.base_corrected_erps_chans(:,:,ep) = EEG.data(:,:,ep) - EEG.etc.analysis.erp.baseline_chans;
        EEG.etc.analysis.erp.base_corrected_erps_comps(:,:,ep) = EEG.icaact(:,:,ep) - EEG.etc.analysis.erp.baseline_comps;
    end
    
    % now compute difference ERP of baseline corrected single conditions
    EEG.etc.analysis.erp.base_corrected_difference_erps_chans_vis = ...
        EEG.etc.analysis.erp.base_corrected_erps_chans(:,:,EEG.etc.epoching.mismatch_vis) -...
        squeeze(mean(EEG.etc.analysis.erp.base_corrected_erps_chans(:,:,EEG.etc.epoching.match_vis),3));
    EEG.etc.analysis.erp.base_corrected_difference_erps_comps_vis = ...
        EEG.etc.analysis.erp.base_corrected_erps_comps(:,:,EEG.etc.epoching.mismatch_vis) -...
        squeeze(mean(EEG.etc.analysis.erp.base_corrected_erps_comps(:,:,EEG.etc.epoching.match_vis),3));
    
    EEG.etc.analysis.erp.base_corrected_difference_erps_chans_vibro = ...
        EEG.etc.analysis.erp.base_corrected_erps_chans(:,:,EEG.etc.epoching.mismatch_vibro) -...
        squeeze(mean(EEG.etc.analysis.erp.base_corrected_erps_chans(:,:,EEG.etc.epoching.match_vibro),3));
    EEG.etc.analysis.erp.base_corrected_difference_erps_comps_vibro = ...
        EEG.etc.analysis.erp.base_corrected_erps_comps(:,:,EEG.etc.epoching.mismatch_vibro) -...
        squeeze(mean(EEG.etc.analysis.erp.base_corrected_erps_comps(:,:,EEG.etc.epoching.match_vibro),3));

    % save epoched datasets with clean trial indices of the 2x2 study design
    pop_saveset(EEG, 'filename', [bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.epochs_filename] , 'filepath', output_filepath)
end

%% mobi behavior: hand velocity

% generate plots (d trials only) with opaque lines of velocity curve from start to stop threshold
% add an illustration with drawn hand movement and overlayed symbolic velocity curve to better illustrate the parameter

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

for subject = subjects
	
	disp(['Subject #' num2str(subject)]);
	
	input_filepath_mocap = [bemobil_config.study_folder bemobil_config.raw_EEGLAB_data_folder bemobil_config.filename_prefix num2str(subject)];
	input_filepath_eeg = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	
    % first, merge mocap files
    % make sure EEGLAB has no files other than the ones to be merged
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    % load mocap data
    for filename = bemobil_config.filenames
        EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
            filename{1} '_MOCAP.set' ], 'filepath', input_filepath_mocap);
        EEG = pop_resample(EEG, bemobil_config.resample_freq);
        [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'study',0);
    end
    % and merge, saves merged file to single subject analysis folder
    [ALLEEG, mocap, CURRENTSET] = bemobil_merge(ALLEEG,EEG,CURRENTSET,1:length(ALLEEG),...
        [bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.merged_filename_mocap], output_filepath);
    
    % load epoch info from matchig EEG dataset
    EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.epochs_filename], 'filepath', input_filepath_eeg);
	[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'study',0);
    
    % extract velocity at box touched for all the n clean difference trials
    ori_mocap = mocap;
    % parse events
    mocap = parse_events_PE(mocap);
    % epoch mocap around box:touched event
    mocap = pop_epoch( mocap, cfg.epoching.event_epoching_event, cfg.epoching.event_epochs_boundaries_mocap, 'newname',...
        'epochs', 'epochinfo', 'yes');
    
    % calculate magnitude of velocity in 3D
    EEG.etc.analysis.mocap.mag_vel = squeeze(sqrt(mocap.data(7,1,:).^2 + mocap.data(8,1,:).^2 + mocap.data(9,1,:).^2));

    % plot if desired
%     % vel erp of mismatch
%     figure;
%     plot(mean(vel_total(:,[EEG.etc.epoching.match_vibro]),2))    
%     hold on
%     plot(mean(vel_total(:,[EEG.etc.epoching.mismatch_vibro]),2))
%     vline(75);

    % save EEG with velocity info
    pop_saveset(EEG, 'filename', [bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.epochs_filename] , 'filepath', output_filepath)
    
end

%% 1st Level summary statistic ERP (all channels and comps): betas from fit lm at each point of the ERP post event

% -> effect of hand velocity when box is touched on subsequent feedback processing

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

for subject = subjects
	
	disp(['Subject #' num2str(subject)]);
	
	input_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	
	EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.epochs_filename], 'filepath', input_filepath);

    % get data
    vel_vis = EEG.etc.analysis.mocap.mag_vel(EEG.etc.epoching.mismatch_vis);
    vel_vibro = EEG.etc.analysis.mocap.mag_vel(EEG.etc.epoching.mismatch_vibro);
    
    % make predictors
    predictor_vel = [vel_vis; vel_vibro];
    predictor_immersion = [zeros(size(vel_vis,1),1); ones(size(vel_vibro,1),1)];

    % now fit linear model for each condition
    nr_effects = 4;
    effect_erp = zeros(size(EEG.data,1), size(EEG.data,1), nr_effects);
    for chan = 1:size(EEG.data,1)
        h = waitbar(0, ['Now fitting LM for channel: ' EEG.chanlocs(chan).labels]);
        waitbar(chan/size(EEG.data,1),h)
        
        for sample = 1:size(EEG.etc.analysis.erp.base_corrected_difference_erps_chans_vibro,2)

            erp_sample_vis = squeeze(EEG.etc.analysis.erp.base_corrected_difference_erps_chans_vis(chan,sample,:));
            erp_sample_vibro = squeeze(EEG.etc.analysis.erp.base_corrected_difference_erps_chans_vibro(chan,sample,:));
            erp_sample = [erp_sample_vis; erp_sample_vibro];
            
            design = table(erp_sample, predictor_immersion, predictor_vel);

            mdl = fitlm(design, 'erp_sample ~ predictor_immersion * predictor_vel');
            EEG.etc.analysis.results.effect_erp_chans(chan,sample,:) = mdl.Coefficients.Estimate;
        end

        close(h)
    end

    for comp = 1:size(EEG.icaact,1)
        h = waitbar(0, ['Now fitting LM for component: ' num2str(comp)]);
        waitbar(comp/size(EEG.icaact,1),h)
        
        for sample = 1:size(EEG.etc.analysis.erp.base_corrected_difference_erps_comps_vibro,2)

            erp_sample_vis = squeeze(EEG.etc.analysis.erp.base_corrected_difference_erps_comps_vis(comp,sample,:));
            erp_sample_vibro = squeeze(EEG.etc.analysis.erp.base_corrected_difference_erps_comps_vibro(comp,sample,:));
            erp_sample = [erp_sample_vis; erp_sample_vibro];
            
            design = table(erp_sample, predictor_immersion, predictor_vel);

            mdl = fitlm(design, 'erp_sample ~ predictor_immersion * predictor_vel');
            EEG.etc.analysis.results.effect_erp_comps(comp,sample,:) = mdl.Coefficients.Estimate;
        end

        close(h)
    end
    
    % add parameter names
    EEG.etc.analysis.results.parameter_names = mdl.CoefficientNames;
    
%     figure;plot(squeeze(EEG.etc.analysis.results.effect_erp_chans(:,:,3)'))

end

%% build eeglab study and cluster

% cluster using only dipole location
% inspect and select cluster #s to analyze

%% per cluster: 2nd Level inference

% - paired t-test effect of hand velocity between visual vs visual-vibro
% conditions (is the effect of velocity on the error processing ERP moderated by haptic immersion)
%( - unpaired t-test average (across factorial design) effect of hand velocity between median splitted high and low IPQ
% people)


