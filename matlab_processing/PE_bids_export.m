%% declare settings and folders 

clear all;

cfg.subjects = 2:20;

cfg.study_folder = '/Volumes/Data_and_photos/work/studies/Prediction_Error/data/';
cfg.filename_prefix = 's';

%cfg.filenames =  {'s2_PredError_block_TestEMS.xdf', 's2_PredError_block_TestVibro.xdf', 's2_PredError_block_TestVisual.xdf'};
%cfg.filenames = {'PredError_block_TestVibro' 'PredError_block_TestVisual'};
%cfg.filenames_s3 = {'PredError_block_TestVibro_erste100' 'PredError_block_TestVibro_101bis300' 'PredError_block_TestVisual'};

% be as specific as possible (uses regexp)
cfg.unprocessed_data_streams = {'brainvision_rda_bpn-c012'};
cfg.event_streams = {'unity_markers_prederror_BPN-C043'};
cfg.rigidbody_streams = {'rigid_handr_BPN-C043','rigid_head_BPN-C043'};

% enter channels that you did not use at all (e.g. with the MoBI 160 chan layout, only 157 chans are used):
cfg.channels_to_remove = [];

% enter EOG channel names here:
cfg.eog_channels  = {''};
cfg.ref_channel = 'FCz';

% leave this empty if you have standard channel names that should use standard locations:
% the standard .elc file can be found at "M:/BrainVision Stuff/Cap Layouts/standard_MoBI_channel_locations" and must
% be copied into every subject's data folder (where the .xdf files are)
cfg.channel_locations_filename = '';


% general foldernames and filenames
cfg.raw_data_folder = '0_raw-data/';
cfg.mobilab_data_folder = '1_mobilab-data/';
cfg.raw_EEGLAB_data_folder = '2_basic-EEGLAB/';

cfg.merged_filename = 'merged.set';
cfg.preprocessed_filename = 'preprocessed.set';
cfg.interpolated_avRef_filename = 'interpolated_avRef.set';
cfg.filtered_filename = 'filtered.set';

% preprocessing
cfg.mocap_lowpass = 6;
cfg.rigidbody_derivatives = 2;
cfg.resample_freq = 250;

%% set to mobids export / conversion

if ~exist('ALLEEG','var')
	eeglab;
	runmobilab;
end

% read data, align modalities and merge to one file
for subject = cfg.subjects(10)
    
    % get all xdf filename in subject folder
    filenames = dir(fullfile(cfg.study_folder, cfg.raw_data_folder, [cfg.filename_prefix num2str(subject)]));
    xdf_ix = find(contains({filenames.name}, 'xdf'));
    filenames = {filenames(xdf_ix).name};
    % remove prefix and suffix to keep compatible with below function...
    for i = 1:numel(filenames)
        % index of first underscore: participant id _ filename
        u_ix = find(ismember(filenames{i}, '_'), 1, 'first');
        cfg.filenames{i} = filenames{i}(u_ix+1:end-4);
    end
    
	% load xdf files and process them with mobilab, export to eeglab, split MoBI and merge all conditions for EEG
	[ALLEEG, EEG_merged, CURRENTSET] = bemobil_process_all_mobilab(subject, cfg, ALLEEG, CURRENTSET, mobilab, 1);

    % merge
    % load all _MoBI sets
    for f = files
        EEG = pop_loadset(fullfile([cfg.filename_prefix num2str(subject) '_'...
                f{1} '_MoBI.set'],...
                [cfg.study_folder ...
                cfg.raw_EEGLAB_data_folder ...
                cfg.filename_prefix num2str(subject)]));
        [ALLEEG EEG index] = eeg_store(ALLEEG, EEG);
    end
    EEG = pop_mergeset(ALLEEG, 1:size(ALLEEG,2));

end
