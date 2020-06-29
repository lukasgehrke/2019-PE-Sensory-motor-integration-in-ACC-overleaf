%% declare settings and folders 

clear all;

cfg.subjects = 2:20;

cfg.study_folder = 'P:\Lukas_Gehrke\studies\Prediction_Error\data\';
cfg.filename_prefix = 's';

% be as specific as possible (uses regexp)
cfg.unprocessed_data_streams = {'brainvision_rda_bpn-c012'};
cfg.event_streams = {'unity_markers_prederror_BPN-C043'};
cfg.rigidbody_streams = {'rigid_handr_BPN-C043','rigid_head_BPN-C043'};

% enter channels that you did not use at all (e.g. with the MoBI 160 chan layout, only 157 chans are used):
cfg.channels_to_remove = [];
cfg.rename_channels = {'brainvision_rda_bpn-c012_Fp1' 'Fp1';
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

% enter EOG channel names here:
cfg.eog_channels  = {''};
cfg.ref_channel = 'FCz';

% leave this empty if you have standard channel names that should use standard locations:
% the standard .elc file can be found at "M:\BrainVision Stuff\Cap Layouts\standard_MoBI_channel_locations" and must
% be copied into every subject's data folder (where the .xdf files are)
cfg.channel_locations_filename = '';

% general foldernames and filenames
cfg.raw_data_folder = '0_raw-data\';
cfg.mobilab_data_folder = '1_mobilab-data\';
cfg.raw_EEGLAB_data_folder = '2_basic-EEGLAB\';

cfg.merged_filename = 'merged.set';
cfg.preprocessed_filename = 'preprocessed.set';
cfg.interpolated_avRef_filename = 'interpolated_avRef.set';
cfg.filtered_filename = 'filtered.set';

% preprocessing
cfg.mocap_lowpass = 6;
cfg.rigidbody_derivatives = 2;
cfg.resample_freq = 250;

%% set to mobids export \ conversion

if ~exist('ALLEEG','var')
	eeglab;
	runmobilab;
end

% read data, align modalities and merge to one file
for subject = cfg.subjects(3:end)
    
    %% get all xdf filename in subject folder
    filenames = dir(fullfile(cfg.study_folder, cfg.raw_data_folder, [cfg.filename_prefix num2str(subject)]));
    xdf_ix = find(contains({filenames.name}, 'xdf'));
    filenames = {filenames(xdf_ix).name};
    % remove prefix and suffix to keep compatible with below function...
    for i = 1:numel(filenames)
        % index of first underscore: participant id _ filename
        u_ix = find(ismember(filenames{i}, '_'), 1, 'first');
        cfg.filenames{i} = filenames{i}(u_ix+1:end-4);
    end
    
	%% load xdf files and process them with mobilab, export to eeglab
    % this is taken from Marius Klug's bemobil pipeline bemobil_process_all_mobilab
	bemobil_process_all_mobilab(subject, cfg, ALLEEG, CURRENTSET, mobilab, 0);
    
    % merge
    % load all _MoBI sets
    for fname = cfg.filenames
        EEG = pop_loadset(fullfile([cfg.study_folder ...
                cfg.raw_EEGLAB_data_folder ...
                cfg.filename_prefix num2str(subject)], ...
                [cfg.filename_prefix num2str(subject) '_'...
                fname{1} '_MoBI.set']));
        [ALLEEG EEG index] = eeg_store(ALLEEG, EEG);
    end
    EEG = pop_mergeset(ALLEEG, 1:size(ALLEEG,2));
    
    %% 9. save and clear

    EEG = eeg_checkset(EEG);
    pop_saveset(EEG, 'filename', ['s' num2str(subject) '_full_MoBI'], 'filepath', fullfile([cfg.study_folder ...
                cfg.raw_EEGLAB_data_folder ...
                cfg.filename_prefix num2str(subject)]));
    
    % clear EEG sets
    ALLEEG = pop_delset(ALLEEG, 1:size(ALLEEG,2));
    if size(ALLEEG,2)>1
        for i = 2:size(ALLEEG,2)
            ALLEEG(2) = [];
        end
    end

end

%% TODO add changes related to PE dataset, set to mobids
% This script provides the transformation of raw EEG and motion capture 
% data (XDF, extensible data format) recorded from participants in the 
% invisible maze task. First, raw data is corrected manually for 
% experimenter shortcomings and non-informative events. Second, EEGLAB 
% compatible '.set' files are created and parsed.

% Ultimately, '.set' files are exported to (MoBI) BIDS format with
% information of participant descriptives.

% L. Gehrke - June 2020

%--------------------------------------------------------------------------
% 0. General Information and Directory Management 
%--------------------------------------------------------------------------

% add the path to the bids matlab tools folder 
addpath(genpath('/Users/lukasgehrke/Documents/MATLAB/toolboxes/bids-matlab-tools'));

% directories
% -----------
% participant ID strings
subjects = [2:20];
participantIDArray = {'s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20'};

% path to the .set files 
eegFileFolder         = '/Volumes/Data_and_photos/work/studies/Prediction_Error/data/2_basic-EEGLAB';

% EEG file suffix (participant ID string + suffix = EEG file name)
eegFileSuffix         = '_full_MoBI.set';   

% path to the chanloc files 
%export chanlocs? - they are default chanlocs
% chanloc file suffix (participant ID string + suffix = chanloc file name)
%chanlocFileSuffix         = '_locs.elc';   

% warning : target path will be emptied and overwritten when you run
%           the export function
targetFolder          = '/Volumes/Data_and_photos/work/studies/Prediction_Error/data/BIDS';

% general information for dataset_description.json file
% -----------------------------------------------------
gInfo.Name    = 'Prediction Error';
gInfo.BIDSVersion = '1.4';
gInfo.License = '';
gInfo.Authors = {"Lukas Gehrke, Sezen Akman, Albert Chen, Pedro Lopes, Klaus Gramann"};
gInfo.Acknowledgements = '';
gInfo.HowToAcknowledge = '';
gInfo.ReferencesAndLinks = { "" };
gInfo.DatasetDOI = '';

% Content for README file
% -----------------------
README = sprintf( [ '19 participants were tested in a virtual reality (VR) reach-to-object task.\n'...
    'Participants experienced visual, visual with vibrotractile or visual with vibrotactile and \n'...
    'electrical muscle stimulation (EMS) feedback.\n'...
    'Participants rated their experience on the Immersion and Presence Questionnaire (IPQ) and their workload on the NASA-TLX']);
                

% Content for CHANGES file
% ------------------------
CHANGES = sprintf([ 'Revision history for prediction error dataset\n\n' ...
                    'version 1.0 beta - 29 Jun 2020\n' ...
                    ' - Initial release\n' ]);

% Task information for xxxx-eeg.json file
% ---------------------------------------
tInfo.InstitutionAddress = 'Strasse des 17. Juni, Berlin, Germany';
tInfo.InstitutionName = 'Technische Universitaet zu Berlin';
tInfo.InstitutionalDepartmentName = 'Biopsychology and Neuroergonomics';
tInfo.PowerLineFrequency = 50;
tInfo.ManufacturersModelName = 'n/a';
tInfo.SamplingFrequency = 500;
tInfo.TaskName = 'reach-to-touch prediction error';
tInfo.EOGChannelCount = 0;

%--------------------------------------------------------------------------
% 1. Add Participant Info and Raw File Paths 
%--------------------------------------------------------------------------

% participant information for participants.tsv file
% -------------------------------------------------
tmp = readtable('/Volumes/Data_and_photos/work/studies/Prediction_Error/admin/questionnaires_PE_2018.xlsx', 'Sheet', 'Matlab Import');
varnames = tmp.Properties.VariableNames;
pInfo = table2cell(tmp);
pInfo = [varnames;pInfo];
        
% participant column description for participants.json file
% ---------------------------------------------------------
pInfoDesc.participant_id.Description = 'unique participant identifier';
pInfoDesc.biological_sex.Description = 'biological sex of the participant';
pInfoDesc.biological_sex.Levels.m = 'male';
pInfoDesc.biological_sex.Levels.f = 'female';
pInfoDesc.age.Description = 'age of the participant';
pInfoDesc.age.Units       = 'years';
pInfoDesc.cap_size.Description = 'head circumference and EEG cap sized used';
pInfoDesc.cap_size.Units       = 'centimeter';
pInfoDesc.block_1.Description = 'pseudo permutation of sensory feedback conditions: condition of first block';
pInfoDesc.block_1.Units       = 'Visual = Visual only condition; Visual + Vibro = Simultaneous visual and vibrotactile sensory feedback';
pInfoDesc.block_2.Description = 'pseudo permutation of sensory feedback conditions: condition of second block';
pInfoDesc.block_3.Description = 'some select participants completed a third block with Visual + Vibro + EMS sensory feedback';
pInfoDesc.block_3.Units       = 'Visual + Vibro +EMS = Simultaneous visual, vibrotactile and electrical muscle stimulation sensory feedback';

% file paths (chanlocs are optional, do not specify if not using)
% ---------------------------------------------------------------
for subjectID = 1:numel(participantIDArray)

    % here participants are re-indexed from 1
    subject(subjectID).file     = fullfile(eegFileFolder, participantIDArray{subjectID},...
                                   [participantIDArray{subjectID}, eegFileSuffix]);
    subject(subjectID).chanlocs = fullfile(chanlocFileFolder, participantIDArray{subjectID},... 
                                   [participantIDArray{subjectID}, chanlocFileSuffix]);
    
end

%--------------------------------------------------------------------------
% 2. Process Events (OPTIONAL)
%--------------------------------------------------------------------------  

% By default, function mobids_events_set will parse all markers
% and put all keys into fields of EEG.event struct.
% The new events will be saved in a new .set file, overwritting the old one. 
% Every experiment has different markers, so everyone needs to modify this function.
% Keys and types are assumed to be the same across all participants. 
for subjectID = 1%:numel(participantIDArray)
    
    [keys,types] = PE_set_to_mobids_events([participantIDArray{subjectID} eegFileSuffix], fullfile(eegFileFolder, participantIDArray{subjectID}));

end

trialType = [types' types'];

% add custom event columns
% ------------------------
eInfo = {}; 

% for field 'onset', take 'latency' from the EEG.event struct
% default implementation in bids_export will convert latency to seconds
eInfo{1,1}            = 'onset' ;
eInfo{1,2}            = 'latency';

% for field 'trial_type', take 'trial_type' from the EEG.event struct
eInfo{2,1}            = 'trial_type' ;
eInfo{2,2}            = 'trial_type';

% other fields are kept the same
for keyIndex    = 1:numel(keys)
    eInfo{end + 1 ,1}    = keys{keyIndex};
    eInfo{end,2}         = keys{keyIndex};
end

% event column description for xxx-events.json file (only one such file)
% ---------------------------------------------------------------------
eInfoDesc.onset.Description = 'Event onset';
eInfoDesc.onset.Units = 'second';

eInfoDesc.duration.Description = 'Event duration';
eInfoDesc.duration.Units = 'second';

eInfoDesc.trial_type.Description = 'Type of event (different from EEGLAB convention)';

for typeIndex = 1:numel(types)
    % for now add all types in the marker to the levels
    eInfoDesc.trial_type.Levels.(types{typeIndex}) =types{typeIndex} ;
end


% event information description
% -----------------------------
eInfoDesc.response_time.Description = 'Response time column not used for this data';
eInfoDesc.sample.Description = 'Event sample starting at 0 (Matlab convention starting at 1)';
eInfoDesc.value.Description = 'Value of event (raw makers)';

% custom part: key 'type' is not included
%----------------------------------------
eInfoDesc.G.Description = 'RVD task; test object: global landmark (Lighthouse)';
eInfoDesc.L.Description = 'RVD task; test object: local landmark (path end)';
eInfoDesc.S.Description = 'RVD task; test object: start landmark (path start)';
eInfoDesc.duration.Description = 'time elapsed from exploration start to end';
eInfoDesc.duration.Units = 'second';
eInfoDesc.duration_drawing.Description = 'time elapsedfrom start till end of drawing task';
eInfoDesc.duration_drawing.Units = 'second';
eInfoDesc.duration_outward.Description = 'time elapsed from exploration start to reaching the dead end, or, the start of the pointing task';
eInfoDesc.duration_outward.Units = 'second';
eInfoDesc.duration_return.Description = 'time elapsed from leaving the dead end, or end of pointing task, till return to the start position';
eInfoDesc.duration_return.Units = 'second';
eInfoDesc.duration_walk.Description = 'time elapsed from start till end of rewalking task';
eInfoDesc.duration_walk.Units = 'second';
eInfoDesc.end_pos.Description = 'position of local landmark, or end of path / dead end';
eInfoDesc.end_pos.Units = 'meter';
eInfoDesc.event.Description = 'RVD Task; start, participant response and end (set to 3 seconds following participants response)';
eInfoDesc.global_pos.Description = 'position of Lighthouse landmark';
eInfoDesc.global_pos.Units = 'meter';
eInfoDesc.maze.Description = 'maze type, can be L, Z, U, S';
eInfoDesc.num_head_collision.Description = 'continouos counter of head collisions with the maze walls within maze and run';
eInfoDesc.num_wall_touch.Description = 'continouos counter of hand collisions, touches, with the maze walls';
eInfoDesc.number.Description = 'index of sphere touch during baseline measurement';
eInfoDesc.object.Description = 'RVD task; the tested object id, see eventfield G, L, S';
eInfoDesc.object_location.Description = 'position where participant placed tested object in RVD task';
eInfoDesc.object_location.Units = 'meter';
eInfoDesc.start_pos.Description = 'position of start location';
eInfoDesc.start_pos.Units = 'meter';
eInfoDesc.total_touches.Description = 'num_wall_touch at return to start';
eInfoDesc.trial_run.Description = 'repetition of maze exploration';
eInfoDesc.x.Description = 'x location of hand collision, touch, with maze wall';
eInfoDesc.x.Units = 'meter';
eInfoDesc.y.Description = 'y location of hand collision, touch, with maze wall';
eInfoDesc.y.Units = 'meter';
eInfoDesc.z.Description = 'z location of hand collision, touch, with maze wall';
eInfoDesc.z.Units = 'meter';

%--------------------------------------------------------------------------
% 3. Export in BIDS format
%--------------------------------------------------------------------------

bids_export(subject,                                                ...
    'targetdir', targetFolder,                                      ... 
    'taskName', 'IMT',                                              ...
    'gInfo', gInfo,                                           ...
    'pInfo', pInfo,                                                 ...
    'pInfoDesc', pInfoDesc,                                         ...
    'eInfo',eInfo,                                                  ...
    'eInfoDesc', eInfoDesc,                                         ...
    'README', README,                                               ...
    'trialtype', trialType,                                         ...
    'tInfo', tInfo,                                                 ...
    'mobi', 1)