%% 
config_XDF_pe;
subjects = 2:20;
eeglab;

%% set to mobids export \ conversion: prepare data to include both EEG and mocap

% read data, align modalities and merge to one file
for subject = subjects
    
    filenames = dir(fullfile(bemobil_config.study_folder, bemobil_config.raw_data_folder, [bemobil_config.filename_prefix num2str(subject)]));
    xdf_ix = find(contains({filenames.name}, 'xdf'));
    filenames = {filenames(xdf_ix).name};
    % remove prefix and suffix to keep compatible with below function...
    for i = 1:numel(filenames)
        % index of first underscore: participant id _ filename
        u_ix = find(ismember(filenames{i}, '_'), 1, 'first');
        bemobil_config.filenames{i} = filenames{i}(u_ix+1:end-4);
    end
    
    % get mocap data
    for fname = bemobil_config.filenames
        EEG = pop_loadset(fullfile([bemobil_config.study_folder ...
            bemobil_config.raw_EEGLAB_data_folder ...
            bemobil_config.filename_prefix num2str(subject)], ...
            [bemobil_config.filename_prefix num2str(subject) '_'...
            fname{1} '_MoBI.set']));
        [ALLEEG EEG index] = eeg_store(ALLEEG, EEG);
    end
    mocap = pop_resample(pop_mergeset(ALLEEG, 1:size(ALLEEG,2)), bemobil_config.resample_freq);
    
    % get processed EEG data
    EEG = pop_loadset(fullfile([bemobil_config.study_folder ...
        bemobil_config.single_subject_analysis_folder ...
        bemobil_config.filename_prefix num2str(subject)], ...
        [bemobil_config.filename_prefix num2str(subject) '_'...
        bemobil_config.single_subject_cleaned_ICA_filename]));
    
    % hardcoded copy mocap channels to EEG set
    if subject == 7
        EEG.data(66:101,:) = mocap.data(129:end,:);
        [EEG.chanlocs(66:101).labels] = deal(mocap.chanlocs(129:end).labels);
    else
        EEG.data(66:101,:) = mocap.data(65:end,:);
        [EEG.chanlocs(66:101).labels] = deal(mocap.chanlocs(65:end).labels);
    end
    [EEG.chanlocs(66:101).type] = deal('MOCAP');
    [EEG.chanlocs(66:101).ref] = deal('');
    
    %% save

    EEG = eeg_checkset(EEG);
    pop_saveset(EEG, 'filename', [bemobil_config.filename_prefix num2str(subject) '_' ...
        bemobil_config.to_BIDS_filename], 'filepath', fullfile([bemobil_config.study_folder ...
        bemobil_config.single_subject_analysis_folder ...
        bemobil_config.filename_prefix num2str(subject)]));
    
    % clear EEG sets
    %ALLEEG = pop_delset(ALLEEG, 1:size(ALLEEG,2));
%     if size(ALLEEG,2)>1
%         for i = 2:size(ALLEEG,2)
%             ALLEEG(2) = [];
%         end
%     end

    ALLEEG = [];
    bemobil_config.filenames = [];

end

%% load to_bids file, kick out the EEG data and channels and save as mocap in single_subject_analysis_folder
for subject = subjects
    
    TO_BIDS = pop_loadset(fullfile([bemobil_config.study_folder ...
        bemobil_config.single_subject_analysis_folder ...
        bemobil_config.filename_prefix num2str(subject)], ...
        [bemobil_config.filename_prefix num2str(subject) '_'...
        bemobil_config.to_BIDS_filename]));
    
    TO_BIDS.data(1:65,:) = [];
    TO_BIDS.chanlocs(1:65) = [];
    TO_BIDS.icaact = [];
    TO_BIDS.icawinv = [];
    TO_BIDS.icaspehere = [];
    TO_BIDS.icaweights = [];
    TO_BIDS.icachansind = [];
    TO_BIDS.ref = '';
    
    TO_BIDS = eeg_checkset(TO_BIDS);
    pop_saveset(TO_BIDS, 'filename', [bemobil_config.filename_prefix num2str(subject) '_' ...
        bemobil_config.mocap_BIDS_filename], 'filepath', fullfile([bemobil_config.study_folder ...
        bemobil_config.single_subject_analysis_folder ...
        bemobil_config.filename_prefix num2str(subject)]));
    
end

%% set to mobids definitions and export
% This script provides the transformation of raw EEG and motion capture 
% data (XDF, extensible data format) recorded from participants in the 
% predicition error task. 

% L. Gehrke - February 2021

%--------------------------------------------------------------------------
% 0. General Information and Directory Management 
%--------------------------------------------------------------------------

% add the path to the bids matlab tools folder 
addpath(genpath('P:\Lukas_Gehrke\toolboxes\bids-matlab-tools'));

% directories
% -----------
% participant ID strings
subjects = [2:20];
participantIDArray = {'s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20'};

% path to the .set files 
eegFileFolder         = fullfile(bemobil_config.study_folder, bemobil_config.single_subject_analysis_folder);
    
% EEG file suffix (participant ID string + suffix = EEG file name)
% eegFileSuffix         = ['_', bemobil_config.single_subject_cleaned_ICA_filename];
% eegFileSuffix         = ['_', bemobil_config.to_BIDS_filename];
eegFileSuffix         = ['_', bemobil_config.mocap_BIDS_filename];

% path to the chanloc files 
%export chanlocs? - they are default chanlocs
% chanloc file suffix (participant ID string + suffix = chanloc file name)
%chanlocFileSuffix         = '_locs.elc';   

% warning : target path will be emptied and overwritten when you run
%           the export function
targetFolder          = 'P:\Lukas_Gehrke\studies\Prediction_Error\data\BIDSmotion';

% general information for dataset_description.json file
% -----------------------------------------------------
gInfo.Name    = 'Prediction Error';
gInfo.BIDSVersion = '1.4';
gInfo.License = 'ODbL (https://opendatacommons.org/licenses/odbl/summary/)';
gInfo.Authors = {"Lukas Gehrke", "Sezen Akman", "Albert Chen", "Pedro Lopes", "Klaus Gramann"};
gInfo.Acknowledgements = 'We thank Avinash Singh, Tim Chen and C.-T. Lin from the Univsersity of Sydney (New South Wales, Australia) for their help developing the task.';
gInfo.HowToAcknowledge = 'Please cite: Lukas Gehrke, Sezen Akman, Albert Chen, Pedro Lopes, Klaus Gramann (2021, March 1). Prediction Error: A reach-to-touch Mobile Brain/Body Imaging Dataset. https://doi.org/10.17605/OSF.IO/X7HNM';
gInfo.ReferencesAndLinks = { "Detecting Visuo-Haptic Mismatches in Virtual Reality using the Prediction Error Negativity of Event-Related Brain Potentials. Lukas Gehrke, Sezen Akman, Pedro Lopes, Albert Chen, Avinash Kumar Singh, Hsiang-Ting Chen, Chin-Teng Lin and Klaus Gramann | In Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems (CHI ’19). ACM, New York, NY, USA, Paper 427, 11 pages. DOI: https://doi.org/10.1145/3290605.3300657" };
gInfo.DatasetDOI = 'DOI 10.17605/OSF.IO/X7HNM';

% Content for README file
% -----------------------
README = sprintf( [ '19 participants were tested in a virtual reality (VR) reach-to-object task.\n'...
    'Participants experienced visual, visual with vibrotractile or visual with vibrotactile and \n'...
    'electrical muscle stimulation (EMS) feedback.\n'...
    'Participants rated their experience on the Immersion and Presence Questionnaire (IPQ) and their workload on the NASA-TLX']);
                

% Content for CHANGES file
% ------------------------
CHANGES = sprintf([ 'Revision history for prediction error dataset\n\n' ...
                    'version 1.0 - 1 Mar 2021\n' ...
                    ' - Initial release\n' ]);

% Task information for xxxx-eeg.json file
% ---------------------------------------
tInfo.InstitutionAddress = 'Strasse des 17. Juni, Berlin, Germany';
tInfo.InstitutionName = 'Technische Universitaet zu Berlin';
tInfo.InstitutionalDepartmentName = 'Biopsychology and Neuroergonomics';
tInfo.PowerLineFrequency = 50;
tInfo.ManufacturersModelName = 'n/a';
tInfo.SamplingFrequency = 500;
tInfo.TaskName = 'reach to touch prediction error';
tInfo.EOGChannelCount = 0;

%--------------------------------------------------------------------------
% 1. Add Participant Info and Raw File Paths 
%--------------------------------------------------------------------------

% participant information for participants.tsv file
% -------------------------------------------------
tmp = readtable('P:\Lukas_Gehrke\studies\Prediction_Error\admin\questionnaires_PE_2018.xlsx', 'Sheet', 'Matlab Import');
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
pInfoDesc.block_3.Units       = 'Visual + Vibro + EMS = Simultaneous visual, vibrotactile and electrical muscle stimulation sensory feedback';

% file paths (chanlocs are optional, do not specify if not using)
% ---------------------------------------------------------------
for subjectID = 1:numel(participantIDArray)

    % here participants are re-indexed from 1
    subject(subjectID).file     = fullfile(eegFileFolder, participantIDArray{subjectID},...
                                   [participantIDArray{subjectID}, eegFileSuffix]);
%     subject(subjectID).chanlocs = fullfile(chanlocFileFolder, participantIDArray{subjectID},... 
%                                    [participantIDArray{subjectID}, chanlocFileSuffix]);
    
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
    
    [keys,types] = set_to_BIDS_events_pe([participantIDArray{subjectID} eegFileSuffix], fullfile(eegFileFolder, participantIDArray{subjectID}));

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
eInfoDesc.block.Description = 'start and end of an experimental block';
eInfoDesc.block.Levels.start = 'start';
eInfoDesc.block.Levels.end = 'end';

eInfoDesc.currentBlockNr.Description = 'three experimental blocks per condition';
eInfoDesc.currentBlockNr.Units = 'integer';

eInfoDesc.condition.Description = 'sensory feedback type';
eInfoDesc.condition.Levels.visual = 'visual';
eInfoDesc.condition.Levels.vibro = 'visual + vibrotactile';
eInfoDesc.condition.Levels.ems = 'visual + vibrotactile + electrical muscle stimulation';

eInfoDesc.training.Description = '1 (true) if pre experimental training condition';

eInfoDesc.box.Description = 'reach-to-touch trial procedure';
eInfoDesc.box.Levels.spawned = 'box spawned on table in front of participant';
eInfoDesc.box.Levels.touched = 'participant completed reach-to-touch, moment of collision with object';

eInfoDesc.trial_nr.Description = 'increasing counter of trials';
eInfoDesc.currentBlockNr.Units = 'integer';

eInfoDesc.normal_or_conflict.Description = 'reach-to-touch trial condition';
eInfoDesc.normal_or_conflict.Levels.normal = 'congruent sensory feedback, collider size matches object size';
eInfoDesc.normal_or_conflict.Levels.conflict = 'incongruent sensory feedback, collider size bigger than object size causing to too-early sensory feedback';

eInfoDesc.cube.Description = 'location of spawned object from participants perspective';
eInfoDesc.cube.Levels.left = 'to participants left';
eInfoDesc.cube.Levels.middle = 'in front of the participant';
eInfoDesc.cube.Levels.right = 'to participants right';

eInfoDesc.isiTime.Description = 'inter-stimulus-interval; time elapsed from trial start to object spawn';
eInfoDesc.isiTime.Units = 'seconds';

eInfoDesc.emsFeedback.Description = 'whether electrical muscle stimulation occurred';
eInfoDesc.emsFeedback.Levels.on = 'electrical muscle stimulation active';

eInfoDesc.reaction_time.Description = 'duration of reach-to-touch; time elapsed between object spawn and object touch';
eInfoDesc.reaction_time.Units = 'seconds';

eInfoDesc.emsCurrent.Description = 'electrical muscle stimulation parameter: current';
eInfoDesc.emsCurrent.Units = 'milliampere';

eInfoDesc.emsWidth.Description = 'electrical muscle stimulation parameter: pulse width';
eInfoDesc.emsWidth.Units = 'microseconds';

eInfoDesc.pulseCount.Description = 'electrical muscle stimulation parameter: pulse count';
eInfoDesc.pulseCount.Units = 'count';

eInfoDesc.vibroFeedback.Description = 'whether vibrotactile stimulation occurred';
eInfoDesc.vibroFeedback.Levels.on = 'vibrotactile stimulation active';

eInfoDesc.vibroFeedbackDuration.Description = 'duration of activated vibrotactile motor';
eInfoDesc.vibroFeedbackDuration.Units = 'seconds';

eInfoDesc.visualFeedback.Description = 'remove rendering of object after touching';
eInfoDesc.visualFeedback.Levels.off = 'object removed';

eInfoDesc.ipq_question_nr_1_answer.Description = 'answer to IPQ item 1';
eInfoDesc.ipq_question_nr_1_answer.Units = 'Likert';

eInfoDesc.ipq_question_nr_2_answer.Description = 'answer to IPQ item 2';
eInfoDesc.ipq_question_nr_2_answer.Units = 'Likert';

eInfoDesc.ipq_question_nr_3_answer.Description = 'answer to IPQ item 3';
eInfoDesc.ipq_question_nr_3_answer.Units = 'Likert';

eInfoDesc.ipq_question_nr_4_answer.Description = 'answer to IPQ item 4';
eInfoDesc.ipq_question_nr_4_answer.Units = 'Likert';
%--------------------------------------------------------------------------
% 3. Export in BIDS format
%--------------------------------------------------------------------------

bids_export(subject,                                                ...
    'targetdir', targetFolder,                                      ... 
    'taskName', 'ReachToTouchPredictionError',                                              ...
    'gInfo', gInfo,                                           ...
    'pInfo', pInfo,                                                 ...
    'pInfoDesc', pInfoDesc,                                         ...
    'eInfo',eInfo,                                                  ...
    'eInfoDesc', eInfoDesc,                                         ...
    'README', README,                                               ...
    'trialtype', trialType,                                         ...
    'tInfo', tInfo)