% to run on bpn server, change
% 1. '/Volumes/work/' to 'projects/Lukas_Gehrke/' in all path definitions in this file
% 2. change all '/' to '\' in path definitions
% 3. change the next line to: addpath('P:\projects\Sein_Jeung\Project_BIDS\Examples_public\fieldtrip')

addpath('/Volumes/work/studies/Examples_public/fieldtrip')

% general metadata shared across all modalities
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
generalInfo = [];

% required for dataset_description.json
generalInfo.dataset_description.Name                = 'Prediction Error';
generalInfo.dataset_description.BIDSVersion         = 'unofficial extension';

% optional for dataset_description.json
generalInfo.dataset_description.License             = 'CC0';
generalInfo.dataset_description.Authors             = ['Lukas Gehrke', 'Sezen Akman', 'Albert Chen', 'Pedro Lopes', 'Klaus Gramann'];
generalInfo.dataset_description.Acknowledgements    = 'We thank Avinash Singh, Tim Chen and C.-T. Lin from the Univsersity of Sydney (New South Wales, Australia) for their help developing the task.';
generalInfo.dataset_description.Funding             = ['n/a'];
generalInfo.dataset_description.ReferencesAndLinks  = ['Detecting Visuo-Haptic Mismatches in Virtual Reality using the Prediction Error Negativity of Event-Related Brain Potentials. Lukas Gehrke, Sezen Akman, Pedro Lopes, Albert Chen, Avinash Kumar Singh, Hsiang-Ting Chen, Chin-Teng Lin and Klaus Gramann | In Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems (CHI ’19). ACM, New York, NY, USA, Paper 427, 11 pages. DOI: https://doi.org/10.1145/3290605.3300657'];    
generalInfo.dataset_description.DatasetDOI          = '10.18112/openneuro.ds003552.v1.2.0';
generalInfo.dataset_description.EthicsApproval      = ["GR_10_20180603"];

% general information shared across modality specific json files 
generalInfo.InstitutionName                         = 'Technische Universitaet zu Berlin';
generalInfo.InstitutionalDepartmentName             = 'Biological Psychology and Neuroergonomics';
generalInfo.InstitutionAddress                      = 'Strasse des 17. Juni 135, 10623, Berlin, Germany';
generalInfo.TaskDescription                         = 'Mismatch Negativity paradigm in which participants equipped with VR HMD and 64 Channel EEG reached to touch virtual objects';
 

% Content for README file
% -----------------------
README = sprintf( [ '19 participants were tested in a virtual reality (VR) reach-to-object task.\n'...
    'Participants experienced visual, visual with vibrotractile or visual with vibrotactile and \n'...
    'electrical muscle stimulation (EMS) feedback.\n'...
    'Participants rated their experience on the Immersion and Presence Questionnaire (IPQ) and their workload on the NASA-TLX']);
                

% Content for CHANGES file
% ------------------------
CHANGES = sprintf([ 'Revision history for prediction error dataset\n\n' ...
                    'version 1.0.0 - 1 Mar 2021\n' ...
                    ' - Initial release\n' ...
                    'version 1.2.0 - 12 Oct 2021\n' ...
                    ' - Raw Data only']);

% information about the eeg recording system 
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
eegInfo     = []; 
eegInfo.eeg.ManufacturersModelName = 'BrainProducts BrainAmp';
eegInfo.eeg.SamplingFrequency = 500;
eegInfo.eeg.EOGChannelCount = 0;
eegInfo.eeg.PowerLineFrequency = 50;
eegInfo.eeg.EEGReference = 'REF';
eegInfo.eeg.SoftwareFilters = 'n/a';

                                                   
% information about the motion recording system 
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
motionInfo  = []; 

% motion specific fields in json
motionInfo.motion = [];
motionInfo.motion.RecordingType                     = 'continuous';

% system 1 information
motionInfo.motion.TrackingSystems.Optical.Manufacturer                     = 'HTC';
motionInfo.motion.TrackingSystems.Optical.ManufacturersModelName           = 'Vive Pro';
motionInfo.motion.TrackingSystems.Optical.SamplingFrequencyNominal         = 'n/a'; %  If no nominal Fs exists, n/a entry returns 'n/a'. If it exists, n/a entry returns nominal Fs from motion stream.

% coordinate system
motionInfo.coordsystem.MotionCoordinateSystem      = 'RUF';
motionInfo.coordsystem.MotionRotationRule          = 'left-hand';
motionInfo.coordsystem.MotionRotationOrder         = 'ZXY';

% participant information 
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% here describe the fields in the participant file
% for numerical values  : 
%       subjectData.fields.[insert your field name here].Description    = 'describe what the field contains';
%       subjectData.fields.[insert your field name here].Unit           = 'write the unit of the quantity';
% for values with discrete levels :
%       subjectData.fields.[insert your field name here].Description    = 'describe what the field contains';
%       subjectData.fields.[insert your field name here].Levels.[insert the name of the first level] = 'describe what the level means';
%       subjectData.fields.[insert your field name here].Levels.[insert the name of the Nth level]   = 'describe what the level means';
%--------------------------------------------------------------------------
participantInfo = [];

% participant information for participants.tsv file
% -------------------------------------------------
tmp = readtable('/Volumes/work/studies/Prediction_Error/admin/questionnaires_PE_2018.xlsx', 'Sheet', 'Matlab Import');
participantInfo.cols = tmp.Properties.VariableNames;
participantInfo.cols{1} = 'nr';
participantInfo.data = [table2cell(tmp)];
        
% participant column description for participants.json file
% ---------------------------------------------------------
participantInfo.fields.nr.Description = 'unique participant identifier';
participantInfo.fields.biological_sex.Description = 'biological sex of the participant';
participantInfo.fields.biological_sex.Levels.m = 'male';
participantInfo.fields.biological_sex.Levels.f = 'female';
participantInfo.fields.age.Description = 'age of the participant';
participantInfo.fields.age.Units       = 'years';
participantInfo.fields.cap_size.Description = 'head circumference and EEG cap sized used';
participantInfo.fields.cap_size.Units       = 'centimeter';
participantInfo.fields.block_1.Description = 'pseudo permutation of sensory feedback conditions: condition of first block';
participantInfo.fields.block_1.Units       = 'Visual = Visual only condition; Visual + Vibro = Simultaneous visual and vibrotactile sensory feedback';
participantInfo.fields.block_2.Description = 'pseudo permutation of sensory feedback conditions: condition of second block';
participantInfo.fields.block_3.Description = 'some select participants completed a third block with Visual + Vibro + EMS sensory feedback';
participantInfo.fields.block_3.Units       = 'Visual + Vibro + EMS = Simultaneous visual, vibrotactile and electrical muscle stimulation sensory feedback';

sessions = {'TestVisual', 'TestVibro', 'TestEMS', 'Training'};
ems_subjects = [1,2,5,6,7,10,11,12,13,14,15];
runs = {'erste100', '101bis300'};

%% loop over participants
for subject = 1:size(participantInfo.data,1)
    
    config                        = [];                                 % reset for each loop 
    config.bids_target_folder     = '/Volumes/work/studies/Prediction_Error/data/1_BIDS-data';                     % required            
    config.task                   = 'PredError';                      % optional 
    config.subject                = subject;                            % required
    config.overwrite              = 'on';
    config.eeg.stream_name        = 'BrainVision';                      % required
    config.eeg.SamplingFrequency = 500;
    config.eeg.EOGChannelCount = 0;
    config.eeg.PowerLineFrequency = 50;
    config.eeg.EEGReference = 'REF';
    config.eeg.SoftwareFilters = 'n/a';
    
    config.motion.streams{1}.stream_name        = 'Rigid_Head';
    config.motion.streams{1}.tracking_system    = 'HTCVive';
    config.motion.streams{1}.tracked_points     = 'Rigid_Head';
    config.motion.streams{1}.tracked_points_anat= 'head';
    config.motion.streams{2}.stream_name        = 'Rigid_handR';
    config.motion.streams{2}.tracking_system    = 'HTCVive';
    config.motion.streams{2}.tracked_points     = 'Rigid_handR';
    config.motion.streams{2}.tracked_points_anat= 'right_hand';    
    
    config.bids_parsemarkers_custom = 'bids_parsemarkers_pe';
    
    for session = sessions
        config.session                = session{1};              % optional
        
        if strcmp(session{1},'TestEMS') && ~ismember(subject,ems_subjects)
            break;
        end
        
        if strcmp(session{1},'TestVibro') && subject == 2
            for run = 1:2
                config.filename               = ['/Volumes/work/studies/Prediction_Error/data/0_raw-data/s' num2str(subject+1) '/s' num2str(subject+1) '_PredError_block_' session{1} '_' runs{run} '.xdf']; % required
                
                bemobil_xdf2bids(config, ...
                    'general_metadata', generalInfo,...
                    'participant_metadata', participantInfo,...
                    'motion_metadata', motionInfo, ...
                    'eeg_metadata', eegInfo);
            end
        else
            config.filename               = ['/Volumes/work/studies/Prediction_Error/data/0_raw-data/s' num2str(subject+1) '/s' num2str(subject+1) '_PredError_block_' session{1} '.xdf']; % required
            bemobil_xdf2bids(config, ...
                'general_metadata', generalInfo,...
                'participant_metadata', participantInfo,...
                'motion_metadata', motionInfo, ...
                'eeg_metadata', eegInfo);
        end
    end
end

%         % optional run processing 
%         if participant == 1 && session == 1
%             for run = 1:3
%                 config.filename              = ['rec' num2str(run) '.xdf']; 
%                 config.run                   = run;                           
%                 bemobil_xdf2bids(config, ...
%                     'general_metadata', generalInfo,...
%                     'participant_metadata', participnatInfo,...
%                     'motion_metadata', motionInfo, ...
%                     'eeg_metadata', eegInfo);
%             end
%         else 
%             bemobil_xdf2bids(config, ...
%                 'general_metadata', generalInfo,...
%                 'participant_metadata', participnatInfo,...
%                 'motion_metadata', motionInfo, ...
%                 'eeg_metadata', eegInfo);
%         end
    