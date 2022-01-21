% load single subject files with ICA and clean the events and copy design
% matrix to EEG.etc.analysis

clear all;

config_XDF_pe;
subjects = 2:20;
eeglab;

%% load to_bids file, kick out the EEG data and channels and save as mocap in single_subject_analysis_folder
for subject = subjects
    
    EEG = pop_loadset(fullfile([bemobil_config.study_folder ...
        bemobil_config.single_subject_analysis_folder ...
        bemobil_config.filename_prefix num2str(subject)], ...
        [bemobil_config.filename_prefix num2str(subject) '_'...
        bemobil_config.copy_weights_interpolate_avRef_filename]));
    
    EEG.event = EEG.urevent;
    EEG = parse_events_PE(EEG);
    
    test = pop_epoch( EEG, {  'box:touched'  }, [-1  2], 'newname', 'Merged datasets resampled epochs', 'epochinfo', 'yes');
    epochs = pop_loadset(['sub-', sprintf('%03d', subject-1), '_epochs_box_touched.set'],...
        '/Volumes/work/studies/Prediction_Error/derivatives/data');
    assert(size(test.epoch,2)==size(epochs.epoch,2))
    
    
    EEG.etc.analysis = epochs.etc.analysis;
    EEG.etc.analysis.epoching = '[-3 2] around box:touched';
    
    pop_saveset(EEG, ['/Volumes/work/studies/Prediction_Error/data/DFA/', 'sub-', sprintf('%03d', subject)]);
    
end

% print out the struct to explain it to chris