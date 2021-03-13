function [ noisy_epochs ] = pe_clean_epochs( EEG, events_to_include, bemobil_config )
%PE_CLEAN_EPOCHS Find noisy epochs using eeglab's autorej

% cleaning on the basis of select ERP epochs 
EEG.event = EEG.event(events_to_include);
[EEG.event.type] = EEG.event.trial_type;
% EEG.event = renamefields(EEG.event, 'trial_type', 'type');


clean_EEG = pop_epoch( EEG, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
    'epochs', 'epochinfo', 'yes');

% EEG: find ~10% noisiest epoch indices by searching for large amplitude
% fluctuations on the channel level using eeglab auto_rej function
EEG.event = []; % for speed up
[~, noisy_epochs] = pop_autorej(clean_EEG, 'maxrej', 2, 'nogui','on','eegplot','off');

end

