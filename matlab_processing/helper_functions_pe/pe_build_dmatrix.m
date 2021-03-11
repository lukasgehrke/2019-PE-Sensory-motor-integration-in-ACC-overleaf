function [dmatrix, epoch_event_ixs] = pe_build_dmatrix(EEG, bemobil_config)
%BUILD_DMATRIX_PE Summary of this function goes here
%   Detailed explanation goes here
% hardoced catching marker mistakes

% get event indices
spawn_ixs = find(strcmp({EEG.event.trial_type}, bemobil_config.epoching.base_epoching_event));
epoch_event_ixs = find(strcmp({EEG.event.trial_type}, bemobil_config.epoching.event_epoching_event));

design = EEG;
design.event = design.event(epoch_event_ixs);
design.event = renamefields(design.event, 'trial_type', 'type');
design = pop_epoch( design, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
    'epochs', 'epochinfo', 'yes');

% remove all trials that have a distance greater than 2s between 
% spawn and touch: participants where slow to react / start the trial
latency_diff = (cell2mat({EEG.event(epoch_event_ixs).latency}) - cell2mat({EEG.event(spawn_ixs).latency})) / EEG.srate;
dmatrix.slow_rt_ixs = find(latency_diff>2);

dmatrix.rt_spawned_touched = latency_diff;
dmatrix.isitime = str2double({design.epoch.eventisiTime});

% factor: oddball
dmatrix.haptics = categorical({design.epoch.eventfeedback})=='vibro';

% factor: haptics
dmatrix.oddball = categorical({design.epoch.eventnormal_or_conflict})=='conflict';
% factor: direction
direction = {design.epoch.eventcube};
direction = strrep(direction, 'CubeLeft (UnityEngine.GameObject)', 'left');
direction = strrep(direction, 'CubeRight (UnityEngine.GameObject)', 'right');
direction = strrep(direction, 'CubeMiddle (UnityEngine.GameObject)', 'middle');
dmatrix.direction = direction;

% factor: trial number
dmatrix.trial_number = str2double({design.epoch.eventtrial_nr});

% factor: sequence
count = 0;
for i = 1:size(design.epoch,2)
    if ~dmatrix.oddball(1,i)
        count = count+1;
        dmatrix.sequence(i) = 0;
    else
        dmatrix.sequence(i) = count;
        count = 0;
    end
end
    
end

