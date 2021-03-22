function [dmatrix, epoch_event_ixs] = pe_build_dmatrix(EEG, bemobil_config)
%PE_BUILD_DMATRIX

% get event indices
spawn_ixs = find(strcmp({EEG.event.type}, bemobil_config.epoching.base_epoching_event));
epoch_event_ixs = find(strcmp({EEG.event.type}, bemobil_config.epoching.event_epoching_event));

design = EEG;
design.event = design.event(union(epoch_event_ixs, spawn_ixs));
design = pop_epoch( design, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
    'epochs', 'epochinfo', 'yes');

%% design matrix

% check if each touch epoch has a preceeding spawn
sequence_count = 0;
for i = 1:numel(design.epoch)
    if ~isequal(design.epoch(i).eventtype{1}, bemobil_config.epoching.base_epoching_event{1})
        dmatrix.bad_trial_order_ixs = i;
        continue;
    end
    
    dmatrix.spawn_event_sample(i) = abs((bemobil_config.epoching.event_epochs_boundaries(1) * EEG.srate) - ceil(design.epoch(i).eventlatency{1} * EEG.srate / 1000));
    
    % factor inter-stimulus time
    dmatrix.isitime(i) = str2double(design.epoch(i).eventisiTime{2});

    % factor: haptics
    dmatrix.haptics(i) = categorical(isequal(design.epoch(i).eventcondition{2}, "vibro"));

    % factor: oddball
    dmatrix.oddball(i) = categorical(isequal(design.epoch(i).eventnormal_or_conflict{2}, 'conflict'));
    
    % factor: sequence
    if dmatrix.oddball(i) == "false"
        dmatrix.sequence(i) = sequence_count;
        sequence_count = sequence_count+1;
    else
        dmatrix.sequence(i) = sequence_count;
        sequence_count = 0;
    end
    
    % factor: direction
    direction = design.epoch(i).eventcube{2};
    direction = strrep(direction, 'CubeLeft (UnityEngine.GameObject)', 'left');
    direction = strrep(direction, 'CubeRight (UnityEngine.GameObject)', 'right');
    direction = strrep(direction, 'CubeMiddle (UnityEngine.GameObject)', 'middle');
    dmatrix.direction(i) = string(direction);

    % factor: trial number
    dmatrix.trial_number(i) = str2double(design.epoch(i).eventtrial_nr{2});
    
end

% mark all trials that have a distance greater than 2s between spawn and touch: participants where slow to react / start the trial
dmatrix.rt_spawn_touch_events = ceil((cell2mat({EEG.event(epoch_event_ixs).latency}) - cell2mat({EEG.event(spawn_ixs).latency})));
dmatrix.slow_rt_spawn_touch_events_ixs = find(dmatrix.rt_spawn_touch_events>500);
    
end

