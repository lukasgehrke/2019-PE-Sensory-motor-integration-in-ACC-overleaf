function [keys, types] = PE_set_to_mobids_events(eegFileName, eegFilePath)

% if ~exist('eeglab','var'); eeglab; end

EEG             = pop_loadset('filename', eegFileName, 'filepath', eegFilePath);
EEG             = parse_events_pe(EEG); % function used for CHI submission
% EEG.event = renamefields(EEG.event, 'trial_type', 'type');

% change empty fields to 'n/a' and ':' to '_' 
for i = 1:numel(EEG.event)
    fn = fieldnames(EEG.event(i));
    tf = cellfun(@(c) isempty(EEG.event(i).(c)), fn);
    fields = fn(tf);
    for j = 1:numel(fields)
        EEG.event(i).(string(fields(j))) = 'n/a';
    end
    
    EEG.event(i).type = strrep(EEG.event(i).type,':','_');
end

% find all keys (properties of events)
keys = fieldnames(EEG.event);
% exclude fields that are not properties in BIDS
% key 'type' is excluded and replaced with 'trial_type' later
keys = keys(~contains(keys,{'type', 'duration', 'latency', 'urevent', 'hedTag'}));
% find all types of events
% ---------------------------------------------------------------------
EEG.event = renamefields(EEG.event, 'type', 'trial_type');

types = unique({EEG.event.trial_type});
ipq_types = types(contains(types,'ipq')); % clean ipq events
ipq_types = cellfun(@(x) x(1:end-2), ipq_types, 'un', 0);
types = types(~contains(types,{'ipq', 'duplicate_event'}));
types = unique([types, ipq_types]);

pop_saveset(EEG, 'filename', eegFileName, 'filepath', eegFilePath);

end


