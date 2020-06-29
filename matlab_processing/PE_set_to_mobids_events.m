function [keys, types] = IMT_set_to_mobids_events(eegFileName, eegFilePath)

if ~exist('eeglab','var'); eeglab; end

EEG             = pop_loadset('filename', eegFileName, 'filepath', eegFilePath);

% find all keys (properties of events)
% ---------------------------------------------------------------------
allKeys         = {};
event           = EEG.event;

for eventIndex = 1:numel(event)
    
    marker = event(eventIndex).type;
    splitMarker = regexp(marker,';','split');
    
    for partIndex = 1:numel(splitMarker)
        
        part = splitMarker(partIndex);
        
        if ~isempty(part{1})
            
            pair    = regexp(part,':','split');
            allKeys{end+1}    = pair{1}{1};
            
        end
    end
end

keys = unique(allKeys);

% key 'type' is excluded and replaced with 'trial_type' later
keys = keys(~strcmp(keys,'type'));


% find all types of events
% ---------------------------------------------------------------------
allTypes         = {};

for eventIndex = 1:numel(event)
    
    marker = [event(eventIndex).type ';'];
    type = extractBetween(marker,'type:',';');
    if ~isempty(type)
        allTypes{end+1} = type{1};
    end
end

types = unique(allTypes);

% rewrite events
%----------------------------------------------------------------------
for eventIndex = 1:numel(event)
    
    % define the type of the event and save as 'trial_type'
    marker  = [event(eventIndex).type ';'];
    trialType = extractBetween(marker,'type:',';');
    
    if ~isempty(trialType)
        EEG.event(eventIndex).trial_type = trialType{1};
    else
        EEG.event(eventIndex).trial_type = 'n/a';
    end
    
    % extract key-value pairs
    pairs   =  regexp(marker,';','split');
    
    % iterate over all key-value pairs
    for pairIndex = 1:numel(pairs)
        
        % separate keys and values
        pairParts = regexp(pairs(pairIndex),':','split');
        
        % iterate over the list of keys
        for keyIndex = 1:numel(keys)
            
            % create a field if it is not already there
            if ~isfield(EEG.event,keys{keyIndex})
                
                EEG.event(eventIndex).(keys{keyIndex}) = [];
                
            end
            
            % fill in the values of matching keys
            if strcmp(pairParts{1}{1},keys{keyIndex})
                
                EEG.event(eventIndex).(keys{keyIndex}) = pairParts{1}{2};
                
                % 'n/a' if the value for the key does not exist
            elseif isempty( EEG.event(eventIndex).(keys{keyIndex}))
                
                EEG.event(eventIndex).(keys{keyIndex}) = 'n/a';
                
            end
            
        end
        
    end
end

pop_saveset(EEG, 'filename', eegFileName, 'filepath', eegFilePath);


end


