function [events, eventsJSON] = bids_parsemarkers_pe(events)
% default function does not parse markers
% enters zeros in duration fields and constructs a struct to be used in
% events.json 

% parse markers
for iEvent = 1:numel(events)
    events(iEvent).duration = 0;
    
    if strfind(events(iEvent).value, 'box:touched')
       first_touch = iEvent;
       second_touch = [];
       spawn = [];
       
       % find second touch event occuring before spawn of next trial
       try
           for j = 1:100
               if strfind(events(iEvent+j).value, 'box:touched')
                   second_touch = iEvent+j;
                   break
               end
           end
           
           for j = 1:100
               if strfind(events(iEvent+j).value, 'box:spawned')
                   spawn = iEvent+j;
                   break
               end
           end

           % if second touch found before next trial, delete first touch
           % event, use second touch event for parsing, contains EMS info
           if spawn > second_touch
               events(first_touch).value = 'duplicate_event';
           end
       end
    end
    
    current_event = cellstr(strsplit(events(iEvent).value, ';'));

    % parse all events
    try
        for j=1:length(current_event)
            key_val = cellstr(strsplit(current_event{j}, ':'));
            events(iEvent).(key_val{1}) = key_val{2};
            if j==1
                events(iEvent).value = strcat(key_val{1}, ':', key_val{2});
            end
        end
    end
    
    % maintain last 'box:spawned' event to overwrite condition param
    if strcmp(events(iEvent).value, 'box:spawned')
        last_spawned = iEvent;
    end
    
    if isfield(events(iEvent), 'emsFeedback') && strcmp(events(iEvent).emsFeedback, 'on')
        events(iEvent).condition = 'ems';
        events(last_spawned).condition = 'ems';
    end
    
end


% ---------------------------------------------------------------------
eventsJSON.timestamp.Description = 'Event onset';
eventsJSON.timestamp.Units = 'second';

eventsJSON.duration.Description = 'Event duration';
eventsJSON.duration.Units = 'second';

eventsJSON.type.Description = 'Type of event';

% event information description
% -----------------------------
eventsJSON.response_time.Description = 'Response time column not used for this data';
eventsJSON.sample.Description = 'Event sample starting at 0 (Matlab convention starting at 1)';
eventsJSON.value.Description = 'Value of event (raw makers)';

% custom part: key 'type' is not included
%----------------------------------------
eventsJSON.block.Description = 'start and end of an experimental block';
eventsJSON.block.Levels.start = 'start';
eventsJSON.block.Levels.end = 'end';

eventsJSON.currentBlockNr.Description = 'three experimental blocks per condition';
eventsJSON.currentBlockNr.Units = 'integer';

eventsJSON.condition.Description = 'sensory feedback type';
eventsJSON.condition.Levels.visual = 'visual';
eventsJSON.condition.Levels.vibro = 'visual + vibrotactile';
eventsJSON.condition.Levels.ems = 'visual + vibrotactile + electrical muscle stimulation';

eventsJSON.training.Description = '1 (true) if pre experimental training condition';

eventsJSON.box.Description = 'reach-to-touch trial procedure';
eventsJSON.box.Levels.spawned = 'box spawned on table in front of participant';
eventsJSON.box.Levels.touched = 'participant completed reach-to-touch, moment of collision with object';

eventsJSON.trial_nr.Description = 'increasing counter of trials';
eventsJSON.currentBlockNr.Units = 'integer';

eventsJSON.normal_or_conflict.Description = 'reach-to-touch trial condition';
eventsJSON.normal_or_conflict.Levels.normal = 'congruent sensory feedback, collider size matches object size';
eventsJSON.normal_or_conflict.Levels.conflict = 'incongruent sensory feedback, collider size bigger than object size causing to too-early sensory feedback';

eventsJSON.cube.Description = 'location of spawned object from participants perspective';
eventsJSON.cube.Levels.left = 'to participants left';
eventsJSON.cube.Levels.middle = 'in front of the participant';
eventsJSON.cube.Levels.right = 'to participants right';

eventsJSON.isiTime.Description = 'inter-stimulus-interval; time elapsed from trial start to object spawn';
eventsJSON.isiTime.Units = 'seconds';

eventsJSON.emsFeedback.Description = 'whether electrical muscle stimulation occurred';
eventsJSON.emsFeedback.Levels.on = 'electrical muscle stimulation active';

eventsJSON.reaction_time.Description = 'duration of reach-to-touch; time elapsed between object spawn and object touch';
eventsJSON.reaction_time.Units = 'seconds';

eventsJSON.emsCurrent.Description = 'electrical muscle stimulation parameter: current';
eventsJSON.emsCurrent.Units = 'milliampere';

eventsJSON.emsWidth.Description = 'electrical muscle stimulation parameter: pulse width';
eventsJSON.emsWidth.Units = 'microseconds';

eventsJSON.pulseCount.Description = 'electrical muscle stimulation parameter: pulse count';
eventsJSON.pulseCount.Units = 'count';

eventsJSON.vibroFeedback.Description = 'whether vibrotactile stimulation occurred';
eventsJSON.vibroFeedback.Levels.on = 'vibrotactile stimulation active';

eventsJSON.vibroFeedbackDuration.Description = 'duration of activated vibrotactile motor';
eventsJSON.vibroFeedbackDuration.Units = 'seconds';

eventsJSON.visualFeedback.Description = 'remove rendering of object after touching';
eventsJSON.visualFeedback.Levels.off = 'object removed';

eventsJSON.ipq_question_nr_1_answer.Description = 'answer to IPQ item 1';
eventsJSON.ipq_question_nr_1_answer.Units = 'Likert';

eventsJSON.ipq_question_nr_2_answer.Description = 'answer to IPQ item 2';
eventsJSON.ipq_question_nr_2_answer.Units = 'Likert';

eventsJSON.ipq_question_nr_3_answer.Description = 'answer to IPQ item 3';
eventsJSON.ipq_question_nr_3_answer.Units = 'Likert';

eventsJSON.ipq_question_nr_4_answer.Description = 'answer to IPQ item 4';
eventsJSON.ipq_question_nr_4_answer.Units = 'Likert';


end