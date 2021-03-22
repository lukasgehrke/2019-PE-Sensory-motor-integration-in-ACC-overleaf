function [onset_sample, bad_movement_profile] = pe_movement_onset_detector(motion_segment, motion_onset_threshold)
%PE_MOVEMENT_DETECTOR Detects movement onset and subsequent peak in
%velocity
%
% returns movement onset in the give profile, additionally returns boolean
% for segments where onset detection was difficult and might lack accuracy

movement_detected = min(find(motion_segment>(max(motion_segment)*.6)));
early_move_phase_reverse = flipud(motion_segment(1:movement_detected));

try
    if isempty(min(find(early_move_phase_reverse < 0.05)))
        bad_movement_profile = true;
        first_local_min = min(find(islocalmin(early_move_phase_reverse)));
        if isempty(first_local_min)
            onset_sample = size(early_move_phase_reverse,1);
        else
            onset_sample = size(early_move_phase_reverse,1) - first_local_min;
        end
    else
        bad_movement_profile = false;
        onset_sample = size(early_move_phase_reverse,1) - min(find(early_move_phase_reverse < motion_onset_threshold));
    end
catch
    bad_movement_profile = true;
    onset_sample = size(early_move_phase_reverse,1);
end
end

