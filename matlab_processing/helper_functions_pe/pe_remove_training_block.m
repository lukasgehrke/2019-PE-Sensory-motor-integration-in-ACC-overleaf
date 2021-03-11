function [EEG_pe] = pe_remove_training_block(EEG_pe)
%REMOVE_TRAINING_BLOCK removes all events occuring in training mode of
%prediction error experiment (pe)

training = find(ismember({EEG_pe.event.training}, 'true'));
training_block_start = intersect(training, find(ismember({EEG_pe.event.block}, 'start')));
training_block_end = intersect(training, find(ismember({EEG_pe.event.block}, 'end')));

training_start_end_ixs = [training_block_start(1)];
tmp_end_ix = 1;
for i = 2:size(training_block_start,2)
    if training_block_start(i) > training_block_end(tmp_end_ix)
        training_start_end_ixs = [training_start_end_ixs, training_block_end(tmp_end_ix), training_block_start(i)];
        tmp_end_ix = tmp_end_ix + 1;
    end
end
training_start_end_ixs = [training_start_end_ixs, training_block_end(tmp_end_ix)];

if size(training_start_end_ixs,2) > 2
    training_start_end_ixs = reshape(training_start_end_ixs, size(training_start_end_ixs,2)/2, 2)';
end

rem_events = [];
for i = 1:size(training_start_end_ixs,1)
    rem_events = [rem_events, training_start_end_ixs(i,1):training_start_end_ixs(i,2)];
end

EEG_pe.event(rem_events) = [];

end

