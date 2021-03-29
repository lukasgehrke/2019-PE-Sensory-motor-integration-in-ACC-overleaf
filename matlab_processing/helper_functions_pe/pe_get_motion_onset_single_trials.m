function [md] = pe_get_motion_onset_single_trials(peEEG, event_sample_ix, motion_onset_threshold, subject, bemobil_config)
%PE_GET_MOTION_ONSET_SINGLE_TRIALS Detects movement onsets based on velocity thresholding and returns:
% 
% returns md (= motion detection) with fields:
% - movement_onset_sample: the detected movement onset sample index
% - peak velocity for the next peak following movement onset

% load(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_single_trial_dmatrix.mat']));
% motion = single_trial_dmatrix.motion.mag_vel;
motion = peEEG.etc.analysis.motion.mag_vel;

% make a subplot and save every 20th trial
f = figure('visible','off');
sgtitle(['movement magnitude profiles subect ', num2str(subject)])
subplot_ixs = 1:20:numel(peEEG.etc.analysis.design.trial_number);
plot_ix = 1;

for i = 1:size(peEEG.etc.analysis.design.trial_number,2)
    %% movement onset detector
    if peEEG.etc.analysis.design.spawn_event_sample(i) == 0
        md.missing_spawn_event(i) = 1;
        md.reach_onset_sample(i) = 0;
    else
        md.missing_spawn_event(i) = 0;
        
        %% max velocity, acceleration reach and return movement
        reach_onset_segment = motion(peEEG.etc.analysis.design.spawn_event_sample(i):event_sample_ix,i);
        [md.reach_onset_sample(i), md.bad_reach_movement_profile(i)] = pe_movement_onset_detector(reach_onset_segment, motion_onset_threshold);
        md.reach_onset_sample(i) = peEEG.etc.analysis.design.spawn_event_sample(i) + md.reach_onset_sample(i);
        
        retract_segment = motion(event_sample_ix:end,i);
        motion_stop = min(find(islocalmin(retract_segment)));
        buffer = 5;
        motion_stop = motion_stop - buffer;
        retract_segment = motion(event_sample_ix+motion_stop:end,i);
        [md.retract_onset_sample(i), md.bad_retracht_movement_profile(i)] = pe_movement_onset_detector(retract_segment, motion_onset_threshold);
        md.retract_onset_sample(i) = event_sample_ix + motion_stop + md.retract_onset_sample(i);
        md.retract_max_vel(i) = max(retract_segment);
        md.retract_max_vel_ix(i) = event_sample_ix + motion_stop + find(retract_segment == md.retract_max_vel(i)) - 1;
        
        reach_segment = motion(md.reach_onset_sample(i):md.retract_onset_sample(i),i);
        md.reach_max_vel(i) = max(reach_segment);
        md.reach_max_vel_ix(i) = md.reach_onset_sample(i) + find(reach_segment == md.reach_max_vel(i)) - 1;
        
    end

    %% plot
    if ismember(i, subplot_ixs) && ~md.missing_spawn_event(i)
        subplot(6,6,plot_ix);
        full_reach_segment = motion(:,i);
        x = 1:size(motion,1);
        plot(x, full_reach_segment);
        hold on;
        
        plot(x(peEEG.etc.analysis.design.spawn_event_sample(i)),full_reach_segment(peEEG.etc.analysis.design.spawn_event_sample(i)),'b*');
        
        plot(x(md.reach_onset_sample(i)),full_reach_segment(md.reach_onset_sample(i)),'r*');
        plot(x(md.reach_max_vel_ix(i)),full_reach_segment(md.reach_max_vel_ix(i)),'g*');
        plot(x(md.retract_onset_sample(i)),full_reach_segment(md.retract_onset_sample(i)),'r*');
        plot(x(md.retract_max_vel_ix(i)),full_reach_segment(md.retract_max_vel_ix(i)),'y*');
        
        ylim([0,2]);
        
        vline(peEEG.etc.analysis.design.spawn_event_sample(i), 'b');
        vline(md.reach_onset_sample(i), 'r');
        vline(md.reach_max_vel_ix(i), 'g');
        vline(md.retract_onset_sample(i), 'm');
        vline(md.retract_max_vel_ix(i), 'y');
        
        title(['trial: ', num2str(i)])
        plot_ix = plot_ix + 1;
    end

end

legend('mag. of velocity', 'spawn', 'movement onset', 'move max. vel.', 'retract max. vel.', 'inflection point'); % , 'move max. acc.'
out_folder = fullfile(bemobil_config.study_folder, 'movement_figs');
if ~isfolder(out_folder)
    mkdir(out_folder);
end
saveas(f, fullfile(out_folder, ['sub-', sprintf('%03d', subject), '_movement_onsets']), 'fig');

end

