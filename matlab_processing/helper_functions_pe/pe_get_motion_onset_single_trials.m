function [md] = pe_get_motion_onset_single_trials(peEEG, event_sample_ix, motion_onset_threshold, subject, bemobil_config)
%PE_GET_MOTION_ONSET_SINGLE_TRIALS Detects movement onsets based on velocity thresholding and returns:
% 
% returns md (= motion detection) with fields:
% - movement_onset_sample: the detected movement onset sample index
% - peak velocity for the next peak following movement onset

% load(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_single_trial_dmatrix.mat']));
% motion = single_trial_dmatrix.motion.mag_vel;
motion = peEEG.etc.analysis.motion.mag_vel;
z = peEEG.etc.analysis.motion.z;

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
        
        % I ONLY NEED ONSET, VEL PEAK AND OFFSET/FIRST BACK MOVEMENT
        
        % outward movement onset
        reach_onset_segment = motion(peEEG.etc.analysis.design.spawn_event_sample(i):event_sample_ix,i);
        [md.reach_onset_sample(i), md.bad_reach_movement_profile(i)] = pe_movement_onset_detector(reach_onset_segment, motion_onset_threshold);
        md.reach_onset_sample(i) = peEEG.etc.analysis.design.spawn_event_sample(i) + md.reach_onset_sample(i);
        [md.reach_max_vel(i), md.reach_max_vel_ix(i)] = max(motion(md.reach_onset_sample(i):event_sample_ix,i));
        md.reach_max_vel_ix(i) = md.reach_onset_sample(i) + md.reach_max_vel_ix(i);
        
        % outward movement offset: first change in z direction after movement onset
        buffer = 5;
        z_onset_segment = motion(md.reach_max_vel_ix(i)+buffer:end,i);
        md.reach_off_sample(i) = md.reach_max_vel_ix(i) + buffer + min(find(diff(sign(diff(z_onset_segment)))));

%         % retract movement onset
%         if peEEG.etc.analysis.design.oddball(i)=='true'
% %             motion_stop = min(find(islocalmin(retract_segment)));
%             buffer = 10;
% %             motion_stop = motion_stop - buffer;
% %             retract_segment = motion(event_sample_ix+motion_stop:end,i);
%         else
%             buffer = 0;
%         end

% buffer = 10;
% 
%         retract_segment = motion(event_sample_ix+buffer:end,i);
%         [md.retract_onset_sample(i), md.bad_retract_movement_profile(i)] = pe_movement_onset_detector(retract_segment, motion_onset_threshold);
%         md.retract_onset_sample(i) = event_sample_ix + buffer + md.retract_onset_sample(i);
%         
%         % retract movement off
%         retract_segment_reverse = flipud(motion(md.retract_onset_sample(i):end,i));
%         [md.retract_off_sample(i), ~] = pe_movement_onset_detector(retract_segment_reverse, motion_onset_threshold);
%         md.retract_off_sample(i) = md.retract_onset_sample(i) + (size(retract_segment_reverse,1) - md.retract_off_sample(i));
% 
%         % retract descriptives
%         [md.retract_max_vel(i), md.retract_max_vel_ix(i)] = max(motion(md.retract_onset_sample(i):md.retract_off_sample(i),i));
%         md.retract_max_vel_ix(i) = md.retract_onset_sample(i) + md.retract_max_vel_ix(i);
%         
%         % outward movement off
%         reach_onset_reverse = flipud(motion(md.reach_onset_sample(i):md.retract_onset_sample(i)+buffer,i));
%         [md.reach_off_sample(i), ~] = pe_movement_onset_detector(reach_onset_reverse, motion_onset_threshold);
%         md.reach_off_sample(i) = md.reach_onset_sample(i) + (size(reach_onset_reverse,1) - md.reach_off_sample(i));
%         
%         % reach descriptives
%         [md.reach_max_vel(i), md.reach_max_vel_ix(i)] = max(motion(md.reach_onset_sample(i):md.reach_off_sample(i),i));
%         md.reach_max_vel_ix(i) = md.reach_onset_sample(i) + md.reach_max_vel_ix(i);

%         reach_segment = motion(md.reach_onset_sample(i):md.retract_onset_sample(i),i);
%         md.reach_max_vel(i) = max(reach_segment);
%         md.reach_max_vel_ix(i) = md.reach_onset_sample(i) + find(reach_segment == md.reach_max_vel(i)) - 1;
%         
%         approach_segment = z(1,md.reach_max_vel_ix(i):end,i);
%         z_approach = min(find(diff(sign(approach_segment>max(approach_segment) * .1))));
%         md.outward_motion_stop(i) = md.reach_max_vel_ix(i) + z_approach;

    end

    %% plot
    if ismember(i, subplot_ixs) && ~md.missing_spawn_event(i)
        subplot(6,6,plot_ix);
        full_reach_segment = motion(:,i);
        x = 1:size(motion,1);
        plot(x, full_reach_segment);
        hold on;
        
        plot(x(md.reach_onset_sample(i)),full_reach_segment(md.reach_onset_sample(i)),'r*');
        plot(x(md.reach_max_vel_ix(i)),full_reach_segment(md.reach_max_vel_ix(i)),'g*');
        plot(x(md.reach_off_sample(i)),full_reach_segment(md.reach_off_sample(i)),'b*');
        
%         plot(x(md.retract_onset_sample(i)),full_reach_segment(md.retract_onset_sample(i)),'r*');
%         plot(x(md.retract_max_vel_ix(i)),full_reach_segment(md.retract_max_vel_ix(i)),'g*');
%         plot(x(md.retract_off_sample(i)),full_reach_segment(md.retract_off_sample(i)),'b*');
        
        ylim([0,2]);
        
        vline(peEEG.etc.analysis.design.spawn_event_sample(i), 'k');
        vline(md.reach_onset_sample(i), 'r');
        vline(md.reach_max_vel_ix(i), 'g');
        vline(md.reach_off_sample(i), 'b');
        
        vline(event_sample_ix, 'k');
%         vline(md.retract_onset_sample(i), 'r');
%         vline(md.retract_max_vel_ix(i), 'g');
%         vline(md.retract_off_sample(i), 'b');
        
        title(['trial: ', num2str(i)])
        plot_ix = plot_ix + 1;
    end

end

legend('mag. of velocity', 'movement onset', 'move max. vel.', 'inflection point'); % , 'move max. acc.'
out_folder = fullfile(bemobil_config.study_folder, 'movement_figs');
if ~isfolder(out_folder)
    mkdir(out_folder);
end
saveas(f, fullfile(out_folder, ['sub-', sprintf('%03d', subject), '_movement_onsets']), 'fig');

end

