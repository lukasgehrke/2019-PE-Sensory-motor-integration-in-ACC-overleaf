function [movement_detection] = pe_get_motion_onset_single_trials(peEEG, event_sample_ix, subject, bemobil_config)
%PE_GET_MOTION_ONSET_SINGLE_TRIALS Detects movement onsets based on velocity thresholding and returns:
% 
% - movement_onset_sample: the detected movement onset sample index
% - reaction_time: duration between box spawn and movement onset
% - action_time: duration between movement onset and box touched

load(fullfile(bemobil_config.study_folder, 'data', ['sub-', sprintf('%03d', subject), '_single_trial_dmatrix.mat']));

% make a subplot and save every 20th trial
f = figure('visible','on');
sgtitle(['movement magnitude profiles subect ', num2str(subject)])
subplot_ixs = 1:20:numel(peEEG.etc.analysis.design.trial_number);
plot_ix = 1;

bad_move_ix = 1;
movement_detection.bad_movement_profile = [];
for i = 1:size(peEEG.etc.analysis.design.trial_number,2)
    
    %% movement onset detector
    
    if peEEG.etc.analysis.design.spawn_event_sample(i) == 0
        movement_detection.movement_onset_sample(i) = 0;
        movement_detection.bad_movement_profile(bad_move_ix) = i;
        bad_move_ix = bad_move_ix + 1;
    else
        movement = single_trial_dmatrix.motion.mag_vel(peEEG.etc.analysis.design.spawn_event_sample(i):end,i);
        movement_detected(i) = min(find(movement>(max(movement)*.6)));
        early_move_phase_reverse = flipud(movement(1:movement_detected(i)));
        try
            if isempty(min(find(early_move_phase_reverse < 0.05)))
                movement_detection.bad_movement_profile(bad_move_ix) = i;
                bad_move_ix = bad_move_ix + 1;
                movement_onset_sample(i) = size(early_move_phase_reverse,1) - min(find(islocalmin(early_move_phase_reverse)));
            else
                movement_onset_sample(i) = size(early_move_phase_reverse,1) - min(find(early_move_phase_reverse < 0.05));
            end
        catch
            movement_onset_sample(i) = size(early_move_phase_reverse,1);
        end
        movement_detection.move.onset_sample(i) = movement_onset_sample(i) + peEEG.etc.analysis.design.spawn_event_sample(i);
        
        %% max velocity, acceleration outward and return movement
        
        move_phase_vel = single_trial_dmatrix.motion.mag_vel(movement_detection.move.onset_sample(i):end,i);
        move_phase_vel_smooth = movmean(move_phase_vel,100);
        
        % find the two peak maxima locations on the smoothed curve
        local_max_smooth_vel = find(islocalmax(move_phase_vel_smooth, 'MinSeparation', 75));
        tmp_smooth = sortrows([local_max_smooth_vel, move_phase_vel(local_max_smooth_vel)],2,'descend');
        tmp_smooth(3:end,:) = [];
        tmp_smooth = sortrows(tmp_smooth,'ascend');

        % find the two peak maxima locations on the raw curve
        local_max = find(islocalmax(move_phase_vel, 'MinSeparation', 50));
        tmp_raw = sortrows([local_max, move_phase_vel(local_max)],2,'descend');
                
        % outward
        peaks_ixs = tmp_raw(:,1) - tmp_smooth(1,1);
        max_peak_ix = find(peaks_ixs == min(peaks_ixs));
        movement_detection.move.max_vel(i) = tmp_raw(max_peak_ix,2);
        movement_detection.move.max_vel_ix(i) = movement_detection.move.onset_sample(i) + tmp_raw(max_peak_ix,1);
        
        % retract
        peaks_ixs = abs(tmp_raw(:,1) - tmp_smooth(2,1));
        max_peak_ix = find(peaks_ixs == min(peaks_ixs));
        movement_detection.retract.max_vel(i) = tmp_raw(max_peak_ix,2);
        movement_detection.retract.max_vel_ix(i) = movement_detection.move.onset_sample(i) + tmp_raw(max_peak_ix,1);
        
        % inflection point
%         tmp = single_trial_dmatrix.motion.z(:,1:movement_detection.retract.max_vel_ix(i),i);
%         tmp = fliplr(tmp);
%         movement_detection.inflection.vel_ix(i) = movement_detection.retract.max_vel_ix(i) - min(find(diff(sign(tmp))));
        
        buffer = 10;
        tmp = single_trial_dmatrix.motion.mag_vel(1:movement_detection.retract.max_vel_ix(i)-buffer,i);
        tmp = flipud(tmp);
        movement_detection.inflection.vel_ix(i) = movement_detection.retract.max_vel_ix(i) - min(find(sign(diff(tmp))>0)) - buffer + 1;
        
%         single_trial_dmatrix.motion.z(1,movement_detection.move.max_vel_ix(i):movement_detection.retract.max_vel_ix(i),i);
%         movement_detection.inflection.vel_ix(i) = movement_detection.move.max_vel_ix(i) + max(find(diff(sign(tmp))));
        
%         %% maximum acceleration and velocity ballistic phase
%         
%         move_phase_acc = single_trial_dmatrix.motion.mag_acc(movement_detection.move.onset_sample(i):end,i);
%         
%         tmp = min(find(islocalmax(move_phase_acc)));
%         movement_detection.move.max_acc(i) = move_phase_acc(tmp);
%         movement_detection.move.max_acc_ix(i) = movement_detection.move.onset_sample(i) + tmp;
% 
%         move_phase_vel = single_trial_dmatrix.motion.mag_vel(movement_detection.move.onset_sample(i):end,i);
%         local_max = find(islocalmax(move_phase_vel, 'MinSeparation', 50));
%         
%         tmp = sortrows([local_max, move_phase_vel(local_max)],2,'descend');
%         tmp(3:end,:) = [];
%         tmp = sortrows(tmp,'ascend');
%         
%         % outward
%         movement_detection.move.max_vel(i) = tmp(1,2);
%         movement_detection.move.max_vel_ix(i) = movement_detection.move.onset_sample(i) + tmp(1,1);
%         
%         % retract
%         movement_detection.retract.max_vel(i) = tmp(2,2);
%         movement_detection.retract.max_vel_ix(i) = movement_detection.move.onset_sample(i) + tmp(2,1);
%         
%         % inflection point
%         tmp = single_trial_dmatrix.motion.z(1,movement_detection.move.max_vel_ix(i):movement_detection.retract.max_vel_ix(i),i);
%         movement_detection.inflection.vel_ix(i) = movement_detection.move.max_vel_ix(i) + max(find(diff(sign(tmp))));
        
    end

    %% plot
    if ismember(i, subplot_ixs) && ~ismember(i, movement_detection.bad_movement_profile)
        subplot(6,6,plot_ix);
        movement = single_trial_dmatrix.motion.mag_vel(:,i);
        x = 1:size(single_trial_dmatrix.motion.mag_vel,1);
        plot(x, movement);
        hold on;
        
        plot(x(peEEG.etc.analysis.design.spawn_event_sample(i)),movement(peEEG.etc.analysis.design.spawn_event_sample(i)),'b*');
        plot(x(movement_detection.move.onset_sample(i)),movement(movement_detection.move.onset_sample(i)),'r*');
        plot(x(movement_detection.move.max_vel_ix(i)),movement(movement_detection.move.max_vel_ix(i)),'g*');
%         plot(x(movement_detection.move.max_acc_ix(i)),movement(movement_detection.move.max_acc_ix(i)),'k*');
        plot(x(movement_detection.retract.max_vel_ix(i)),movement(movement_detection.retract.max_vel_ix(i)),'y*');
        plot(x(movement_detection.inflection.vel_ix(i)),movement(movement_detection.inflection.vel_ix(i)),'m*');
        
        ylim([0,2]);
        
        vline(peEEG.etc.analysis.design.spawn_event_sample(i), 'b');
        vline(movement_detection.move.onset_sample(i), 'r');
        vline(movement_detection.move.max_vel_ix(i), 'g');
%         vline(movement_detection.move.max_acc_ix(i), 'k');
        vline(movement_detection.retract.max_vel_ix(i), 'y');
        vline(movement_detection.inflection.vel_ix(i), 'm');
        
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

