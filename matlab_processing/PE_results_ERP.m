%% clear all and load params
close all; clear all; clc;

PE_config;

%% load study : 2nd Level inference

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 1, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

% load IMT_v1 EEGLAB study struct, keeping at most 1 dataset in memory
input_path_STUDY = [bemobil_config.study_folder bemobil_config.study_level];
if isempty(STUDY)
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    [STUDY ALLEEG] = pop_loadstudy('filename', bemobil_config.study_filename, 'filepath', input_path_STUDY);
    CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
    
    eeglab redraw
end
STUDY_sets = cellfun(@str2num, {STUDY.datasetinfo.subject});

%% TODO:add main effect congruency

%% TODO:adapt compute result 1: main effect velocity components & channels

% 3: effect of velocity on ERP per subject and condition, i.e. pERP, then
% average betas across subjects for each condition

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

robustfit = 0;

% select timewindow of interest
%zero = 3*EEG.srate; % [-3 2] epoch around event    
%event_win_samples = zero + (bemobil_config.epoching.event_win(1) * EEG.srate):zero+(bemobil_config.epoching.event_win(2) * EEG.srate);
    

% try out several ts prior to event
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);
seconds_before_event = .3;
samples_before_event = seconds_before_event * 250;
ts_of_ints = (event_onset-samples_before_event):25:event_onset+50;
% ts_of_ints = ts_of_ints - mocap_aos_samples;

% select best tf_of_ints
% ts_of_ints = ts_of_ints(4);

erp_type = 'non_corrected';

for ts = ts_of_ints
    
%     %% components
%     for cluster = clusters_of_int
% 
%         disp(['Now running analysis for cluster: ' num2str(cluster)]);
%         tic
% 
%         % outpath
%         save_fpath = [bemobil_config.study_folder bemobil_config.study_level 'analyses/erp/' bemobil_config.study_filename(1:end-6) '/cluster_' num2str(cluster)];
%         if ~exist(save_fpath, 'dir')
%             mkdir(save_fpath);
%         end        
% 
%         %% get matching datasets from EEGLAB Study struct
%         unique_setindices = unique(STUDY.cluster(cluster).sets);
%         unique_subjects = STUDY_sets(unique_setindices);
%         all_setindices = STUDY.cluster(cluster).sets;
%         all_sets = STUDY_sets(all_setindices);
%         all_comps = STUDY.cluster(cluster).comps;
% 
%         % load IC data
%         count = 1;
%         for subject = unique_subjects
% 
%             % select EEG dataset
%             [~, ix] = find(subject==subjects);
%             s_eeg = ALLEEG(ix);
% 
%             % loop through all vels and accs at different time points before the
%             % event
%             model = 'erp_sample ~ immersion * vel + trial + direction + sequence';
% 
%             %DESIGN make continuous and dummy coded predictors
%             vel_vis = s_eeg.etc.analysis.mocap.visual.mag_vel(ts,:)'; % correct for age-of-sample
%             vel_vibro = s_eeg.etc.analysis.mocap.vibro.mag_vel(ts,:)';
% 
%             vel = zscore([vel_vis; vel_vibro]);
%             immersion = categorical([zeros(size(vel_vis,1),1); ones(size(vel_vibro,1),1)]);
%             trial = zscore([s_eeg.etc.epoching.visual_tr_num, s_eeg.etc.epoching.vibro_tr_num]');
%             direction = categorical([s_eeg.etc.epoching.visual_dir, s_eeg.etc.epoching.vibro_dir]');
%             sequence = zscore([s_eeg.etc.epoching.visual_match_seq, s_eeg.etc.epoching.vibro_match_seq]');
%             
%             % old predictor_time = [1:size(vel_vis), 1:size(vel_vibro)]';
% 
%             % average ERP if more than 1 comp
%             compos = all_comps(all_sets==subject);
%             if compos > 1
% %                 s_eeg.etc.analysis.erp.base_corrected.visual.comps(compos(1),:,:) = ...
% %                     mean(s_eeg.etc.analysis.erp.base_corrected.visual.comps(compos,:,:),1);
% %                 s_eeg.etc.analysis.erp.base_corrected.vibro.comps(compos(1),:,:) = ...
% %                     mean(s_eeg.etc.analysis.erp.base_corrected.vibro.comps(compos,:,:),1);
%                 compos = compos(1); % select first IC
%             end
% 
%             % now fit linear model for each component
%             % after averaging take one IC per subject in cluster
%             disp(['running lm for subject ' num2str(subject) ' and comp ' num2str(compos(1))]);
%             for sample = 1:size(s_eeg.etc.analysis.erp.(erp_type).visual.comps,2)
%                 erp_sample_vis = squeeze(s_eeg.etc.analysis.erp.(erp_type).visual.comps(compos(1),sample,:));
%                 erp_sample_vibro = squeeze(s_eeg.etc.analysis.erp.(erp_type).vibro.comps(compos(1),sample,:));
%                 erp_sample = [erp_sample_vis; erp_sample_vibro];
% 
%                 design = table(erp_sample, immersion, vel, trial, direction, sequence);
%                 if robustfit
%                     mdl = fitlm(design, model, 'RobustOpts', 'on');
%                 else
%                     mdl = fitlm(design, model);
%                 end
% 
%                 res.betas(count,sample,:) = mdl.Coefficients.Estimate;
%                 res.t(count,sample,:) = mdl.Coefficients.tStat;
%                 res.p(count,sample,:) = mdl.Coefficients.pValue;
%                 res.r2(count,sample,:) = mdl.Rsquared.Ordinary;
%                 res.adj_r2(count,sample,:) = mdl.Rsquared.Adjusted;
%             end
%             count = count + 1;
%         end
% 
%         %TO SAVE: statistics and design info
%         % add parameter names
%         res.timepoint_before_touch = ts;
%         res.event_onset = event_onset;
%         res.model = model;
%         res.parameter_names = mdl.CoefficientNames;
%         this_ts = (event_onset - ts) / 250;
%         save([save_fpath '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event.mat'], 'res');
%         clear res
%         disp(['fitting took: ' num2str(toc/250) ' minutes']);
% 
%     end
    
    %% channels
    for chan = channels_of_int
    
        disp(['Now running analysis for channel: ' num2str(chan)]);

        % outpath

        save_fpath = [bemobil_config.study_folder bemobil_config.study_level 'analyses/erp/' bemobil_config.study_filename(1:end-6) '/channel_' num2str(chan)];
        
        if ~exist(save_fpath, 'dir')
            mkdir(save_fpath);
        end        

        % load data
        count = 1;
        for subject = subjects

            % select EEG dataset
            s_eeg = ALLEEG(count);

            % loop through all vels and accs at different time points before the
            % event
            model = 'erp_sample ~ immersion * vel + trial + direction + sequence';

            %DESIGN make continuous and dummy coded predictors
            vel_vis = s_eeg.etc.analysis.mocap.visual.mag_vel(ts,:)'; % correct for age-of-sample
            vel_vibro = s_eeg.etc.analysis.mocap.vibro.mag_vel(ts,:)';

            vel = zscore([vel_vis; vel_vibro]);
            immersion = categorical([zeros(size(vel_vis,1),1); ones(size(vel_vibro,1),1)]);
            trial = zscore([s_eeg.etc.epoching.visual_tr_num, s_eeg.etc.epoching.vibro_tr_num]');
            direction = categorical([s_eeg.etc.epoching.visual_dir, s_eeg.etc.epoching.vibro_dir]');
            sequence = zscore([s_eeg.etc.epoching.visual_match_seq, s_eeg.etc.epoching.vibro_match_seq]');

            % now fit linear model for each component
            disp(['running lm for subject ' num2str(subject) ' and chan ' num2str(chan)]);
            for sample = 1:size(s_eeg.etc.analysis.erp.(erp_type).visual.chans,2)
                erp_sample_vis = squeeze(s_eeg.etc.analysis.erp.(erp_type).visual.chans(chan,sample,:));
                erp_sample_vibro = squeeze(s_eeg.etc.analysis.erp.(erp_type).vibro.chans(chan,sample,:));
                erp_sample = [erp_sample_vis; erp_sample_vibro];

                design = table(erp_sample, immersion, vel, trial, direction, sequence);
                if robustfit
                    mdl = fitlm(design, model, 'RobustOpts', 'on');
                else
                    mdl = fitlm(design, model);
                end

                res.betas(count,sample,:) = mdl.Coefficients.Estimate;
                res.t(count,sample,:) = mdl.Coefficients.tStat;
                res.p(count,sample,:) = mdl.Coefficients.pValue;
                res.r2(count,sample,:) = mdl.Rsquared.Ordinary;
                res.adj_r2(count,sample,:) = mdl.Rsquared.Adjusted;
            end
            count = count + 1;
        end

        %TO SAVE: statistics and design info
        % add parameter names
        res.timepoint_before_touch = ts;
        res.event_onset = event_onset;
        res.model = model;
        res.parameter_names = mdl.CoefficientNames;
        this_ts = (event_onset - ts) / 250;
        save([save_fpath '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event.mat'], 'res');
        clear res
    end
end

%% TODO:confirm signficance of main effects across subjects

% settings
alpha = .05;
measure = 'erp';
save_info.robustfit = 0;
save_info.model = [measure '_sample ~ immersion * vel + trial + direction + sequence'];
save_info.parameters = {'vel', 'immersion_1', 'trial', 'sequence', 'immersion_1:vel'};

% res.parameter_names'
% 
% ans =
% 
%   8�1 cell array
% 
%     {'(Intercept)'     }
%     {'immersion_1'     }
%     {'vel'             }
%     {'trial'           }
%     {'direction_middle'}
%     {'direction_right' }
%     {'sequence'        }
%     {'immersion_1:vel' }

save_info.sensor = 'cluster'; % cluster
% try out several ts prior to event
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);
seconds_before_event = .3;
samples_before_event = seconds_before_event * 250;
ts_of_ints = (event_onset-samples_before_event):25:event_onset+50;
% ts_of_ints = ts_of_ints - mocap_aos_samples;
% select best tf_of_ints
save_info.this_ts = ts_of_ints(4) - abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);

for p = save_info.parameters
    save_info.parameter = p{1};
    for c = clusters_of_int
        
        %% select data
        save_info.load_p = [bemobil_config.study_folder bemobil_config.study_level 'analyses/' measure '/' bemobil_config.study_filename(1:end-6) '/' save_info.sensor '_' num2str(c)];
        save_info.c = c;
        load([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '_vel-at-' num2str(save_info.this_ts) 'ms-pre-event.mat'])

        %% run limo ttest against 0
        PE_limo(save_info, res, 0, [], [], [], [], []);

        % load LIMO output: save mean value and sig. mask to res and resave res
        load([save_info.load_p '/ttest_' save_info.parameter '/one_sample_ttest_parameter_1.mat']);
        save_name = regexprep(save_info.parameter, ':' , '_');
        res.ttest.(save_name).mean_value = squeeze(one_sample(1,:,1)); % mean betas in this case

        % mcc
        % load bootstrap results
        load([save_info.load_p '/ttest_' save_info.parameter '/H0/tfce_H0_one_sample_ttest_parameter_1.mat']);
        % get max dist of tfce
        for i = 1:size(tfce_H0_one_sample,3)
            this_tfce = squeeze(tfce_H0_one_sample(1,:,i));
            max_dist(i) = max(this_tfce(:));
        end
        % threshold
        thresh = prctile(max_dist, (1-alpha)*100);
        % load true result
        load([save_info.load_p '/ttest_' save_info.parameter '/tfce/tfce_one_sample_ttest_parameter_1.mat']);
        % threshold true data with bootstrap prctile thresh
        res.ttest.(save_name).sig_mask = find(tfce_one_sample>thresh);

        % resave res
        save([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '_vel-at-' num2str(save_info.this_ts) 'ms-pre-event.mat'], 'res');
        
    end
end