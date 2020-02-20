% component erp single-trial regression


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