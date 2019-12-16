%% clear all and load params
close all; clear

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

%% effect of velocity on ERSP per subject and condition, i.e. pERSP, then
% average betas across subjects for each condition

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

robustfit = 0;

% try out several ts prior to event
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);
seconds_before_event = .3;
samples_before_event = seconds_before_event * 250;
ts_of_ints = (event_onset-samples_before_event):25:event_onset+50;
% ts_of_ints = ts_of_ints - mocap_aos_samples;

% select best tf_of_ints
ts_of_ints = ts_of_ints(4);
ersp_type = 'tfdata';

what_c = 'channel'; % cluster
cs_of_int = channels_of_int(2:end);

for ts = ts_of_ints
    
    % components
    for c = cs_of_int

        disp(['Now running analysis for c: ' num2str(c)]);
        tic

        % outpath
        save_fpath = [bemobil_config.study_folder bemobil_config.study_level 'analyses/ersp/' bemobil_config.study_filename(1:end-6) '/' what_c '_' num2str(c)];
        if ~exist(save_fpath, 'dir')
            mkdir(save_fpath);
        end        

        %% get matching datasets from EEGLAB Study struct
        if strcmp(what_c, 'cluster')
            unique_setindices = unique(STUDY.c(c).sets);
            unique_subjects = STUDY_sets(unique_setindices);
            all_setindices = STUDY.c(c).sets;
            all_sets = STUDY_sets(all_setindices);
            all_comps = STUDY.c(c).comps;
            subs = unique_subjects
        else
            subs = subjects;
        end

        % load IC data
        count = 1;
        for subject = subs

            % select EEG dataset
            [~, ix] = find(subject==subjects);
            s_eeg = ALLEEG(ix);

            % loop through all vels and accs at different time points before the
            % event
            model = 'ersp_sample ~ immersion * vel + trial + direction + sequence';

            %DESIGN make continuous and dummy coded predictors
            vel_vis = s_eeg.etc.analysis.mocap.visual.mag_vel(ts,:)'; % correct for age-of-sample
            vel_vibro = s_eeg.etc.analysis.mocap.vibro.mag_vel(ts,:)';

            vel = zscore([vel_vis; vel_vibro]);
            immersion = categorical([zeros(size(vel_vis,1),1); ones(size(vel_vibro,1),1)]);
            trial = zscore([s_eeg.etc.epoching.visual_tr_num, s_eeg.etc.epoching.vibro_tr_num]');
            direction = categorical([s_eeg.etc.epoching.visual_dir, s_eeg.etc.epoching.vibro_dir]');
            sequence = zscore([s_eeg.etc.epoching.visual_match_seq, s_eeg.etc.epoching.vibro_match_seq]');
            
            % old predictor_time = [1:size(vel_vis), 1:size(vel_vibro)]';

            
            % select 1st comp if more than 1 comp per subject
            if strcmp(what_c, 'cluster')
                compos = all_comps(all_sets==subject);
                if compos > 1
    %                 % averaging
    %                 s_eeg.etc.analysis.ersp.base_corrected_dB.visual(compos(1),:,:,:) = ...
    %                     mean(s_eeg.etc.analysis.ersp.base_corrected_dB.visual(compos,:,:,:),1);
    %                 s_eeg.etc.analysis.ersp.base_corrected_dB.vibro(compos(1),:,:,:) = ...
    %                     mean(s_eeg.etc.analysis.ersp.base_corrected_dB.vibro(compos,:,:,:),1);

                    % selecting first index
                    c = compos(1);
                end
            end

            % now fit linear model for each component
            % after averaging take one IC per subject in c
            disp(['running lm for subject ' num2str(subject) ' and ' what_c ' ' num2str(c)]);
            
            % for each time frequency pixel
            for t = 1:size(s_eeg.etc.analysis.ersp.(ersp_type).visual.(what_c),2)
                for f = 1:size(s_eeg.etc.analysis.ersp.(ersp_type).visual.(what_c),3)
                    
                    ersp_sample_vis = squeeze(s_eeg.etc.analysis.ersp.(ersp_type).visual.(what_c)(c,t,f,:));
                    ersp_sample_vibro = squeeze(s_eeg.etc.analysis.ersp.(ersp_type).vibro.(what_c)(c,t,f,:));
                    ersp_sample = [ersp_sample_vis; ersp_sample_vibro];

                    design = table(ersp_sample, immersion, vel, trial, direction, sequence);
                    if robustfit
                        mdl = fitlm(design, model, 'RobustOpts', 'on');
                    else
                        mdl = fitlm(design, model);
                    end

                    res.betas(count,t,f,:) = mdl.Coefficients.Estimate;
                    res.t(count,t,f,:) = mdl.Coefficients.tStat;
                    res.p(count,t,f,:) = mdl.Coefficients.pValue;
                    res.r2(count,t,f,:) = mdl.Rsquared.Ordinary;
                    res.adj_r2(count,t,f,:) = mdl.Rsquared.Adjusted;
                end
                
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
        disp(['fitting took: ' num2str(toc/250) ' minutes']);

    end
end

%% signficance of main effects across subjects

% settings
alpha = .05;
measures = {'ersp'}; %, 'erp'};
save_info.robustfit = 0;
save_info.parameters = {'immersion_1', 'vel', 'trial', 'sequence', 'immersion_1:vel'};
% save_info.parameters = {'mean'};
save_info.sensors = {'channel'}; %  'cluster', 

% res.parameter_names'
% 
% ans =
% 
%   8×1 cell array
% 
%     {'(Intercept)'     }
%     {'immersion_1'     }
%     {'vel'             }
%     {'trial'           }
%     {'direction_middle'}
%     {'direction_right' }
%     {'sequence'        }
%     {'immersion_1:vel' }

% select time and freq limits
% load times and freqs
load_path = [bemobil_config.study_folder bemobil_config.study_level 'analyses/' measure];

if strcmp(measure, 'ersp')
    load([load_path, '/times.mat']);
    load([load_path '/times_all.mat']);
    load([load_path '/freqs.mat']);
    first_ix = find(times_all==times(1));
    last_ix = find(times_all==times(end));
    times_ixs = [first_ix, last_ix];
    max_freq_ix = find(freqs>=40,1,'first');
    freqs = freqs(1:max_freq_ix);
end

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
    
    for m = measures
        measure = m{1};
        
        save_info.model = [measure '_sample ~ immersion * vel + trial + direction + sequence'];
    
        for s = save_info.sensors
            save_info.sensor = s{1};

            if strcmp(save_info.sensor, 'cluster')
                cs = clusters_of_int;
            else
                cs = channels_of_int;
            end

            for c =  cs

                %% get matching datasets from EEGLAB Study struct

                if strcmp(save_info.sensor, 'cluster')
                    unique_setindices = unique(STUDY.cluster(c).sets);
                    unique_subjects = STUDY_sets(unique_setindices);
                    all_setindices = STUDY.cluster(c).sets;
                    all_sets = STUDY_sets(all_setindices);
                    all_comps = STUDY.cluster(c).comps;
                end

                %% select data
                save_info.load_p = [bemobil_config.study_folder bemobil_config.study_level 'analyses/' measure '/' bemobil_config.study_filename(1:end-6) '/' save_info.sensor '_' num2str(c)];
                save_info.c = c;
                load([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '_vel-at-' num2str(save_info.this_ts) 'ms-pre-event.mat'])

                %% run limo ttest/regression
                if strcmp(measure,'ersp')
                    PE_limo(save_info, res, 1, times, times_ixs, freqs, max_freq_ix, []);
                else
                    PE_limo(save_info, res, 0, [], [], [], [], []);
                end

%                 % load LIMO output: save mean value and sig. mask to res and resave res
%                 load([save_info.load_p '/regress_' save_info.parameter '/Betas.mat']);
%                 save_name = regexprep(save_info.parameter, ':' , '_');
%                 res.regress_IPQ.(save_name).mean_value = squeeze(Betas(1,:,2)); % mean betas in this case

                % mcc
                % load bootstrap results
                load([save_info.load_p '/ttest_' save_info.parameter '/H0/tfce_H0_one_sample_ttest_parameter_1.mat']);
                % get max dist of tfce
                for i = 1:size(tfce_H0_one_sample,3)
                    if strcmp(measure,'ersp')
                        this_tfce = squeeze(tfce_H0_one_sample(:,:,i));
                    else
                        this_tfce = squeeze(tfce_H0_one_sample(:,i));
                    end
                    max_dist(i) = max(this_tfce(:));
                end
                % remove NaN
                max_dist(isnan(max_dist)) = [];
                % threshold
                thresh = prctile(max_dist, (1-alpha)*100);
                % load true result
                load([save_info.load_p '/ttest_' save_info.parameter '/tfce/tfce_one_sample_ttest_parameter_1.mat']);
                % threshold true data with bootstrap prctile thresh
                if strcmp(measure,'ersp')
                    save_name = regexprep(save_info.parameter, ':' , '_');
                    res.ttest.(save_name).tfce_map = squeeze(tfce_one_sample);
                    res.ttest.(save_name).thresh = prctile(max_dist, (1-alpha)*100);
                end

                % resave res
                save([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '_vel-at-' num2str(save_info.this_ts) 'ms-pre-event.mat'], 'res');

            end
        end
    end
end