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

%% FINAL single trial regression first-leveland group level statistics
% model 1: congruency
% model 2: mismatch trials only: velocity * haptics

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

% get betas from single subject level of design:
models = {'erp_sample ~ congruency * haptics + trial_nr + direction + sequence', 'erp_sample ~ velocity * haptics + trial_nr + direction + sequence'};

robustfit = 0; % fit robust regression with squared weights, see fitlm
event_sample = 750;
window = event_sample-25:event_sample+200; %[-.1 .8]seconds start and end of interesting, to be analyzed, samples
count = 1;

for model = models
    model = model{1};

    % channels
    for chan = channels_of_int

        % outpath
        save_fpath = [bemobil_config.study_folder bemobil_config.study_level ...
            'analyses/erp/' bemobil_config.study_filename(1:end-6) ...
            '/channel_' num2str(chan) '/' model];
        if ~exist(save_fpath, 'dir')
            mkdir(save_fpath);
        end        

        % load data
        count = 1;
        for s = ALLEEG
            disp(['Now running analysis for channel: ' num2str(chan) ' and subject: ' num2str(count+1)]);

            %DESIGN make continuous and dummy coded predictors
            congruency = s.etc.epoching.oddball';
            haptics = s.etc.epoching.haptics';
            trial_nr = s.etc.epoching.trial_number';
            direction = categorical(s.etc.epoching.direction)';
            sequence = s.etc.epoching.sequence';
            
            if contains(model, 'velocity')
                % select mismatch trials only
                congruency = logical(congruency);
                haptics = haptics(congruency);
                trial_nr = trial_nr(congruency);
                direction = direction(congruency);
                sequence = sequence(congruency);                
                % add velocity at moment of collision
                velocity = s.etc.analysis.mocap.mag_vel(event_sample,congruency)';
            end

            tic
            for sample = window

                erp_sample = squeeze(s.etc.analysis.erp.non_baseline_corrected.chans(chan,sample,:));
                if contains(model,'velocity')
                    erp_sample = erp_sample(congruency,:);
                    design = table(erp_sample, haptics, velocity, trial_nr, direction, sequence); % design matrix per sample
                else
                    design = table(erp_sample, congruency, haptics, trial_nr, direction, sequence); % design matrix per sample
                end
                    

                if robustfit
                    mdl = fitlm(design, model, 'RobustOpts', 'on');
                else
                    mdl = fitlm(design, model);
                end

                ix = sample-window(1)+1;
                res.betas(count,ix,:) = mdl.Coefficients.Estimate;
                res.t(count,ix,:) = mdl.Coefficients.tStat;
                res.p(count,ix,:) = mdl.Coefficients.pValue;
                res.r2(count,ix,:) = mdl.Rsquared.Ordinary;
                res.adj_r2(count,ix,:) = mdl.Rsquared.Adjusted;
            end
            toc
            count = count + 1;
        end

        % add parameter names
        res.model = model;
        res.parameter_names = string(mdl.CoefficientNames)';
        save([save_fpath '/res_' model '_robust-' num2str(robustfit) '.mat'], 'res');

    %     % examplary plots
    %     figure
    %     c = 1;
    %     for i = [1,2,3,8] % 1:size(res.betas,3)
    %         subplot(1,4,c);
    %         c = c+1;
    %         plot(mean(res.betas(:,100:300,i))); xline(25); title(res.parameter_names(i));
    %     end

        %LIMO ttests
        % settings
        sig_alpha = .05;
        save_info.robustfit = 0;
        save_info.model = model;
        % select model
        if contains(model, 'velocity')
            save_info.parameters = {'(Intercept)', 'haptics_1', 'velocity', 'haptics_1:velocity'}; %
        else
            save_info.parameters = {'(Intercept)', 'congruency_1', 'haptics_1', 'congruency_1:haptics_1'}; %
        end
        %save_info.load_p = ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/erp/chan_' num2str(chan)];
        save_info.load_p = save_fpath;

        % main effects: run limo ttest against 0
        for param = save_info.parameters
            save_info.parameter = param{1};
            PE_limo(save_info, res, 0, [], [], [], [], []);

            % load LIMO output: save mean value and sig. mask to res and resave res
            save_name = regexprep(save_info.parameter, ':' , '_');
            save_name = regexprep(save_name, '(' , '');
            save_name = regexprep(save_name, ')' , '');
            load([save_info.load_p '/ttest_' save_name '/one_sample_ttest_parameter_1.mat']);
            res.ttest.(save_name).mean_value = squeeze(one_sample(1,:,1)); % mean betas in this case

            % mcc
            % load bootstrap results
            load([save_info.load_p '/ttest_' save_name '/H0/tfce_H0_one_sample_ttest_parameter_1.mat']);
            % get max dist of tfce
            for i = 1:size(tfce_H0_one_sample,3)
                this_tfce = squeeze(tfce_H0_one_sample(1,:,i));
                max_dist(i) = max(this_tfce(:));
            end
            % threshold
            thresh = prctile(max_dist, (1-sig_alpha)*100);
            % load true result
            load([save_info.load_p '/ttest_' save_name '/tfce/tfce_one_sample_ttest_parameter_1.mat']);
            % threshold true data with bootstrap prctile thresh
            res.ttest.(save_name).sig_mask = find(tfce_one_sample>thresh);

            % resave res
            save([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '.mat'], 'res');
        end
        
        % clear results struct
        clear res
    end
end
