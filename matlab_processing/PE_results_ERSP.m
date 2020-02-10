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

%% TODO:LIMO_LOOP single-trial regression first-level and group level statistics : congruency

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

% get betas from single subject level of design:
models = {'ersp_sample ~ congruency * haptics + trial_nr + direction + sequence',...
    'ersp_sample ~ congruency * haptics + trial_nr + direction + sequence + base'};
robustfit = 0; % fit robust regression with squared weights, see fitlm
event_sample = 750;
window = event_sample-25:event_sample+200; %[-.1 .8]seconds start and end of interesting, to be analyzed, samples
count = 1;


for cluster = 11 % clusters_of_int
    for model = models
        model = model{1};
        
        % outpath
        save_fpath = [bemobil_config.study_folder bemobil_config.study_level ...
            'analyses/ersp/' bemobil_config.study_filename(1:end-6) ...
            '/cluster_' num2str(cluster) '/' model];
        if ~exist(save_fpath, 'dir')
            mkdir(save_fpath);
        end        

        % get matching datasets from EEGLAB Study struct
        unique_setindices = unique(STUDY.cluster(cluster).sets);
        unique_subjects = STUDY_sets(unique_setindices);
        all_setindices = STUDY.cluster(cluster).sets;
        all_sets = STUDY_sets(all_setindices);
        all_comps = STUDY.cluster(cluster).comps;
        subjects = unique_subjects;

        % fit model to IC data
        count = 1;
        for subject = subjects
            
            % select EEG dataset
            [~, ix] = find(subject==subjects);
            s = ALLEEG(ix);

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

            % select 1st comp if more than 1 comp per subject
            comp = all_comps(all_sets==subject);
            if comp > 1
    %             % averaging
    %             s_eeg.etc.analysis.ersp.base_corrected_dB.visual(compos(1),:,:,:) = ...
    %                 mean(s_eeg.etc.analysis.ersp.base_corrected_dB.visual(compos,:,:,:),1);
    %             s_eeg.etc.analysis.ersp.base_corrected_dB.vibro(compos(1),:,:,:) = ...
    %                 mean(s_eeg.etc.analysis.ersp.base_corrected_dB.vibro(compos,:,:,:),1);

                % selecting first index
                comp = comp(1);
            end

            % fitlm for each time frequency pixel
            tic
            disp(['Now running analysis for cluster: ' num2str(cluster) ' and subject: ' num2str(subject) ' with component: ' num2str(comp)]);
            for t = 1:size(s.etc.analysis.ersp.times,2)
                for f = 1:size(s.etc.analysis.ersp.freqs,2)

                    % add ersp and baseline sample to design matrix
                    ersp_sample = squeeze(s.etc.analysis.ersp.tfdata.comp(comp,f,t,:));
                    base = squeezemean(s.etc.analysis.ersp.tfdata.comp(comp,f,1:19,:),3);
                    
                    if contains(model,'velocity')
                        ersp_sample = ersp_sample(congruency,:);
                        base = base(congruency,:);
                        design = table(ersp_sample, haptics, velocity, trial_nr, direction, sequence, base); % design matrix per sample
                    else
                        design = table(ersp_sample, congruency, haptics, trial_nr, direction, sequence, base); % design matrix per sample
                    end
                    
                    if robustfit
                        mdl = fitlm(design, model, 'RobustOpts', 'on');
                    else
                        mdl = fitlm(design, model);
                    end

                    res.betas(count,f,t,:) = mdl.Coefficients.Estimate;
                    res.t(count,f,t,:) = mdl.Coefficients.tStat;
                    res.p(count,f,t,:) = mdl.Coefficients.pValue;
                    res.r2(count,f,t,:) = mdl.Rsquared.Ordinary;
                    res.adj_r2(count,f,t,:) = mdl.Rsquared.Adjusted;
                end
            end
            toc
            count = count + 1;
        end
        
        % add parameter names
        res.model = model;
        res.parameter_names = string(mdl.CoefficientNames)';
        save([save_fpath '/res_' model '_robust-' num2str(robustfit) '.mat'], 'res');

%         % exemplary plot
%         figure;
%         for i = 1:9
%             subplot(3,3,i)
%             dat = squeeze(res.p(1,:,:,i));
%             lims = max(abs(dat(:))) * [-1 1];
%             imagesclogy(times, freqs, dat, [0 .05]); axis xy; xline(0); 
%             title(res.parameter_names{i}); cbar;
%         end

        % LIMO
        % settings
        sig_alpha = .05;
        save_info.robustfit = 0;
        save_info.model = model;
        
        % load times and freqs
        load_path = [bemobil_config.study_folder bemobil_config.study_level 'analyses/ersp'];
        times = s.etc.analysis.ersp.times;
        times_ixs = [1, size(times,2)];
        freqs = s.etc.analysis.ersp.freqs;
        max_freq_ix = freqs(end);
        
        % select model
        if contains(model, 'velocity')
            save_info.parameters = {'(Intercept)', 'haptics_1', 'velocity', 'haptics_1:velocity'}; %
        else
            save_info.parameters = {'(Intercept)', 'congruency_1', 'haptics_1', 'congruency_1:haptics_1'}; %
        end
        save_info.load_p = save_fpath;
        
        % main effects: run limo ttest against 0
        for param = save_info.parameters
            save_info.parameter = param{1};
            
            % run limo
            PE_limo(save_info, res, 1, times, times_ixs, freqs, max_freq_ix, []);

            % load LIMO output: save mean value and sig. mask to res and resave res
            save_name = regexprep(save_info.parameter, ':' , '_');
            save_name = regexprep(save_name, '(' , '');
            save_name = regexprep(save_name, ')' , '');
            
            % todo what is the correct name here
            load([save_info.load_p '/ttest_' save_name '/one_sample_ttest_parameter_1.mat']);
            %res.ttest.(save_name).mean_value = squeeze(one_sample(1,:,1)); % mean betas in this case
            
            % mcc
            % load bootstrap results
            load([save_info.load_p '/ttest_' save_name '/H0/tfce_H0_one_sample_ttest_parameter_1.mat']);
            % get max dist of tfce
            for i = 1:size(tfce_H0_one_sample,3)
                this_tfce = squeeze(tfce_H0_one_sample(:,:,i));
                max_dist(i) = max(this_tfce(:));
            end
            % remove NaN
            max_dist(isnan(max_dist)) = [];
            % threshold
            thresh = prctile(max_dist, (1-sig_alpha)*100);
            % load true result
            load([save_info.load_p '/ttest_' save_name '/tfce/tfce_one_sample_ttest_parameter_1.mat']);
            % threshold true data with bootstrap prctile thresh
            res.ttest.(save_name).tfce_map = squeeze(tfce_one_sample);
            res.ttest.(save_name).thresh = prctile(max_dist, (1-sig_alpha)*100);
            res.ttest.(save_name).sig_mask = res.ttest.(save_name).tfce_map>res.ttest.(save_name).thresh;
            
            % save res
            save([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '.mat'], 'res');
        end
        
        % clear results struct
        clear res
    end
end