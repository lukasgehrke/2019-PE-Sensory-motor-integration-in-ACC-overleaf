%% clear all and load params
clear all;

if ~exist('ALLEEG','var')
	eeglab;
end

% add downloaded analyses code to the path
addpath(genpath('/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing'));
% TODO add to path bemobil_pipeline repository download folder
% TODO add to path custom scripts repository Lukas Gehrke folder

% BIDS data download folder
bemobil_config.BIDS_folder = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/ds003552';
% Results output folder -> external drive
bemobil_config.study_folder = fullfile('/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error', 'derivatives');

% init
config_processing_pe;
subjects = 1:19;

%% load study

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 1, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

if isempty(STUDY)
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    [STUDY ALLEEG] = pop_loadstudy('filename', ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_' bemobil_config.study_filename], 'filepath', bemobil_config.study_folder);
    CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];    
    eeglab redraw
end
STUDY_sets = cellfun(@str2num, {STUDY.datasetinfo.subject});

%% export for paraheat 22/10/2020, corrected 04/11/2020

% get participant descriptive: age, ipq, biological_sex
ipq = table2array(readtable('/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/admin/ipq_long.csv'));
p_desc = ipq(1:4:end,:,:); % select only item 1 on questionnaire which is the general presence item
p_desc(:,end+1) = table2array(readtable('/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/admin/age.csv'));
% p_desc(:,end+1) = table2array(readtable('/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/admin/biological_sex.csv'));
p_desc_vars = {'pID', 'ipq_vibro', 'ipq_visual', 'age'};

% standardize the mocap data on the starting location for each trial?
s_design = [];
samples_after_touch = 250 * 3/10; % 300ms post touch

for i = 1:size(ALLEEG,2)
    
    % first remove bad trials and keep only congruent trials -> simplifies analyses
    keep_trials = logical(abs(ALLEEG(i).etc.analysis.design.oddball-1));
    keep_trials(ALLEEG(i).etc.analysis.design.rm_ixs) = [];
    tmp_trials = 1:size(ALLEEG(i).etc.analysis.design.trial_number,2);
    tmp_trials(ALLEEG(i).etc.analysis.design.rm_ixs) = [];
    trials = tmp_trials(keep_trials);
    
    rt = ALLEEG(i).etc.analysis.design.rt_spawned_touched;
    rt = rt(trials);
    start_sample = 750 - round(rt*250);
    total_trials = size((ALLEEG(i).etc.analysis.design.trial_number),2);
    nan_trials = nan(size(start_sample,2),total_trials, 4);
    
    for j = 1:size(start_sample,2)
        
        x = ALLEEG(i).etc.analysis.mocap.x(1,start_sample(j):751,j) - ALLEEG(i).etc.analysis.mocap.x(1,start_sample(1),j);
        z = ALLEEG(i).etc.analysis.mocap.z(1,start_sample(j):751,j) - ALLEEG(i).etc.analysis.mocap.z(1,start_sample(1),j);
        mag_vel = ALLEEG(i).etc.analysis.mocap.mag_vel(start_sample(j):751,j);
        fcz_erp = ALLEEG(i).etc.analysis.filtered_erp.chan(4,start_sample(j):751,j);
        
        nan_trials(j,total_trials+1-size(start_sample(j):751,2):end,1) = x;
        nan_trials(j,total_trials+1-size(start_sample(j):751,2):end,2) = z;
        nan_trials(j,total_trials+1-size(start_sample(j):751,2):end,3) = mag_vel;
        nan_trials(j,total_trials+1-size(start_sample(j):751,2):end,4) = fcz_erp;
        
    end

    x = nan_trials(:,:,1)';
    z = nan_trials(:,:,2)';
    mag_vel = nan_trials(:,:,3)';
    erp = nan_trials(:,:,4)';

    haptics = ALLEEG(i).etc.analysis.design.haptics;
    haptics = repelem(haptics(trials),size(nan_trials,2))';
    
    tr_nr = ALLEEG(i).etc.analysis.design.trial_number;
    tr_nr = repelem(tr_nr(keep_trials),size(nan_trials,2))';

    % add participant descriptives
    pID = ones(size(tr_nr,1),1) * p_desc(i,1);
    ipq_vibro = repmat(p_desc(i,2), size(tr_nr,1), 1);
    ipq_vis = repmat(p_desc(i,3), size(tr_nr,1), 1);
    age = repmat(p_desc(i,4), size(tr_nr,1), 1);

    p_design = table(pID, ipq_vibro, ipq_vis, age, tr_nr, haptics, x(:), z(:), mag_vel(:), erp(:), 'VariableNames', {p_desc_vars{1}, p_desc_vars{2}, p_desc_vars{3}, p_desc_vars{4}, 'TrialNr', 'Haptics', 'X', 'Z', 'Mag_Vel', 'FCz_ERP'});
    
    % remove nans and exceeding edges
    p_design = rmmissing(p_design);

    p_design(p_design.X > 1.5, :)=[];
    p_design(p_design.X < -1.5, :)=[];
    p_design(p_design.Z > 1.5, :)=[];
    p_design(p_design.Z < -.2, :)=[];
    
    s_design = [s_design; p_design];
end

% save to csv for paraheat
writetable(s_design, '/Users/lukasgehrke/Documents/temp/chatham/pe_reach_all_good_s.csv');
