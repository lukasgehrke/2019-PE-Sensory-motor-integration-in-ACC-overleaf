%% clear all and load params
close all; clear all;

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

%% FINAL result 1.1: match velocity ERP with confidence interval and sig. pixel (LIMO one-sample ttest)

begin_ms = -2400;
end_ms = 1400;

% make xtlabels
all_samples = 1250;
xtlabels = 1:all_samples;
xtlabels = (xtlabels / ALLEEG(1).srate) + bemobil_config.epoching.event_epochs_boundaries(1);
xtlabels = round(xtlabels*1000);

% find nearest matching index
begin_sample = find(ismember(xtlabels,begin_ms));
end_sample = find(ismember(xtlabels,end_ms));
xtlabels = xtlabels(begin_sample:end_sample);

c = 1;
mismatch_all_subjects = [];
mismatch_rts = [];
match_all_subjects = [];
match_rts = [];
difference = [];

for s = ALLEEG
    bad_trs = union(s.etc.analysis.ersp.rm_ixs, s.etc.analysis.erp.rm_ixs);
    oddball = s.etc.analysis.design.oddball;
    oddball(bad_trs) = [];
    
    rt = s.etc.analysis.design.rt_spawned_touched;
    rt(bad_trs) = [];
    mismatch_rts(c,:) = mean(rt(oddball),2);
    match_rts(c,:) = mean(rt(~oddball),2);
    
    % select timeseries measure
    measure = s.etc.analysis.mocap.mag_vel;
    measure(:,bad_trs) = [];
    match_all_subjects(c,:) = squeeze(nanmean(measure(begin_sample:end_sample,~oddball),2));
    c = c+1;
end

this_measure = match_all_subjects;

mean_measure = mean(this_measure,1);
% Calculate Standard Error Of The Mean
SEM_A = std(this_measure', [], 2)./ sqrt(size(this_measure',2));
% 95% Confidence Intervals
CI95 = bsxfun(@plus, mean(this_measure',2), bsxfun(@times, [-1  1]*1.96, SEM_A));
upper = CI95(:,2)';
lower = CI95(:,1)';

% statistics
% hack so can use PE_limo
sig_alpha = .001;
save_info.load_p = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing';
res.betas = match_all_subjects;
PE_limo(save_info, res, 0, [], [], [], [], []);
save_name = 'grand_mean';
load([save_info.load_p '/ttest_' save_name '/one_sample_ttest_parameter_1.mat']);
res.ttest.(save_name).mean_value = squeeze(one_sample(1,:,1)); % mean betas in this case
% multiple comparison correction
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
res.ttest.(save_name).tfce = tfce_one_sample;
% threshold true data with bootstrap prctile thresh
res.ttest.(save_name).sig_mask = find(tfce_one_sample>thresh);
% resave res
%save([save_info.load_p '/res_vel_' '.mat'], 'res');

% make plot
map = brewermap(2,'Set1'); 
figure('Renderer', 'painters', 'Position', [10 10 1000 200]);
p = patch([xtlabels fliplr(xtlabels)], [lower fliplr(upper)], 'g');
p.FaceColor = map(1,:);
p.FaceAlpha = .5;
p.EdgeColor = 'none';
hold on;
l = plot(xtlabels, mean_measure);
l.MarkerFaceColor = map(2,:);
l.LineWidth = 3;
% add significant pixels to plot
if ~isempty(res.ttest.(save_name).sig_mask)
    scatter(xtlabels(res.ttest.(save_name).sig_mask), zeros(1, size(res.ttest.(save_name).sig_mask,2)), 25, map(1,:), 'filled', 's');
end
% format plot
axis tight
% grid on
set(gca,'FontSize',20)
ylabel('velocity (m/s)');
xlabel('seconds');
ylim([0, .8]);
  
%% FINAL result 1.2: mismatch velocity ERP with confidence interval and sig. pixel (LIMO one-sample ttest)

begin_ms = -2400;
end_ms = 1400;

% make xtlabels
all_samples = 1250;
xtlabels = 1:all_samples;
xtlabels = (xtlabels / ALLEEG(1).srate) + bemobil_config.epoching.event_epochs_boundaries(1);
xtlabels = round(xtlabels*1000);

% find nearest matching index
begin_sample = find(ismember(xtlabels,begin_ms));
end_sample = find(ismember(xtlabels,end_ms));
xtlabels = xtlabels(begin_sample:end_sample);

c = 1;
mismatch_all_subjects = [];
mismatch_rts = [];
match_all_subjects = [];
match_rts = [];
difference = [];

for s = ALLEEG
    bad_trs = union(s.etc.analysis.ersp.rm_ixs, s.etc.analysis.erp.rm_ixs);
    oddball = s.etc.analysis.design.oddball;
    oddball(bad_trs) = [];
    
    % select timeseries measure
    measure = s.etc.analysis.mocap.mag_vel;
    measure(:,bad_trs) = [];
    match_all_subjects(c,:) = squeeze(nanmean(measure(begin_sample:end_sample,oddball),2));
    c = c+1;
end

this_measure = match_all_subjects;
mean_measure = mean(this_measure,1);
% Calculate Standard Error Of The Mean
SEM_A = std(this_measure', [], 2)./ sqrt(size(this_measure',2));
% 95% Confidence Intervals
CI95 = bsxfun(@plus, mean(this_measure',2), bsxfun(@times, [-1  1]*1.96, SEM_A));
upper = CI95(:,2)';
lower = CI95(:,1)';

% statistics
% hack so can use PE_limo
sig_alpha = .001;
save_info.load_p = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing';
res.betas = match_all_subjects;
PE_limo(save_info, res, 0, [], [], [], [], []);
save_name = 'grand_mean';
load([save_info.load_p '/ttest_' save_name '/one_sample_ttest_parameter_1.mat']);
res.ttest.(save_name).mean_value = squeeze(one_sample(1,:,1)); % mean betas in this case
% multiple comparison correction
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
res.ttest.(save_name).tfce = tfce_one_sample;
% threshold true data with bootstrap prctile thresh
res.ttest.(save_name).sig_mask = find(tfce_one_sample>thresh);
% resave res
%save([save_info.load_p '/res_vel_' '.mat'], 'res');

% make plot
map = brewermap(2,'Set1'); 
figure('Renderer', 'painters', 'Position', [10 10 1000 200]);
p = patch([xtlabels fliplr(xtlabels)], [lower fliplr(upper)], 'g');
p.FaceColor = map(1,:);
p.FaceAlpha = .5;
p.EdgeColor = 'none';
hold on;
l = plot(xtlabels, mean_measure);
l.MarkerFaceColor = map(2,:);
l.LineWidth = 3;
% add significant pixels to plot
if ~isempty(res.ttest.(save_name).sig_mask)
    scatter(xtlabels(res.ttest.(save_name).sig_mask), zeros(1, size(res.ttest.(save_name).sig_mask,2)), 25, map(1,:), 'filled', 's');
end
% format plot
axis tight
% grid on
set(gca,'FontSize',20)
ylabel('velocity (m/s)');
xlabel('seconds');
ylim([0, .8]);

%% FINAL result 1.3: match FCz ERP with confidence interval and sig. pixel (LIMO one-sample ttest)

begin_ms = -2400;
end_ms = 1400;

% make xtlabels
all_samples = 1250;
xtlabels = 1:all_samples;
xtlabels = (xtlabels / ALLEEG(1).srate) + bemobil_config.epoching.event_epochs_boundaries(1);
xtlabels = round(xtlabels*1000);

% find nearest matching index
begin_sample = find(ismember(xtlabels,begin_ms));
end_sample = find(ismember(xtlabels,end_ms));
xtlabels = xtlabels(begin_sample:end_sample);

c = 1;
measure = [];

for s = ALLEEG
    bad_trs = union(s.etc.analysis.ersp.rm_ixs, s.etc.analysis.erp.rm_ixs);
    oddball = s.etc.analysis.design.oddball;
    oddball(bad_trs) = [];
    
    % select timeseries measure
    measure = squeeze(s.etc.analysis.erp.data(65,:,:));
    measure(:,bad_trs) = [];
    metric(c,:) = nanmean(measure(begin_sample:end_sample,~oddball),2);
    c = c+1;
end

this_measure = metric;
mean_measure = mean(this_measure,1);
% Calculate Standard Error Of The Mean
SEM_A = std(this_measure', [], 2)./ sqrt(size(this_measure',2));
% 95% Confidence Intervals
CI95 = bsxfun(@plus, mean(this_measure',2), bsxfun(@times, [-1  1]*1.96, SEM_A));
upper = CI95(:,2)';
lower = CI95(:,1)';

% statistics
% hack so can use PE_limo
sig_alpha = .01;
save_info.load_p = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing';
res.betas = metric;
PE_limo(save_info, res, 0, [], [], [], [], []);
save_name = 'grand_mean';
load([save_info.load_p '/ttest_' save_name '/one_sample_ttest_parameter_1.mat']);
res.ttest.(save_name).mean_value = squeeze(one_sample(1,:,1)); % mean betas in this case
% multiple comparison correction
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
res.ttest.(save_name).tfce = tfce_one_sample;
% threshold true data with bootstrap prctile thresh
res.ttest.(save_name).sig_mask = find(tfce_one_sample>thresh);
% resave res
%save([save_info.load_p '/res_vel_' '.mat'], 'res');

% make plot
map = brewermap(2,'Set1'); 
figure('Renderer', 'painters', 'Position', [10 10 1000 200]);
p = patch([xtlabels fliplr(xtlabels)], [lower fliplr(upper)], 'g');
p.FaceColor = map(1,:);
p.FaceAlpha = .5;
p.EdgeColor = 'none';
hold on;
l = plot(xtlabels, mean_measure);
l.MarkerFaceColor = map(2,:);
l.LineWidth = 3;
% add significant pixels to plot
if ~isempty(res.ttest.(save_name).sig_mask)
    scatter(xtlabels(res.ttest.(save_name).sig_mask), repmat(-6,1, size(res.ttest.(save_name).sig_mask,2)), 25, map(1,:), 'filled', 's');
end
% format plot
axis tight
% grid on
set(gca,'FontSize',20)
ylabel('amplitude (\muV)');
xlabel('seconds');
ylim([-6, 6]);
  
%% FINAL result 1.4: mismatch FCz ERP with confidence interval and sig. pixel (LIMO one-sample ttest)

begin_ms = -2400;
end_ms = 1400;

% make xtlabels
all_samples = 1250;
xtlabels = 1:all_samples;
xtlabels = (xtlabels / ALLEEG(1).srate) + bemobil_config.epoching.event_epochs_boundaries(1);
xtlabels = round(xtlabels*1000);

% find nearest matching index
begin_sample = find(ismember(xtlabels,begin_ms));
end_sample = find(ismember(xtlabels,end_ms));
xtlabels = xtlabels(begin_sample:end_sample);

c = 1;
measure = [];

for s = ALLEEG
    bad_trs = union(s.etc.analysis.ersp.rm_ixs, s.etc.analysis.erp.rm_ixs);
    oddball = s.etc.analysis.design.oddball;
    oddball(bad_trs) = [];
    
    % select timeseries measure
    measure = squeeze(s.etc.analysis.erp.data(65,:,:));
    measure(:,bad_trs) = [];
    metric(c,:) = nanmean(measure(begin_sample:end_sample,oddball),2);
    c = c+1;
end

this_measure = metric;
mean_measure = mean(this_measure,1);
% Calculate Standard Error Of The Mean
SEM_A = std(this_measure', [], 2)./ sqrt(size(this_measure',2));
% 95% Confidence Intervals
CI95 = bsxfun(@plus, mean(this_measure',2), bsxfun(@times, [-1  1]*1.96, SEM_A));
upper = CI95(:,2)';
lower = CI95(:,1)';

% statistics
% hack so can use PE_limo
sig_alpha = .01;
save_info.load_p = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing';
res.betas = metric;
PE_limo(save_info, res, 0, [], [], [], [], []);
save_name = 'grand_mean';
load([save_info.load_p '/ttest_' save_name '/one_sample_ttest_parameter_1.mat']);
res.ttest.(save_name).mean_value = squeeze(one_sample(1,:,1)); % mean betas in this case
% multiple comparison correction
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
res.ttest.(save_name).tfce = tfce_one_sample;
% threshold true data with bootstrap prctile thresh
res.ttest.(save_name).sig_mask = find(tfce_one_sample>thresh);

% make plot
map = brewermap(2,'Set1'); 
figure('Renderer', 'painters', 'Position', [10 10 1000 200]);
p = patch([xtlabels fliplr(xtlabels)], [lower fliplr(upper)], 'g');
p.FaceColor = map(1,:);
p.FaceAlpha = .5;
p.EdgeColor = 'none';
hold on;
l = plot(xtlabels, mean_measure);
l.MarkerFaceColor = map(2,:);
l.LineWidth = 3;
% add significant pixels to plot
if ~isempty(res.ttest.(save_name).sig_mask)
    scatter(xtlabels(res.ttest.(save_name).sig_mask), repmat(-6,1, size(res.ttest.(save_name).sig_mask,2)), 25, map(1,:), 'filled', 's');
end
% format plot
axis tight
% grid on
set(gca,'FontSize',20)
ylabel('amplitude (\muV)');
xlabel('seconds');
%ylim([0, .8]);
ylim([-6, 6]);

%% FINAL result 2.1: match ACC ERSP and sig. pixel (LIMO one-sample ttest)

clusters = 10;
plot_ss = 0; % plot single subject ERSP
min_rv = 0; % select IC by min residual variance if more than 1 IC per subject
trials = 'mismatch'; % 'match'

for c = clusters

    u_sets = unique(STUDY.cluster(c).sets);
    comps = STUDY.cluster(c).comps;

    % find ics per set
    plot_ix = 1;
    if plot_ss
        figure;
    end
    for i = u_sets
        
        disp(i);
        set_ixs = find(i==STUDY.cluster(c).sets);
        ic = comps(set_ixs);
        
        % get topovec
        s = ALLEEG(i);
        
%         if size(ic,2) > 1
%             %topo_vec(i,:) = mean(s.icawinv(:,ic),2);
%             topo_vec(i,:) = s.icawinv(:,ic(1));
%         else
%             topo_vec(i,:) = s.icawinv(:,ic);
%         end
        
        % average over ics or select ic with min rv in case of more than 1
        % IC per subject
        if min_rv && size(ic,2) > 1
            choose_rv = find(STUDY.cluster(c).residual_variances(set_ixs)==min(STUDY.cluster(c).residual_variances(set_ixs)));
            ic = ic(choose_rv);
        end
        
        % accress ersp data
        s = ALLEEG(i);
        % reject bad trials
        bad_trs = s.etc.analysis.ersp.rm_ixs;
        oddball = s.etc.analysis.design.oddball;
        oddball(bad_trs) = [];
        
        if strcmp(trials, 'mismatch')
            tr_ix = oddball;
        elseif strcmp(trials, 'match')
            tr_ix = ~oddball;
        end            
 
        % get mean times of events
        %rt(i) = mean(s.etc.analysis.design.rt_spawned_touched(tr_ix));
        % trials taken below considers all trials so a but wrong (should not matter much)
        %start(i) = mean(s.etc.analysis.design.isitime(tr_ix));
        data = s.etc.analysis.ersp.tf_event_raw_power(ic,:,:,:);
        base = s.etc.analysis.ersp.tf_base_raw_power(ic,:,:);
        data(:,:,:,bad_trs) = [];
        base(:,:,bad_trs) = [];
        
        % prepare ersp data
        data = squeezemean(data(:,:,:,tr_ix),4);
        base = squeezemean(base(:,:,tr_ix),3)';
        if ~min_rv && size(ic,2) > 1
            data = squeezemean(data,1);
            base = squeezemean(base,2);
        end
        
        data_grand(i,:,:) = data;
        base_grand(i,:) = base;
        
        if plot_ss
            data_base_db = 10.*log10(data ./ base);
            lims = max(abs(data_base_db(:)))/2 * [-1 1];
            
            subplot(size(u_sets,2),2,plot_ix);
            plot_ix = plot_ix + 1;
            
            if ~min_rv && size(ic,2) > 1
                topo_vec = mean(s.icawinv(:,ic),2);
            else
                topo_vec = s.icawinv(:,ic);
            end
            topoplot(topo_vec, s.chanlocs);

            subplot(size(u_sets,2),2,plot_ix);
            plot_ix = plot_ix + 1;
            imagesc(data_base_db, lims); axis xy; xline(25);
            title(['set: ' num2str(i) ', ic: ' num2str(ic)])
            cbar;
        end
    end
    
%     topo_vec = topo_vec(u_sets,:);
%     figure;topoplot(mean(topo_vec,1), ALLEEG(2).chanlocs);
%     figure;topoplot(s.icawinv(:,ic), s.chanlocs);
    
    if plot_ss
        sgtitle(['cluster: ' num2str(c)])
        tightfig;
    end
    
    % add grand mean plot
    data_grand = data_grand(u_sets,:,:);
    base_grand = base_grand(u_sets,:);
    data_mean = squeezemean(data_grand,1);
    base_mean = squeezemean(base_grand,1);
    grand_db = 10.*log10(data_mean ./ base_mean');
    res.betas = 10.*log10(data_grand./base_grand); % hack so PE_limo works
   
    % statistics
    sig_alpha = .05;
    % load times and freqs
    save_info.load_p = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing';
    times = s.etc.analysis.ersp.tf_event_times;
    times_ixs = [1, size(times,2)];
    freqs = s.etc.analysis.ersp.tf_event_freqs;
    max_freq_ix = freqs(end);
    res.times = times;
    res.freqs = freqs;
    save_info.save_name = 'grand_mean';

    % run limo
    PE_limo(save_info, res, 1, times, times_ixs, freqs, max_freq_ix, []);
    
    % multiple comparison correction
    res.ttest.grand_mean.t = load([save_info.load_p '/ttest_' save_info.save_name '/one_sample_ttest_parameter_1.mat']);
    % load bootstrap results
    load([save_info.load_p '/ttest_' save_info.save_name '/H0/tfce_H0_one_sample_ttest_parameter_1.mat']);
    % get max dist of tfce
    for i = 1:size(tfce_H0_one_sample,3)
        this_tfce = squeeze(tfce_H0_one_sample(:,:,i));
        max_dist(i) = max(this_tfce(:));
    end
    % threshold
    thresh = prctile(max_dist, (1-sig_alpha)*100);
    % load true result
    load([save_info.load_p '/ttest_' save_info.save_name  '/tfce/tfce_one_sample_ttest_parameter_1.mat']);
    % threshold true data with bootstrap prctile thresh
    res.ttest.(save_info.save_name).tfce_map = squeeze(tfce_one_sample);
    res.ttest.(save_info.save_name).thresh = thresh;
    res.ttest.(save_info.save_name).sig_mask = res.ttest.(save_name).tfce_map>res.ttest.(save_name).thresh;
    
    % make plot
    lims = max(abs(grand_db(:)))/2 * [-1 1];
    figure('Renderer', 'painters', 'Position', [10 10 1000 300])
    imagesclogy(s.etc.analysis.ersp.tf_event_times, s.etc.analysis.ersp.tf_event_freqs, ...
        grand_db, lims); axis xy;
    set(gca,'FontSize',20);
    ylabel('frequency (Hz)');
    xlabel('time (ms)');
    
    % plot mask
    hold on
    % optional: clean up sig mask
    %res.ttest.(save_name).sig_mask = bwareaopen(res.ttest.(save_name).sig_mask, 10);
    %res.ttest.(save_name).sig_mask = medfilt2(res.ttest.(save_name).sig_mask, [2 2]);
    contour(s.etc.analysis.ersp.tf_event_times, s.etc.analysis.ersp.tf_event_freqs,...
        res.ttest.grand_mean.sig_mask, 1, 'linecolor', 'black', 'LineWidth', 1.5, 'LineStyle', '-');
    cbar; set(gca,'FontSize',20); ylabel('dB');
    
    clear data_grand base_grand
end
