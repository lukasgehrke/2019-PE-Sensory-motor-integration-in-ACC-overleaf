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

% select subjects out of clusters of int
clusters_of_int = [3, 7, 9, 24, 25, 28, 30, 33, 34];
% clusters
% 3: right parietal
% 7: right motor?
% 24: right SMA
% 25: left parietal
% 28: interesting
% 33: ACC

channels_of_int = [5, 25, 65];
% channels
% 5: Fz
% 25: Pz
% 65: FCz

%% result 0.1: Histograms of reaction times mismatch and match trials

all_match_rt = [];
all_mismatch_rt = []; 

for s = ALLEEG
    
    all_mismatch_rt = [all_mismatch_rt s.etc.epoching.latency_diff_mismatch];
    all_match_rt = [all_match_rt s.etc.epoching.latency_diff_match];
    
end

% covert to time
all_mismatch_rt = all_mismatch_rt/s.srate;
all_match_rt =  all_match_rt/s.srate;

map = brewermap(2,'Set1'); 
all_match_rt(all_match_rt>2) = [];

figure;
grid on;
hold on;
h1 = histfit(all_match_rt, 50);
h2 = histfit(all_mismatch_rt, 50);

h1(1).FaceColor = map(1,:);
h1(1).EdgeColor = 'none';
h1(1).FaceAlpha = .5;
h1(2).Color = map(1,:);

h2(1).FaceColor = map(2,:);
h2(1).EdgeColor = 'none';
h2(1).FaceAlpha = .7;
h2(2).Color = map(2,:);

mean(all_match_rt)
l = line([mean(all_match_rt) mean(all_match_rt)], [0 500]);
l.Color = map(1,:);
l.LineWidth = 2;

mean(all_mismatch_rt)
l2 = line([mean(all_mismatch_rt) mean(all_mismatch_rt)], [0 500]);
l2.Color = map(2,:);
l2.LineWidth = 2;

mean(all_match_rt) - mean(all_mismatch_rt)

%format plot
set(gca,'FontSize',20)
box off
l = legend('match', 'fit',...
    'mismatch', 'fit',...
    'location','northeast');
legend boxoff
xlabel('seconds')
ylabel('frequency')

%% result 0.2: histogram velocity profile / mean velocity curve with confidence interval

cond = 'vibro';
trials = 'good_vibro';
buffertime = .5;
c = 1;
for s = ALLEEG
    
    rt = s.etc.epoching.latency_diff_mismatch(s.etc.epoching.(trials));
    buffer = buffertime * s.srate;
    event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * s.srate);

    event_start = event_onset - rt;
    event_end = event_start + rt + buffer;
    event_wins = floor([event_start; event_end]');

    x = s.etc.analysis.mocap.(cond).x;
    y = s.etc.analysis.mocap.(cond).y;
    z = s.etc.analysis.mocap.(cond).z;
    measure = s.etc.analysis.mocap.(cond).mag_vel;
    
    % get windows
    for i = 1:size(event_wins,1)
        data(event_wins(i,1):event_wins(i,2),:,1) = x(event_wins(i,1):event_wins(i,2),:);
        data(event_wins(i,1):event_wins(i,2),:,2) = y(event_wins(i,1):event_wins(i,2),:);
        data(event_wins(i,1):event_wins(i,2),:,3) = z(event_wins(i,1):event_wins(i,2),:);
        data(event_wins(i,1):event_wins(i,2),:,4) = measure(event_wins(i,1):event_wins(i,2),:);
    end
    
    % standardize offset between subjects
    data(:,:,1) = data(:,:,1) - data(event_wins(1),:,1);
    data(:,:,2) = data(:,:,2) - data(event_wins(1),:,2);
    data(:,:,3) = data(:,:,3) - data(event_wins(1),:,3);
    
    all_dat(c,:,:) = squeeze(nanmean(data,2));
    c = c+1;
    
    clear data
end

% plot mean velocity profile with 2*sd
start = 380;
vel = all_dat(:,start:end,4);
mean_vel = mean(vel,1);
% Calculate Standard Error Of The Mean
SEM_A = std(vel', [], 2)./ sqrt(size(vel',2));
% 95% Confidence Intervals
CI95 = bsxfun(@plus, mean(vel',2), bsxfun(@times, [-1  1]*1.96, SEM_A));
upper = CI95(:,2)';
lower = CI95(:,1)';

% prepare labels
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * s.srate);
xtlabels = start:size(all_dat,2);
xtlabels = xtlabels / s.srate;
xtlabels = xtlabels - xtlabels(event_onset-start);

% make plot
map = brewermap(2,'Set1'); 
figure;
p = patch([xtlabels fliplr(xtlabels)], [lower fliplr(upper)], 'g');
p.FaceColor = map(1,:);
p.FaceAlpha = .5;
p.EdgeColor = 'none';
hold on;
l = plot(xtlabels, mean_vel);
l.MarkerFaceColor = map(2,:);
l.LineWidth = 3;

% line at 0
l2 = line([0 0], [0 .8]);
l2.LineStyle = '-';
l2.Color = 'k';
l2.LineWidth = 8;

% more lines at sampled velocities
cols = brewermap(size(ts_of_ints,2), 'Spectral');
for i = 1:size(ts_of_ints,2)
    ts = (ts_of_ints(i)-event_onset)/250;
    l = line([ts ts], [0 .8]);
    l.LineStyle = '-.';
    l.Color = cols(end-i+1,:);
    l.LineWidth = 2;
end

legend('95% confidence interval', 'hand movement', 'collision', 'sampled velocities');
legend boxoff
axis tight
% grid on
set(gca,'FontSize',20)
ylabel('magnitude of velocity');
xlabel('seconds');

%% result 0.3: plot 3D trajectory colorcoded by hand velocity and erp amplitude (erpimage2D)

% 1: make hand velocity at each x,y for each subject, then plot average
% across subjects and smooth

% 2: make erp amplitude (erpimage2D) at each (x,y) for each subject, then
% plot average across subjects and smooth it -> this is important and cool

cond = 'vibro';
trials = 'good_vibro';
chan = 25;
buffertime = 0;
clusters_of_int = 1:size(STUDY.cluster,2);

for cluster = clusters_of_int

    % get matching datasets from EEGLAB Study struct
    unique_setindices = unique(STUDY.cluster(cluster).sets);
    unique_subjects = STUDY_sets(unique_setindices);
    all_setindices = STUDY.cluster(cluster).sets;
    all_sets = STUDY_sets(all_setindices);
    all_comps = STUDY.cluster(cluster).comps;

    c = 1;
    all_dat = NaN(1,4);

    for subject = unique_setindices(1:6)

        s = ALLEEG(subject);

        rt = s.etc.epoching.latency_diff_mismatch(s.etc.epoching.(trials));
        buffer = buffertime * s.srate;
        event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * s.srate);

        event_start = event_onset - rt;
        event_end = event_start + rt + buffer;
        event_wins = floor([event_start; event_end]');

        x = s.etc.analysis.mocap.(cond).x;
        y = s.etc.analysis.mocap.(cond).y;
        z = s.etc.analysis.mocap.(cond).z;
    %     measure = s.etc.analysis.mocap.(cond).mag_vel;

        % average ERP if more than 1 comp
        compos = all_comps(all_sets==subject+1);
        if compos > 1
            s.etc.analysis.erp.base_corrected.visual.comps(compos(1),:,:) = ...
                mean(s.etc.analysis.erp.base_corrected.visual.comps(compos,:,:),1);
            s.etc.analysis.erp.base_corrected.vibro.comps(compos(1),:,:) = ...
                mean(s.etc.analysis.erp.base_corrected.vibro.comps(compos,:,:),1);
        end
        measure = squeeze(s.etc.analysis.erp.base_corrected.(cond).comps(compos(1),:,:));

        % get windows
        data = NaN(size(x,1), size(x,2),4);
        for i = 1:size(event_wins,1)
            data(event_wins(i,1):event_wins(i,2),:,1) = x(event_wins(i,1):event_wins(i,2),:);
            data(event_wins(i,1):event_wins(i,2),:,2) = y(event_wins(i,1):event_wins(i,2),:);
            data(event_wins(i,1):event_wins(i,2),:,3) = z(event_wins(i,1):event_wins(i,2),:);
            data(event_wins(i,1):event_wins(i,2),:,4) = measure(event_wins(i,1):event_wins(i,2),:);
        end

        % standardize offset between subjects
        data(:,:,1) = data(:,:,1) - data(event_wins(1),:,1);
        data(:,:,2) = data(:,:,2) - data(event_wins(1),:,2);
        data(:,:,3) = data(:,:,3) - data(event_wins(1),:,3);

        % reshape to long formant concatenating trials as rows
        data_long = reshape(data,[],size(data,3),1);

        % remove noisy trials
        dev = 3*nanstd(data_long(:,4));
        rem_ix = find(data_long(:,4)>dev);
        data_long(rem_ix,:) = [];

        all_dat = vertcat(all_dat, data_long);
        c = c+1;
    end

    % plot of param on lines
    figure('visible','off', 'Renderer', 'painters', 'Position', [10 10 900 900]);
    % restrict colorbar
    max_lim = max(all_dat(:,4));
    min_lim = min(all_dat(:,4));  
    lim = min(abs([max_lim, min_lim])) / 2;
    all_dat(:,4) = min( max(all_dat(:,4), -1*lim), lim); % scale to lim
    % plot
    fig_erpimage2d = patch([all_dat(:,1)' nan],[all_dat(:,3)' nan],[all_dat(:,2)' nan],[all_dat(:,4)' nan],...
        'LineWidth', 2, 'FaceColor','none','EdgeColor','interp'); grid on;
%     view(3);
    ylim([-.05, .3]);
    xlim([-.2 .2]);
    set(gca,'FontSize',20)
%     axis tight
    cbar;
    set(gca,'FontSize',20)
    % save
    saveas(gcf, ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/plots/brainXtrajectory/' num2str(cluster) '.png'])
    % clear data
    clear all_dat

end
