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

%% FINAL task description 1: Histograms of reaction times mismatch and match trials

all_match_rt = [];
all_mismatch_rt = []; 

% load data
for s = ALLEEG
        
    % calculate reaction times (time between box:spawned and box:touched)
    % of mismatch and match
    bad_trs = union(s.etc.analysis.ersp.rm_ixs, s.etc.analysis.erp.rm_ixs);
    oddball = s.etc.analysis.design.oddball;
    oddball(bad_trs) = [];
    rt = s.etc.analysis.design.rt_spawned_touched;
    rt(bad_trs) = [];
    
    all_mismatch_rt = [all_mismatch_rt rt(oddball)];
    all_match_rt = [all_match_rt rt(~oddball)];
end

% print means
disp(["mean reaction time matching feedback: " mean(all_match_rt)]);
disp(["mean reaction time mismatching feedback: " mean(all_mismatch_rt)]);
disp(["difference between matching and mismatching feedback: " mean(all_match_rt) - mean(all_mismatch_rt)]);

% plot
map = brewermap(2,'Set1'); 
figure('Renderer', 'painters', 'Position', [10 10 450 300])
grid on;
hold on;
h1 = histfit(all_match_rt, 50);
h2 = histfit(all_mismatch_rt, 50);

h1(1).FaceColor = map(1,:);
h1(1).EdgeColor = 'none';
h1(1).FaceAlpha = .5;
%h1(2).Color = map(1,:);
%set(get(get(h1(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
delete(h1(2))

h2(1).FaceColor = map(2,:);
h2(1).EdgeColor = 'none';
h2(1).FaceAlpha = .7;
%h2(2).Color = map(2,:);
%set(get(get(h2(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
delete(h2(2))

l = line([mean(all_match_rt) mean(all_match_rt)], [0 500]);
l.Color = map(1,:);
l.LineWidth = 3;

l2 = line([mean(all_mismatch_rt) mean(all_mismatch_rt)], [0 500]);
l2.Color = map(2,:);
l2.LineWidth = 3;

%format plot
set(gca,'FontSize',20);
box off
l = legend('match', 'mismatch',...
    'location','northeast');
legend boxoff
xlabel('seconds')
ylabel('frequency')

%% FINAL task description 2: plot 3D trajectory colorcoded by hand velocity (and erp amplitude (erpimage2D))

% 1: make hand velocity at each x,y for each subject, then plot average
% across subjects and smooth

% 2: make erp amplitude (erpimage2D) at each (x,y) for each subject, then
% plot average across subjects and smooth it -> this is important and cool

buffertime = 0;
measure_all_subjects = NaN(1,4);

for s = ALLEEG(2)

    % get reaction times mismatch trials to extract windows
    bad_trs = union(s.etc.analysis.ersp.rm_ixs, s.etc.analysis.erp.rm_ixs);
    oddball = s.etc.analysis.design.oddball;
    oddball(bad_trs) = [];
    rt = s.etc.analysis.design.rt_spawned_touched;
    rt(bad_trs) = [];
    rt = rt(oddball) * s.srate;

    buffer = buffertime * s.srate;
    event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * s.srate);

    event_start = event_onset - rt;
    event_end = event_start + rt + buffer;
    event_wins = floor([event_start; event_end]');

    x = squeeze(s.etc.analysis.mocap.x);
    y = squeeze(s.etc.analysis.mocap.y);
    z = squeeze(s.etc.analysis.mocap.z);
    measure = s.etc.analysis.mocap.mag_vel;
    
    x(:,bad_trs) = [];
    y(:,bad_trs) = [];
    z(:,bad_trs) = [];
    measure(:,bad_trs) = [];
    
    x = x(:,oddball);
    y = y(:,oddball);
    z = z(:,oddball);
    measure = measure(:,oddball);

    % get windows
    data = NaN(size(x,1),size(x,2),4);
    for i = 1:size(event_wins,1)
        data(event_wins(i,1):event_wins(i,2),i,1) = x(event_wins(i,1):event_wins(i,2),i);
        data(event_wins(i,1):event_wins(i,2),i,2) = y(event_wins(i,1):event_wins(i,2),i);
        data(event_wins(i,1):event_wins(i,2),i,3) = z(event_wins(i,1):event_wins(i,2),i);
        data(event_wins(i,1):event_wins(i,2),i,4) = measure(event_wins(i,1):event_wins(i,2),i);
    end

    % standardize offset between subjects
%     data(:,:,1) = data(:,:,1) - data(event_wins(1),:,1);
%     data(:,:,2) = data(:,:,2) - data(event_wins(1),:,2);
%     data(:,:,3) = data(:,:,3) - data(event_wins(1),:,3);

    % reshape to long format concatenating trials as rows
    data_long = reshape(data,[],size(data,3),1);

    % remove noisy trials
    dev = 3*nanstd(data_long(:,4));
    rem_ix = find(data_long(:,4)>dev);
    data_long(rem_ix,:) = [];
    measure_all_subjects = data_long; %vertcat(measure_all_subjects, data_long);
end

% plot of param on lines
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 450 300]);
% % restrict colorbar
% max_lim = max(all_dat(:,4));
% min_lim = min(all_dat(:,4));  
% lim = min(abs([max_lim, min_lim])) / 2;
% all_dat(:,4) = min( max(all_dat(:,4), -1*lim), lim); % scale to lim
% plot
fig_erpimage2d = patch([measure_all_subjects(:,1)' nan],[measure_all_subjects(:,3)' nan],[measure_all_subjects(:,2)' nan],[measure_all_subjects(:,4)' nan],...
    'LineWidth', 2, 'FaceColor','none','EdgeColor','interp'); grid off;
view(3);
%ylim([-.05, .3]);
%xlim([-.2 .2]);
%set(gca,'FontSize',20)
cbar; %title('velocity')
%set(gca,'FontSize',20)
% clear data
clear all_dat

%epsclean('/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing/test_clean.eps');