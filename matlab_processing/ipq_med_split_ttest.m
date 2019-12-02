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
clusters_of_int = [3, 7, 28, 33];
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

%% IPQ
T = table2array(readtable('/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/admin/ipq_long.csv'));
ipq_q1 = T(1:4:end,:,:);

% do visual and vibro differ? no
[H,P,CI,STATS] = ttest(ipq_q1(:,2),ipq_q1(:,3))

% then aggregate across vis and vibro for subject average
q1 = ipq_q1(:,[2,3]);
mean_q1 = mean(q1,2);
ix_lower = find(mean_q1 < mean(mean_q1));
ix_higher = find(mean_q1 > mean(mean_q1));

[H,P,CI,STATS] = ttest2(mean_q1(ix_lower), mean_q1(ix_higher))
% mean split results in significant mean difference, proof that mean split is meaningful

%% plot histogram two IPQ Q1 groups

map = brewermap(2,'Set1'); 

figure;
grid on;
hold on;
edges = 1:.5:7.5;
b = bar(histcounts(mean_q1,edges));
ylim([0,7])
xticklabels(edges)


h1(1).FaceColor = map(1,:);
h1(1).EdgeColor = 'none';
h1(1).FaceAlpha = .5;
h1(2).Color = map(1,:);

h2(1).FaceColor = map(2,:);
h2(1).EdgeColor = 'none';
h2(1).FaceAlpha = .7;
h2(2).Color = map(2,:);

l = line([mean(mean_q1(ix_lower)) mean(mean_q1(ix_lower))], [0 7]);
l.Color = map(1,:);
l.LineWidth = 2;

l2 = line([mean(mean_q1(ix_higher)) mean(mean_q1(ix_higher))], [0 7]);
l2.Color = map(2,:);
l2.LineWidth = 2;

mean(mean_q1(ix_lower)) - mean(mean_q1(ix_higher))

%format plot
set(gca,'FontSize',20)
box off
l = legend('low', '',...
    'high', '',...
    'location','northeast');
legend boxoff
xlabel('ipq q1 score')
ylabel('frequency')

%% match IPQ scores to subjects in a given cluster

%% prepare IPQ score regressor for cluster

cluster = 28;

% get matching datasets from EEGLAB Study struct
unique_setindices = unique(STUDY.cluster(cluster).sets);
unique_subjects = STUDY_sets(unique_setindices);
all_setindices = STUDY.cluster(cluster).sets;
all_sets = STUDY_sets(all_setindices);
all_comps = STUDY.cluster(cluster).comps;

cluster_ipq = ipq_q1(find(ismember(ipq_q1(:,1), unique_subjects)),:,:);
cluster_ipq(:,1) = [];
cluster_ipq = mean(cluster_ipq,2);

%% load EEG data, find name of file
event_onset = abs(bemobil_config.epoching.event_epochs_boundaries(1) * 250);
robustfit = 1;
% shift due to EEG age of sample
eeg_age_of_sample = .066; % ms age of sample eeg data
eeg_age_of_sample_samples = floor(eeg_age_of_sample * 250);
plot_s_after_event = .7; % plot 0 to half a second after event
plot_win = event_onset:event_onset+(plot_s_after_event * 250);
event_0_ix = (event_onset+eeg_age_of_sample_samples-plot_win(1)) / 250;
% load
load_p = '/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/cluster_';
model = 'erp_sample ~ predictor_immersion * predictor_vel';
mocap_aos = .011; % age of sample mocap samples = 11 ms
mocap_aos_samples = ceil(mocap_aos * 250);
% try out several ts prior to event
seconds_before_event = .5;
samples_before_event = seconds_before_event * 250;
ts_of_ints = event_onset-samples_before_event:20:event_onset;
ts_of_ints = ts_of_ints - mocap_aos_samples;
this_ts = (event_onset - ts_of_ints) / 250;
% select best one
this_ts = this_ts(end);
% finally load data
load([load_p num2str(cluster) '/res_' model '_robust-' num2str(robustfit) '_vel-at-' num2str(this_ts) 'ms-pre-event_cluster-' num2str(cluster) '.mat']);

%% LIMO regression IPQ on cluster ERP

%           limo_random_robust(4,y,X,parameter number,nboot,tfce);
%                      4 = regression analysis
%                      y = data (dim electrodes, time or freq, subjects)
%                        = data (dim electrodes, freq, time, subjects)
%                      X = continuous regressor(s)
%                      parameter number = describe which parameter is currently analysed (e.g. 1 - use for maming only)
%                      nboot = nb of resamples (0 for none)
%                      tfce = 0/1 to compute tcfe (only if nboot ~=0).

% y = sum(res.betas(:,plot_win,:),3); % mean ERP
y = res.betas(:,plot_win,3); % velocity
y = permute(y, [3, 2, 1]);
X = cluster_ipq;

% build LIMO struct
LIMO.Level         = 2;
LIMO.Analysis      = 'ERP';
LIMO.Type          = 'Components';
LIMO.design.name   = 'Regression';
LIMO.data.data_dir = [load_p num2str(cluster) '/'];
LIMO.data.chanlocs = [];

LIMO.data.sampling_rate = 250;
LIMO.data.Cat = '';
LIMO.data.Cont = X;
LIMO.data.neighbouring_matrix = '';
LIMO.data.data = '';

LIMO.design.fullfactorial    = 0; % 0/1 specify if interaction should be included
LIMO.design.zscore           = 0; %/1 zscoring of continuous regressors
LIMO.design.method           = ''; % actuially no effect because random_robust looks at the design
LIMO.design.type_of_analysis = 'Mass-univariate'; 
LIMO.design.bootstrap        = 800; % 0/1 indicates if bootstrap should be performed or not (by default 0 for group studies)
LIMO.design.tfce             = 1; %0/1 indicates to compute TFCE or not

LIMO.design.nb_categorical = 0;
LIMO.design.nb_continuous = 1; %scalar that returns the number of continuous variables e.g. [3]
LIMO.design.name = 'regression'; %name of the design
LIMO.design.status = 'to do';

% parameter added using debugger due to errors being thrown in limo_random_robust
LIMO.dir = [LIMO.data.data_dir 'regression_ipq_vel' ];
LIMO.data.size3D = [size(y,1), size(y,2)*size(y,3), size(y,4)];
LIMO.data.size4D = [size(y,1), size(y,2) size(y,3) size(y,4)];

% save LIMO.mat
if exist(LIMO.dir, 'dir')
    rmdir(LIMO.dir, 's')
end
mkdir(LIMO.dir);

save([LIMO.dir filesep 'Y.mat'], 'y');
save([LIMO.dir filesep 'LIMO.mat'], 'LIMO');
current_folder = pwd;
cd(LIMO.dir)

% run analysis
limo_random_robust(4, y, LIMO.data.Cont, 'ipq', LIMO, LIMO.design.bootstrap, LIMO.design.tfce);
close(gcf);
cd(current_folder);    








