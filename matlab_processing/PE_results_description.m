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

%% FINAL result 1.1: mean velocity/ERP (FCz, Pz) curve with confidence interval

% from ersp
begin_ms = -2444;
end_ms = 1440;

% make xtlabels
all_samples = 1250;
xtlabels = 1:all_samples;
xtlabels = (xtlabels / s.srate) + bemobil_config.epoching.event_epochs_boundaries(1);
xtlabels = round(xtlabels*1000);
begin_sample = find(ismember(xtlabels,begin_ms));
end_sample = find(ismember(xtlabels,end_ms));
xtlabels = xtlabels(begin_sample:end_sample);

c = 1;
for s = ALLEEG
    
    bad_trs = union(s.etc.analysis.ersp.rm_ixs, s.etc.analysis.erp.rm_ixs);
    oddball = s.etc.analysis.design.oddball;
    oddball(bad_trs) = [];
    rt = s.etc.analysis.design.rt_spawned_touched;
    rt(bad_trs) = [];
    rt_mismatch = rt(oddball);
    rt = rt_mismatch * s.srate;
    
    % select timeseries measure
    measure = s.etc.analysis.mocap.mag_acc;
    %measure = squeeze(s.etc.analysis.erp.data(65,:,:));
    
    measure(:,bad_trs) = [];
    measure = measure(begin_sample:end_sample,oddball);
    
    measure_all_subjects(c,:,:) = squeeze(nanmean(measure,2));
    c = c+1;
end

mean_measure = mean(measure_all_subjects,1);
% Calculate Standard Error Of The Mean
SEM_A = std(measure_all_subjects', [], 2)./ sqrt(size(measure_all_subjects',2));
% 95% Confidence Intervals
CI95 = bsxfun(@plus, mean(measure_all_subjects',2), bsxfun(@times, [-1  1]*1.96, SEM_A));
upper = CI95(:,2)';
lower = CI95(:,1)';

% make plot
map = brewermap(2,'Set1'); 
figure('Renderer', 'painters', 'Position', [10 10 450 300])
p = patch([xtlabels fliplr(xtlabels)], [lower fliplr(upper)], 'g');
p.FaceColor = map(1,:);
p.FaceAlpha = .5;
p.EdgeColor = 'none';
hold on;
l = plot(xtlabels, mean_measure);
l.MarkerFaceColor = map(2,:);
l.LineWidth = 3;

% line at 0
%l2 = line([0 0], [0 .8]);
%l2.LineStyle = '-';
%l2.Color = 'k';
%l2.LineWidth = 2;

% % more lines at sampled velocities
% cols = brewermap(size(ts_of_ints,2), 'Spectral');
% for i = 1:size(ts_of_ints,2)
%     ts = (ts_of_ints(i)-event_onset)/250;
%     l = line([ts ts], [0 .8]);
%     l.LineStyle = '-.';
%     l.Color = cols(end-i+1,:);
%     l.LineWidth = 2;
% end
% legend('95% confidence interval', 'hand velocity', 'collision', 'sampled velocities');

%legend('95% CI', 'velocity');
%legend boxoff
axis tight
% grid on
set(gca,'FontSize',20)
ylabel('magnitude of velocity');
xlabel('seconds');

%% FINAL result 1.2: grand average ERSP of ACC cluster

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
        base = s.etc.analysis.ersp.tf_event_raw_power(ic,:,:,:);
        data(:,:,:,bad_trs) = [];
        base(:,:,:,bad_trs) = [];
        
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
    
    if plot_ss
        sgtitle(['cluster: ' num2str(c)])
        tightfig;
    end
    
    % add grand mean plot
    figure;
    data_grand = data_grand(u_sets,:,:);
    base_grand = base_grand(u_sets,:);
    
    data_mean = squeezemean(data_grand,1);
    base_mean = squeezemean(base_grand,1);
    %rt(rt==0)=[];
    %start(start==0)=[];
    
    grand_db = 10.*log10(grand_mean ./ base_mean');
    lims = max(abs(grand_db(:)))/2 * [-1 1];
    imagesclogy(s.etc.analysis.ersp.tf_event_times, s.etc.analysis.ersp.tf_event_freqs, grand_db, lims); axis xy;
    %xline(0,'-',{'box:touched'});
    %xline(-1*round(mean(rt)*1000,1),'-',{'box:spawned'});
    %xline(-1*round(mean(start)*1000,1),'-',{'trial:start'});
    
    %title(['grand mean: ' num2str(c)]); 
    cbar;
    
    clear data_grand base_grand
end

