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

%% LIMO regression IPQ

% settings
alpha = .05;
measures = {'erp', 'ersp'};
save_info.robustfit = 0;
% save_info.parameters = {'vel', 'immersion_1', 'trial', 'sequence', 'immersion_1:vel'};
save_info.parameters = {'mean'};
save_info.sensors = {'cluster', 'channel'};

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
                    cluster_ipq = ipq_q1(find(ismember(ipq_q1(:,1), unique_subjects)),:,:);
                    cluster_ipq(:,1) = [];
                    cluster_ipq = mean(cluster_ipq,2);
                else
                    cluster_ipq = mean_q1;
                end

                %% select data
                save_info.load_p = [bemobil_config.study_folder bemobil_config.study_level 'analyses/' measure '/' bemobil_config.study_filename(1:end-6) '/' save_info.sensor '_' num2str(c)];
                save_info.c = c;
                load([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '_vel-at-' num2str(save_info.this_ts) 'ms-pre-event.mat'])

                %% run limo ttest/regression
                if strcmp(measure,'ersp')
                    PE_limo(save_info, res, 1, times, times_ixs, freqs, max_freq_ix, cluster_ipq);
                else
                    PE_limo(save_info, res, 0, [], [], [], [], cluster_ipq);
                end

                % load LIMO output: save mean value and sig. mask to res and resave res
                load([save_info.load_p '/regress_' save_info.parameter '/Betas.mat']);
                save_name = regexprep(save_info.parameter, ':' , '_');
                res.regress_IPQ.(save_name).mean_value = squeeze(Betas(1,:,2)); % mean betas in this case

                % mcc
                % load bootstrap results
                load([save_info.load_p '/regress_' save_info.parameter '/H0/tfce_H0_Covariate_effect_1.mat']);
                % get max dist of tfce
                for i = 1:size(tfce_H0_score,3)
                    if strcmp(measure,'ersp')
                        this_tfce = squeeze(tfce_H0_score(1,:,:,i));
                    else
                        this_tfce = squeeze(tfce_H0_score(:,1,i));
                    end
                    max_dist(i) = max(this_tfce(:));
                end
                % remove NaN
                max_dist(isnan(max_dist)) = [];
                % threshold
                thresh = prctile(max_dist, (1-alpha)*100);
                % load true result
                load([save_info.load_p '/regress_' save_info.parameter '/TFCE/tfce_Covariate_effect_1.mat']);
                % threshold true data with bootstrap prctile thresh
                if strcmp(measure,'ersp')
                    res.regress.(save_name).tfce_map = squeeze(tfce_score);
                    res.regress.(save_name).thresh = prctile(max_dist, (1-alpha)*100);
                else
                    res.regress_IPQ.(save_name).sig_mask = find(tfce_score>thresh);
                end

                % resave res
                save([save_info.load_p '/res_' save_info.model '_robust-' num2str(save_info.robustfit) '_vel-at-' num2str(save_info.this_ts) 'ms-pre-event.mat'], 'res');

            end
        end
    end
end

