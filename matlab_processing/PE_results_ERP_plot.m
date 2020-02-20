%% Params
PE_config;

%% FINAL plot r^2s for ERPs

% what to plot
metric = 'mocap'; % 'erp';
channel = 'vel'; %['channel_' num2str(65)];
channel_name = 'FCz';
plot_sample_start_end = -250:250;
models = {'erp_sample ~ congruency * haptics + trial_nr + direction + sequence', ...
    'erp_sample ~ velocity * haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ after_mismatch * haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ congruency * haptics + trial_nr + direction + sequence'};
model = models{3};

% paths
load_p = ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/' ...
    metric '/' bemobil_config.study_filename(1:end-6) '/'];

% shift due to EEG age of sample?
age_of_sample = 0;

% prepare plot
cols = brewermap(size(channels_of_int,2), 'Accent');
figure('Renderer', 'painters', 'Position', [10 10 450 300]);
hold on;

% load and plot
load([load_p channel '/' model '/res_' model '_robust-' num2str(robustfit) '.mat']);
data = res.r2;
%x_plot = (-25-age_of_sample:size(data,2)-26-age_of_sample) / 250;
% x_plot = x_plot * 1000; % seconds to ms
x_plot = plot_sample_start_end / 250;
ylim([0 max(mean(data,1))+.05]);
l = plot(x_plot, mean(data,1));
l.LineWidth = 3;
l.Color = cols(1,:);

% title({'Modelfit of channels of model:',model});
xlabel('time in ms')
ylabel('R^2')
l2 = line([0 0], [0 max(ylim)]);
l2.LineStyle = '-';
l2.Color = 'k';
l2.LineWidth = 3;

% format further
% legend({channel_name});
grid on
set(gca,'FontSize',20)
% tightfig;

%% FINAL velocity ERP mismatch: stopping faster with vibro motor? -> no

do_save = 1; % save results or not

% what to plot
metric = 'mocap'; % 'erp';
sample_erp_zero = 250;
channel =  'vel';
models = {'vel_erp_sample ~ haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ after_mismatch * haptics + trial_nr + direction + was_sequence'};
m_ix = 1;
model = models{m_ix};
robustfit = 0;

% paths
load_p = ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/' ...
    metric '/' bemobil_config.study_filename(1:end-6) '/'];

% shift due to EEG age of sample?
age_of_sample = 0;

% prepare plot
cols = brewermap(2, 'Spectral');

% load data
load([load_p channel '/' model '/res_' model '_robust-' num2str(robustfit) '.mat']);

% make figure
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 960 200]);
%hold on;
s_ix = 1;

% plot mean ERP -> sum all betas
subplot(1,3,s_ix);
s_ix = s_ix + 1;
data = sum(res.betas,3);
ploterp_lg(data, [], sample_erp_zero, 1, 'modeled ERV m/s', '', [0 .8], cols);

% plot r^2
subplot(1,3,s_ix);
s_ix = s_ix + 1;
data = res.r2;
ploterp_lg(data, [], sample_erp_zero, 1, 'R^2', '', [0 1], cols);

% plot coeff congruency
subplot(1,3,s_ix);
s_ix = s_ix + 1;
data = res.betas(:,:,2);
sig_ix = res.ttest.haptics_1.sig_mask;
ploterp_lg(data, sig_ix, sample_erp_zero, 1, 'haptics', '', [], cols);

% optional
%tightfig;
if do_save
    outp = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/';
    saveas(gcf,[outp channel '_' metric '_' num2str(m_ix) '.eps'],'epsc')
    close(gcf);
end

%% FINAL velocity ERP after mismatch: behavioral adaptation after mismatch? -> yes

do_save = 1; % save results or not

% what to plot
metric = 'mocap'; % 'erp';
sample_erp_zero = 250;
channel =  'vel';
models = {'vel_erp_sample ~ haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ after_mismatch * haptics + trial_nr + direction + was_sequence'};
m_ix = 2;
model = models{m_ix};
robustfit = 0;

% paths
load_p = ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/' ...
    metric '/' bemobil_config.study_filename(1:end-6) '/'];

% shift due to EEG age of sample?
age_of_sample = 0;

% prepare plot
cols = brewermap(2, 'Spectral');

% load data
load([load_p channel '/' model '/res_' model '_robust-' num2str(robustfit) '.mat']);

% make figure
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1600 200]);
%hold on;
s_ix = 1;

% plot mean ERP -> sum all betas
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = sum(res.betas,3);
ploterp_lg(data, [], sample_erp_zero, 1, 'modeled ERV m/s', '', [0 .7], cols);

% plot r^2
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.r2;
ploterp_lg(data, [], sample_erp_zero, 1, 'R^2', '', [0 1], cols);

% plot coeff after_mismatch
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.betas(:,:,2);
sig_ix = res.ttest.after_mismatch.sig_mask;
ploterp_lg(data, sig_ix, sample_erp_zero, 1, 'post mismatch', '', [], cols);

% plot coeff congruency
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.betas(:,:,3);
sig_ix = res.ttest.haptics_1.sig_mask;
ploterp_lg(data, sig_ix, sample_erp_zero, 1, 'haptics', '', [], cols);

% plot coeff congruency
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.betas(:,:,8);
sig_ix = res.ttest.haptics_1_after_mismatch.sig_mask;
ploterp_lg(data, sig_ix, sample_erp_zero, 1, 'post mis. X haptics', '', [], cols);

% optional
%tightfig;
if do_save
    outp = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/';
    saveas(gcf,[outp channel '_' metric '_' num2str(m_ix) '.eps'],'epsc')
    close(gcf);
end

%% FINAL Channels Congruent: make figure with mean ERP, R^2, coefficients

do_save = 0; % save results or not

% what to plot
sample_erp_zero = 25;
robustfit = 0; % fit robust regression with squared weights, see fitlm
metric = 'erp'; 
channel =  ['channel_' num2str(65)];
plot_sample_start_end = -250:250;
models = {'erp_sample ~ congruency * haptics + trial_nr + direction + sequence', ...
    'erp_sample ~ velocity * haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ after_mismatch * haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ congruency * haptics + trial_nr + direction + sequence'};
m_ix = 3;
model = models{m_ix};

% paths
load_p = ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/' ...
    metric '/' bemobil_config.study_filename(1:end-6) '/'];

% shift due to EEG age of sample?
age_of_sample = 0;

% prepare plot
cols = brewermap(2, 'Spectral');

% load data
load([load_p channel '/' model '/res_' model '_robust-' num2str(robustfit) '.mat']);

% make figure
figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1600 200]);
%hold on;
s_ix = 1;

% plot mean ERP -> sum all betas
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = sum(res.betas,3);
ploterp_lg(data, [], sample_erp_zero, 1, 'modeled ERP (\muV)', '', [], cols);

% plot r^2
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.r2;
ploterp_lg(data, [], sample_erp_zero, 1, 'R^2', '', 1, cols);

% plot coeff congruency
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.betas(:,:,2);
sig_ix = res.ttest.congruency_1.sig_mask;
ploterp_lg(data, sig_ix, sample_erp_zero, 1, 'congruency', '', [], cols);

% plot coeff congruency
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.betas(:,:,3);
sig_ix = res.ttest.haptics_1.sig_mask;
ploterp_lg(data, sig_ix, sample_erp_zero, 1, 'haptics', '', [], cols);

% plot coeff congruency
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.betas(:,:,8);
sig_ix = res.ttest.congruency_1_haptics_1.sig_mask;
ploterp_lg(data, sig_ix, sample_erp_zero, 1, 'cong. X haptics', '', [], cols);

% optional
%tightfig;
if do_save
    outp = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/';
    saveas(gcf,[outp channel '_' metric '_' num2str(m_ix) '.eps'],'epsc')
end

%% FINAL Channels Mismatch: make figure with mean ERP, R^2, coefficients

% what to plot
metric = 'erp'; % 'mocap';
channel =  ['channel_' num2str(25)]; %vel;
plot_sample_start_end = -250:250;
models = {'erp_sample ~ congruency * haptics + trial_nr + direction + sequence', ...
    'erp_sample ~ velocity * haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ after_mismatch * haptics + trial_nr + direction + sequence',...
    'vel_erp_sample ~ congruency * haptics + trial_nr + direction + sequence'};
m_ix = 2;
model = models{m_ix};

% paths
load_p = ['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/analyses/' ...
    metric '/' bemobil_config.study_filename(1:end-6) '/'];

% shift due to EEG age of sample?
age_of_sample = 0;

% prepare plot
cols = brewermap(2, 'Spectral');

% load data
load([load_p channel '/' model '/res_' model '_robust-' num2str(robustfit) '.mat']);

% make figure
figure('visible','off', 'Renderer', 'painters', 'Position', [10 10 1600 200]);
%hold on;
s_ix = 1;

% plot mean ERP -> sum all betas
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = sum(res.betas,3);
ploterp_lg(data, [], sample_erp_zero, 1, 'modeled ERP (\muV)', '', [], cols);

% plot r^2
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.r2;
ploterp_lg(data, [], sample_erp_zero, 1, 'R^2', '', 1, cols);

% plot coeff congruency
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.betas(:,:,3);
sig_ix = res.ttest.velocity.sig_mask;
ploterp_lg(data, sig_ix, sample_erp_zero, 1, 'velocity', '', [], cols);

% plot coeff congruency
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.betas(:,:,2);
sig_ix = res.ttest.haptics_1.sig_mask;
ploterp_lg(data, sig_ix, sample_erp_zero, 1, 'haptics', '', [], cols);

% plot coeff congruency
subplot(1,5,s_ix);
s_ix = s_ix + 1;
data = res.betas(:,:,8);
sig_ix = res.ttest.haptics_1_velocity.sig_mask;
ploterp_lg(data, sig_ix, sample_erp_zero, 1, 'vel. X haptics', '', [], cols);

% optional
%tightfig;
outp = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/';
saveas(gcf,[outp channel '_' metric '_' num2str(m_ix) '.eps'],'epsc')


%% TODO Channels & Clusters: make figure with regression IPQ results, mark significant pvals and

cols = brewermap(2, 'Spectral');

sensor = 'cluster'; % channel
cs = clusters_of_int; % channels_of_int

coeff_names =  {'mean', 'immersion_1', 'vel', 'trial', 'sequence', 'immersion_1_vel'};
subplots = [2, 6]; % top coefficient, below R^2

for chan = cs
    
    % subplot ix
    s_ix = 1;
    % make figure
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 1800 600]);
        
    for coeff = coeff_names

        subplot(subplots(1),subplots(2),s_ix);
        s_ix = s_ix + 1;
        load([load_p sensor '_' num2str(chan) '/regress_' coeff{1} '/Betas.mat']);
        data = squeeze(Betas(1,:,2));
        
        % make tfce thresh sig mask
        load([load_p sensor '_' num2str(chan) '/regress_' coeff{1} '/H0/tfce_H0_Covariate_effect_1.mat']); % bootstraps tfce
        load([load_p sensor '_' num2str(chan) '/regress_' coeff{1} '/TFCE/tfce_Covariate_effect_1.mat']); % true tfce
        for i = 1:size(tfce_H0_score,3)
            max_i(i) = max(tfce_H0_score(event_0_ix:end,2,i));
        end
        thresh = prctile(max_i, (1-alpha)*100);
        sig_ix = find(tfce_score>thresh);
        ploterp_lg(data, sig_ix, event_0_ix, 1, ['IPQ: ' coeff{1}], '', [], cols);

        % plot r^2
        subplot(subplots(1),subplots(2),s_ix + (subplots(2)-1));
        load([load_p sensor '_' num2str(chan) '/regress_' coeff{1} '/R2.mat']);
        data = squeeze(R2(1,:,1));
        ploterp_lg(data, [], event_0_ix, 1, 'R^2', '', [0 1], cols);
    end

    tightfig;
    
    saveas(gcf, [load_p sensor '_' num2str(chan) '/ipq_res.png'], 'png')
    close(gcf);

end