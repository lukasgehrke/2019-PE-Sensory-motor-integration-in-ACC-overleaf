%% inspect results

% add downloaded analyses code to the path
addpath(genpath('/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing'));

% Results output folder -> external drive
bemobil_config.study_folder = fullfile('/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error', 'derivatives');

% init
config_processing_pe;
subjects = 1:19;

% load
for t = [0, .5, .7, .8, .9]

    bemobil_config.lda.brain_threshold = t;
    fname = ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_base_removal-' num2str(bemobil_config.lda.approach{3}{4}(1)) '-'  num2str(bemobil_config.lda.approach{3}{4}(2))];
    load(fullfile(bemobil_config.study_folder, ['lda_results-' fname '.mat']));
    
    disp(['tstat: ', num2str(lda_results.ttest.stats.tstat), '; class acc.: ', num2str(mean(lda_results.correct)), '; thresh: ' num2str(t), ' number of dipoles: ', num2str(size(lda_results.dipoles,1))])

end

bemobil_config.lda.brain_threshold = .7;
fname = ['brain_thresh-' num2str(bemobil_config.lda.brain_threshold) '_base_removal-' num2str(bemobil_config.lda.approach{3}{4}(1)) '-'  num2str(bemobil_config.lda.approach{3}{4}(2))];
load(fullfile(bemobil_config.study_folder, ['lda_results-' fname '.mat']));

% plot all dipoles and save for supplements
plot_weighteddipoledensity(lda_results.dipoles)

plot_weighteddipoledensity(lda_results.dipoles,mean(lda_results.weights,2));
plot_weighteddipoledensity(lda_results.dipoles,mean(lda_results.weights(:,2),2));

plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,1));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,2));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,3));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,4)); % posterior, precuneus
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,5));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,6));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,7));
plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,8));