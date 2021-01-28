%% load results

% params
PE_config;

% load
brain_prob = 5;
%fname = '_prob_brain_base_removal_-05-0';
fname = '_prob_brain_base_removal_0-05';
%fname = '_prob_brain';
%fname = '_prob_brain_base_removal_full_wnds';
load([bemobil_config.study_folder bemobil_config.study_level 'lda_results_0' num2str(brain_prob)' fname '.mat']);

% inspect
lda_results
mean(lda_results.correct)
std(lda_results.correct)
%dipole numbers / IClabel threhhold 412 .1 ; 320 .2 ; 290 .3 ; 217 .5

% plot all dipoles and save for supplements
%plot_weighteddipoledensity(lda_results.dipoles)
plot_weighteddipoledensity(lda_results.dipoles,mean(lda_results.weights,2));

%% plot patterns, sample EEG chanlocs have to be loaded (DONE & PLOTTED)

save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/lda/';
mkdir(save_path);
normal;

%ixs = 1:19;
%ixs(7) = [];
%figure;topoplot(mean(normalize(reshape(lda_results.patterns(ixs,i), 65, 19), 1), 2), locs)

% s8 is bad for window 6

%figure;
%for j = 1:19 % for all subjects, save for supplements
for i = 6 %1:8 
%for j = 1:19 % for all subjects, save for supplements
    
    % plot
    mean_pattern = lda_results.patterns(:,i);
    mean_pattern = reshape(mean_pattern, [65,19]);
    mean_pattern(:,7) = [];
    %mean_pattern = mean(mean_pattern,2);
    %figure;topoplot(mean_pattern(:,j),locs);
    figure;topoplot(mean(normalize(mean_pattern, 1), 2), locs);
    
    %subplot(4,6,i);
    %mean_pattern = mean_pattern(:,ixs);
    %topoplot(mean_pattern,locs);
    %topoplot(mean(normalize(mean_pattern, 1), 2), locs);
        
    % plot with normalization
    %figure;topoplot(mean(normalize(lda_results.patterns(:,i), 65, 19), 1), 2), locs);
    
    % save with window name
    print(gcf, [save_path '0' num2str(brain_prob)' fname '_lda_pattern_win_' num2str(i) '.eps'], '-depsc');     
    % close figure
    close(gcf);
end
%end

%% plot weighted dipoles at all timepoints (DONE & PLOTTED)

save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/lda/';
mkdir(save_path);
normal;

for i = 1:8
    
    % plot weighted dipoles
    plot_weighteddipoledensity(lda_results.dipoles,lda_results.weights(:,i));
    % save with window name
    print(gcf, [save_path '0' num2str(brain_prob)' fname '_win_' num2str(i) '.eps'], '-depsc');     
    % close figure
    close(gcf);
end

%% plot control signal ERP style: mark 2 classes (DONE & PLOTTED)
    
save_path = '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/figures/lda/';
mkdir(save_path);
normal;
for i = 1:8 
    
    % prepare plot
    figure('visible','on', 'Renderer', 'painters', 'Position', [10 10 300 200]);
    title('Control Signal 6th Window');

    % plot condition 1
    colors = brewermap(5, 'Spectral');
    colors1 = colors(2, :);
    sync = squeeze(lda_results.control_signal(:,1,i,:));
    ploterp_lg(sync, [], [], 50, 1, 'norm. \muV', '', '', colors1, '-');
    hold on

    % plot condition 2
    colors2 = colors(5, :);
    async = squeeze(lda_results.control_signal(:,2,i,:));
    ploterp_lg(async, [], [], 50, 1, 'norm. \muV', '', '', colors2, '-.');
    
    % save with window name
    print(gcf, [save_path '0' num2str(brain_prob)' fname 'lda_control_signal_win_' num2str(i) '.eps'], '-depsc');     
    % close figure
    close(gcf);
end
