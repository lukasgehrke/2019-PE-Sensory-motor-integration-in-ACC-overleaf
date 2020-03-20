t = load('/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing/ttest_grand_mean/H0/H0_one_sample_ttest_parameter_1.mat');
t = squeeze(t.H0_one_sample);

for i = 1:11:120
    figure;imagesclogy(res.times, res.freqs, t(:,:,1,i),[-5 5]); axis xy; cbar;
end

tfce = load('/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf/matlab_processing/ttest_grand_mean/H0/tfce_H0_one_sample_ttest_parameter_1.mat');
tfce = tfce.tfce_H0_one_sample;

for i = ixs%1:11:120
    figure;imagesclogy(res.times, res.freqs, tfce(:,:,i), [-20000 20000]); axis xy; cbar;
    this_tfce = tfce(:,:,i);
    round(max(this_tfce(:)))
end

% true data
figure;imagesclogy(res.times, res.freqs, true_tfce, [-200 200]); axis xy; cbar;

% 
for i = 1:size(boots,2)
    uniques(i) = size(unique(boots(:,i)),1);
end
figure;hist(uniques);

min_uniques = 12;
ixs = find(uniques>=min_uniques);
max_dist_unique_12 = max_dist(ixs);
figure;hist(max_dist_unique_12)
round(prctile(max_dist_unique_12, [80:2:100]))
disp(['thresh:' num2str(round(prctile(max_dist_unique_12, 95)))]);