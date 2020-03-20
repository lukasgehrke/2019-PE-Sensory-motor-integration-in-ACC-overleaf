c = 6; % 24

data = squeezemean(EEG.etc.analysis.ersp.tf_event_raw_power(c,:,:,:),4);
base = squeezemean(EEG.etc.analysis.ersp.tf_base_raw_power(c,:,:),3);
data = 10.*log10(data./base');

t = EEG.etc.analysis.ersp.tf_event_times;
f = EEG.etc.analysis.ersp.tf_event_freqs;
lims = max(abs(data(:)))/2 * [-1 1];
figure; imagesclogy(t, f, data, lims); axis xy; xline(0); title(num2str(c)); cbar;