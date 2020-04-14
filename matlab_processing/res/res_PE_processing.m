%% processing: ERPs

% for channel ERPs project out eye component activation time courses
eyes = find(EEG.etc.ic_classification.ICLabel.classifications(:,3) > bemobil_config.eye_threshold);
line = find(EEG.etc.ic_classification.ICLabel.classifications(:,5) > bemobil_config.line_threshold);
ERP = pop_subcomp(EEG, unique([eyes', line']));

% filter
ERP = pop_eegfiltnew(ERP, bemobil_config.filter_plot_low, bemobil_config.filter_plot_high);
ERP_event = ERP;
ERP_base = ERP;

% overwrite events with clean epoch events
ERP_event.event = EEG.event(touch_ixs); % touches
ERP_base.event = EEG.event(spawn_ixs); % spawns

% ERPs: both EEG and mocap
ERP_event = pop_epoch( ERP_event, bemobil_config.epoching.event_epoching_event, bemobil_config.epoching.event_epochs_boundaries, 'newname',...
    'epochs', 'epochinfo', 'yes');
ERP_base = pop_epoch( ERP_base , bemobil_config.epoching.base_epoching_event, bemobil_config.epoching.base_epochs_boundaries, 'newname',...
    'epochs', 'epochinfo', 'yes');

% channel ERP
% EEG: find ~10% noisiest epoch indices by searching for large amplitude
% fluctuations on the channel level using eeglab auto_rej function
[~, rmepochs] = pop_autorej(ERP_event, 'maxrej', 1, 'nogui','on','eegplot','off');
EEG.etc.analysis.erp.rm_ixs = sort([bad_tr_ixs, rmepochs]);
% baseline correct
base_win_samples = (bemobil_config.epoching.base_win * ERP_base.srate) ...
    + abs(bemobil_config.epoching.base_epochs_boundaries) * ERP_base.srate;
channel_baseline = mean(ERP_base.data(:,base_win_samples(1):base_win_samples(2),:),2);
ERP_event.data = ERP_event.data - channel_baseline;
EEG.etc.analysis.erp.data = ERP_event.data;


%% cleaning ERSP
% find bad epochs based on component ERP: actually select good
% components as they should be most relevant for the cleaning as they
% are the ones later being analysed
% about the IClabel threshold: Marius said thats what Luca used, cite!
comps = find(EEG.etc.ic_classification.ICLabel.classifications(:,1) > bemobil_config.brain_threshold);
[~, rmepochs] = pop_autorej(ERSP_event, 'electrodes', [], 'icacomps', comps, 'maxrej', 1, 'nogui','on','eegplot','off');
EEG.etc.analysis.ersp.rm_ixs = sort([bad_tr_ixs, rmepochs