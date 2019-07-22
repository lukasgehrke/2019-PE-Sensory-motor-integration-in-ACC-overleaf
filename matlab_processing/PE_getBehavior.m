%% load params
PE_config;

%%

EEG = loadset();
EEG = parse_events_PredError(EEG);