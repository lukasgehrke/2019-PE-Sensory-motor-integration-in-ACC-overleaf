function [EEG, bemobil_config] = select_ICs_pe(EEG, bemobil_config)
%SELECT_ICS project IC activity out of channel data
    
if bemobil_config.lda.brain_threshold > 0
    EEG.etc.analysis.lda.thrown_components = find(EEG.etc.ic_classification.ICLabel.classifications(:,1) < bemobil_config.lda.brain_threshold);
else
    [~,loc] = max(EEG.etc.ic_classification.ICLabel.classifications,[],2);
    EEG.etc.analysis.lda.thrown_components = find(loc~=1);
end
    
EEG = pop_subcomp(EEG, EEG.etc.analysis.lda.thrown_components);
EEG = eeg_checkset(EEG);
    
end

