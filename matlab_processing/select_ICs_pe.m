function [outEEG, bemobil_config] = select_ICs_pe(inEEG, bemobil_config)
%SELECT_ICS project IC activity out of channel data
    
if bemobil_config.lda.brain_threshold > 0
    inEEG.etc.analysis.lda.thrown_components = find(inEEG.etc.ic_classification.ICLabel.classifications(:,1) < bemobil_config.lda.brain_threshold);
else
    [~,loc] = max(inEEG.etc.ic_classification.ICLabel.classifications,[],2);
    inEEG.etc.analysis.lda.thrown_components = find(loc~=1);
end
    
outEEG = pop_subcomp(inEEG, inEEG.etc.analysis.lda.thrown_components);
outEEG = eeg_checkset(outEEG);
    
end

