for i = 1:19 
    rm_trials(i) = size(ALLEEG(i).etc.analysis.design.rm_ixs,2);
end

mean(rm_trials)
std(rm_trials)