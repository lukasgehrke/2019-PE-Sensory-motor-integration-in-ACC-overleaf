for i = 1:19 
    ics(i) = size(ALLEEG(i).icaweights,1);
end

mean(ics)
std(ics)