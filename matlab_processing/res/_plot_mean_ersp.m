%% Example: CORRMAP cluster
pop_corrmap(STUDY,ALLEEG,1, 5,'chanlocs','','th','auto','ics',1,'title','Cluster s','clname','s','badcomps','no', 'resetclusters','off');
ic = CORRMAP.output.ics{2};
sets = CORRMAP.output.sets{2};

%% plot mean ersp test

plot_ss = 0;
min_rv = 0;
trials = 'mismatch'; % 'match'

clusters = [10]; %8, 11, 15, 28, 33, 36];
std_dipoleclusters(STUDY, ALLEEG, 'clusters', clusters,...
    'centroid', 'add',...
    'projlines', 'on',...
    'viewnum', 4);

for c = clusters

    u_sets = unique(STUDY.cluster(c).sets);
    comps = STUDY.cluster(c).comps;

    % find ics per set
    plot_ix = 1;
    if plot_ss
        figure;
    end
    for i = u_sets
        
        set_ixs = find(i==STUDY.cluster(c).sets);
        ic = comps(set_ixs);
        
        % average over ics or select ic with min rv in case of more than 1
        % IC per subject
        if min_rv && size(ic,2) > 1
            choose_rv = find(STUDY.cluster(c).residual_variances(set_ixs)==min(STUDY.cluster(c).residual_variances(set_ixs)));
            ic = ic(choose_rv);
        end
        
        % accress ersp data
        s = ALLEEG(i);
        if strcmp(trials, 'mismatch')
            tr_ix = s.etc.analysis.design.oddball;
        elseif strcmp(trials, 'match')
            tr_ix = ~s.etc.analysis.design.oddball;
        end            
 
        % get mean times of events
        rt(i) = mean(s.etc.analysis.design.rt_spawned_touched(tr_ix));
        % trials taken below considers all trials so a but wrong (should not matter much)
        start(i) = mean(s.etc.analysis.design.isitime(tr_ix));
        
        % prepare ersp data
        data = squeezemean(s.etc.analysis.ersp.tf_event_raw_power(ic,:,:,tr_ix),4);
        base = squeezemean(s.etc.analysis.ersp.tf_base_raw_power(ic,:,tr_ix),3)';
        if ~min_rv && size(ic,2) > 1
            data = squeezemean(data,1);
            base = squeezemean(base,2);
        end
        
        data_grand(i,:,:) = data;
        base_grand(i,:) = base;
        data_base_db = 10.*log10(data ./ base);
        lims = max(abs(data_base_db(:)))/2 * [-1 1];
        
        if plot_ss
            subplot(size(u_sets,2),2,plot_ix);
            plot_ix = plot_ix + 1;
            
            if ~min_rv && size(ic,2) > 1
                topo_vec = mean(s.icawinv(:,ic),2);
            else
                topo_vec = s.icawinv(:,ic);
            end
            topoplot(topo_vec, s.chanlocs);

            subplot(size(u_sets,2),2,plot_ix);
            plot_ix = plot_ix + 1;
            imagesc(data_base_db, lims); axis xy; xline(25);
            title(['set: ' num2str(i) ', ic: ' num2str(ic)])
            cbar;
        end
    end
    
    if plot_ss
        sgtitle(['cluster: ' num2str(c)])
        tightfig;
    end
    
    % add grand mean plot
    figure;
    grand_mean = squeezemean(data_grand,1);
    base_mean = squeezemean(base_grand,1);
    rt(rt==0)=[];
    start(start==0)=[];
    
    grand_db = 10.*log10(grand_mean ./ base_mean');
    lims = max(abs(grand_db(:)))/2 * [-1 1];
    imagesclogy(s.etc.analysis.ersp.tf_event_times, s.etc.analysis.ersp.tf_event_freqs, grand_db, lims); axis xy;
    xline(0,'-',{'box:touched'});
    xline(-1*round(mean(rt)*1000,1),'-',{'box:spawned'});
    xline(-1*round(mean(start)*1000,1),'-',{'trial:start'});
    
    title(['grand mean: ' num2str(c)]); cbar;
    
    clear data_grand base_grand
end

