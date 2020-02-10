function PE_limo(save_info, res, ersp, times, times_ixs, freqs, max_freq_ix, regressor)
%PE_LIMO_TTEST runs a one-sample ttest and saves results

% LIMO one-sample t-test

%   limo_random_robust(1,y,parameter number,nboot,tfce)
%                      1 = a one-sample t-test
%                      y = data (dim electrodes, time or freq, subjects)
%                        = data (dim electrodes, freq, time, subjects)
%                      parameter number = describe which parameter is currently analysed (e.g. 1 - use for maming only)
%                      nboot = nb of resamples (0 for none)
%                      tfce = 0/1 to compute tcfe (only if nboot ~=0).

% LIMO regression

%           limo_random_robust(4,y,X,parameter number,nboot,tfce);
%                      4 = regression analysis
%                      y = data (dim electrodes, time or freq, subjects)
%                        = data (dim electrodes, freq, time, subjects)
%                      X = continuous regressor(s)
%                      parameter number = describe which parameter is currently analysed (e.g. 1 - use for maming only)
%                      nboot = nb of resamples (0 for none)
%                      tfce = 0/1 to compute tcfe (only if nboot ~=0).

% build LIMO struct
LIMO.Level         = 2;
LIMO.Type          = 'Components';

if isempty(regressor)
    LIMO.design.name   = 'ttest';
else
    LIMO.design.name   = 'regression';
end
LIMO.data.data_dir = [save_info.load_p];
LIMO.data.chanlocs = [];

% find index of data
data_ix = find(strcmp(save_info.parameter, res.parameter_names));

if ersp
    LIMO.Analysis      = 'Time-Frequency';
    LIMO.data.tf_times = times;
    LIMO.data.start    = LIMO.data.tf_times(1);
    LIMO.data.end      = LIMO.data.tf_times(end);
    LIMO.data.tf_freqs = freqs;
    LIMO.data.lowf     = LIMO.data.tf_freqs(1); 
    LIMO.data.highf    = LIMO.data.tf_freqs(end);

    if isempty(data_ix) % make the mean
        y = sum(res.betas(:, 1:max_freq_ix, times_ixs(1):times_ixs(2), :), 4);
    else
        y = res.betas(:, 1:max_freq_ix, times_ixs(1):times_ixs(2), data_ix);
    end
    y = permute(y, [4, 2, 3, 1]);

else
    LIMO.Analysis      = 'erp';
    
    if isempty(data_ix) % make the mean
        y = sum(res.betas,3);
    else
        y = res.betas(:,:,data_ix);
    end
    y = permute(y, [3, 2, 1]);
end

LIMO.data.sampling_rate = 250;
LIMO.data.Cat = '';

if isempty(regressor)
    LIMO.data.Cont = '';
else
    LIMO.data.Cont = regressor;
end
LIMO.data.neighbouring_matrix = '';
LIMO.data.data = '';

LIMO.design.fullfactorial    = 0; % 0/1 specify if interaction should be included
LIMO.design.zscore           = 0; %/1 zscoring of continuous regressors
LIMO.design.method           = ''; % actuially no effect because random_robust looks at the design
LIMO.design.type_of_analysis = 'Mass-univariate'; 
LIMO.design.bootstrap        = 1000; % 0/1 indicates if bootstrap should be performed or not (by default 0 for group studies)
LIMO.design.tfce             = 1; %0/1 indicates to compute TFCE or not

LIMO.design.nb_categorical = 0;

if ~isempty(regressor)
    LIMO.design.nb_continuous = regressor;
else
    LIMO.design.nb_continuous = 0; %scalar that returns the number of continuous variables e.g. [3]
end
LIMO.design.status = 'to do';

% parameter added using debugger due to errors being thrown in limo_random_robust
save_param_name = regexprep(save_info.parameter, ':' , '_');
save_param_name = regexprep(save_param_name, '(' , '');
save_param_name = regexprep(save_param_name, ')' , '');

if isempty(regressor)
    LIMO.dir = [LIMO.data.data_dir '/ttest_' save_param_name];
else
    LIMO.dir = [LIMO.data.data_dir '/regress_' save_param_name];
end
LIMO.data.size3D = [size(y,1), size(y,2)*size(y,3), size(y,4)];
LIMO.data.size4D = [size(y,1), size(y,2) size(y,3) size(y,4)];

% save LIMO.mat
if exist(LIMO.dir, 'dir')
    rmdir(LIMO.dir, 's')
end
mkdir(LIMO.dir);

save([LIMO.dir filesep 'Y.mat'], 'y');
save([LIMO.dir filesep 'LIMO.mat'], 'LIMO');
current_folder = pwd;
cd(LIMO.dir)

% run analysis
if isempty(regressor)
    limo_random_robust(1, y, 1, LIMO.design.bootstrap, LIMO.design.tfce);
else
    limo_random_robust(4, y, LIMO.data.Cont, 'ipq', LIMO, LIMO.design.bootstrap, LIMO.design.tfce);
end
close(gcf);
cd(current_folder);

end

