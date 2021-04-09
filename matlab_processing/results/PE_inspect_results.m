

%% ERSP extract stats
% 1. alpha baseline post event
effect = 2;
t = 48; % 200 ms
fit.times(t)
f = 21; % 9 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 2. alpha rt post event
effect = 5;
t = 52; % 250 ms
fit.times(t)
f = 26; % 12 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 3. alpha rt pre event
effect = 5;
t = 16; % -200 ms
fit.times(t)
f = 16; % 12 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 4. alpha haptics post event
effect = 3;
t = 46; % 180 ms
fit.times(t)
f = 11; % 5 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 5. alpha haptics post event
effect = 3;
t = 46; % 180 ms
fit.times(t)
f = 26; % 12 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 6. alpha velocity pre event
effect = 4;
t = 16; % 180 ms
fit.times(t)
f = 26; % 12 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)

% 7. theta interaction post event
effect = 6;
t = 46; % 180 ms
fit.times(t)
f = 6; % 12 Hz
fit.freqs(f)
fit.stats(effect).t_stats(f,t)
df
fit.stats(effect).betas_p_vals(f,t)