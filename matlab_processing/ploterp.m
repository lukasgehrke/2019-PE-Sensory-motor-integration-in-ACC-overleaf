function ploterp(data_to_plot, event_onset, plot_ci, y_label, x_label, ylim0, colors)
%PLOTERP plot erp/any line with 95% confidence interval, data_to_plot must
%be a 2D array with [subjects; data], e.g. [19 250]. Indicate an
%event_onset with a vertical line at 'event_onset'

% make x and y data
y_plot = mean(squeeze(data_to_plot),1);
% find event_onset
x_plot = (1:size(y_plot,2)) / 250;
ix = find(x_plot==event_onset);
x_plot = (-ix:size(y_plot,2)) / 250;
x_plot(end-ix:end) = [];

if plot_ci
    % CI
    % Calculate Standard Error Of The Mean
    SEM_A = std(data_to_plot', [], 2)./ sqrt(size(data_to_plot',2));
    % 95% Confidence Intervals
    CI95 = bsxfun(@plus, mean(data_to_plot',2), bsxfun(@times, [-1  1]*1.96, SEM_A));
    upper = CI95(:,2)';
    lower = CI95(:,1)';

    % plot the CI
    p = patch([x_plot fliplr(x_plot)], [lower fliplr(upper)], 'g');
    p.FaceColor = colors(2,:);
    p.FaceAlpha = .3;
    p.EdgeColor = 'none';
    hold on
end

% plot the ERP
l = plot(x_plot, y_plot);
l.Color = colors(1,:);
l.LineWidth = 3;
l.LineStyle = '-.';

% format plot
if ~isempty(x_label)
    xlabel(x_label);
end
if ~isempty(y_label)
    ylabel(y_label);
end
lim = max(abs(y_plot)) * 3;
axis tight

% line at 0
if ylim0
    ylim([0 lim]);
    l2 = line([0 0], [0 lim]);
else
    ylim([-1*lim lim]);
    l2 = line([0 0], [-1*lim lim]);
end
l2.LineStyle = '-';
l2.Color = 'k';
l2.LineWidth = 4;

% format further
grid on
set(gca,'FontSize',20)

end

