figure;
for i = 1:size(res.betas,1)
    data = squeeze(res.betas(i,:,:,2));
    lims = max(abs(data(:)))/2 * [-1 1];
    subplot(4,4,i);
    imagesclogy(t, f, data, lims);axis xy;xline(0);cbar;
end

%%

for ix = 1:9
%ix = 2;
    data = squeezemean(res.betas(:,:,:,ix),1);
    lims = max(abs(data(:))) * [-1 1];
    subplot(1,9,ix);
    imagesclogy(t, f, data, lims); axis xy; xline(0); title(res.parameter_names(ix)); cbar;
end