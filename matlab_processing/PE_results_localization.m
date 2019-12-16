%% clear all and load params
close all; clear

PE_config;

%% load study : 2nd Level inference

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 1, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

% load IMT_v1 EEGLAB study struct, keeping at most 1 dataset in memory
input_path_STUDY = [bemobil_config.study_folder bemobil_config.study_level];
if isempty(STUDY)
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    [STUDY ALLEEG] = pop_loadstudy('filename', bemobil_config.study_filename, 'filepath', input_path_STUDY);
    CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
    
    eeglab redraw
end
STUDY_sets = cellfun(@str2num, {STUDY.datasetinfo.subject});

%% result 0.0: plot cluster blobs and talairach coordinates

% save dipole location
for cluster = clusters_of_int
    loc(cluster,1:3) = STUDY.cluster(cluster).dipole.posxyz;
    loc(cluster,4) = size(unique(STUDY.cluster(cluster).sets),2);
    loc(cluster,5) = cluster;
end
rem = sum(loc,2) == 0;
loc(rem, :) = [];
save(['/Volumes/Seagate Expansion Drive/work/studies/Prediction_Error/data/5_study_level/clustering_solutions/' bemobil_config.study_filename(1:end-6) '/dipole_locs.mat'], 'loc');
clear loc

%% plot clusters dipoleclusters
colors = brewermap(size(clusters_of_int,2),'Set1');
std_dipoleclusters(STUDY, ALLEEG, 'clusters', clusters_of_int,...
    'centroid', 'add',...
    'projlines', 'on',...
    'viewnum', 4,...
    'colors', colors);

%% makotos NIMAs Blobs
voxelSize      = 4;
FWHM           = 10;
blobOrVoxelIdx = 1;
uniformAlpha   = .4;
optionStr = [];
% optionStr = '''separateColorList'',[0.8941 0.1020 0.1098; 0.2157 0.4941 0.7216; 0.3020 0.6863 0.2902; 0.5961 0.3059 0.6392]';
% optionStr = '''separateColorList'',[0.8941 0.1020 0.1098; 0.2157 0.4941 0.7216; 0.3020 0.6863 0.2902; 0.5961 0.3059 0.6392; 1.0000 0.4980 0; 1.0000 1.0000 0.2000]';

% plot NIMAs Blobs
% Obtain cluster dipoles.
clusterDipoles = std_readdipoles(STUDY, ALLEEG, clusters_of_int);

% Obtain cluster dipole locations.
clusterDipoleLocations = cell(1,length(clusters_of_int));
for clsIdx = 1:length(clusters_of_int)
    currentClsDip = clusterDipoles{clsIdx};
    
    % Dual dipoles are counted as two.
    dipList = [];
    for dipIdx = 1:length(currentClsDip)
        currentDipList = round(currentClsDip(dipIdx).posxyz);
        dipList = cat(1, dipList, currentDipList);
    end
    
    % Store as cluster dipole locations.
    clusterDipoleLocations{1,clsIdx} = dipList;
end

% Parse optional input to construct string-value pairs if it is not empty.
if isempty(optionStr)
    optionalInput = {};
else
    commaIdx = strfind(optionStr, ',');
    for optionalInputIdx = 1:length(commaIdx)+1
        
        if optionalInputIdx == 1;
            currentStrings = optionStr(1:commaIdx(1)-1);
        elseif optionalInputIdx == length(commaIdx)+1
            currentStrings = optionStr(commaIdx(end):length(optionStr));
        else
            currentStrings = optionStr(commaIdx(optionalInputIdx-1)+1:commaIdx(optionalInputIdx)-1);
        end
        currentStrings = strtrim(currentStrings);
        
        if mod(optionalInputIdx,2) == 0 % Odd numbers are strings, even numbers are values.
            optionalInput{optionalInputIdx} = str2num(currentStrings);
        else
            optionalInput{optionalInputIdx} = currentStrings(2:end-1);
        end
    end
end
nimasImagesfromMpA(clusterDipoleLocations, 'voxelSize', voxelSize,...
                                           'FWHM', FWHM,...
                                           'blobOrVoxel', blobOrVoxelIdx,...
                                           'uniformAlpha', uniformAlpha,...
                                           optionalInput);
