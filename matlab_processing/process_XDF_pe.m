close all; clear

config_XDF_pe;

% enter all subjects to process here (you can split it up in more MATLAB instances if you have more CPU power and RAM)
% subjects = 2:20;
% subjects = 2:6;
% subjects = 7:11;
% subjects = 12:16;
subjects = 17:20;

force_recomp = true;

%% processing loop

if ~exist('ALLEEG','var')
	eeglab;
end

if ~exist('mobilab','var') 
	runmobilab;
end

%

for subject = subjects
    
	STUDY = []; CURRENTSTUDY = 0; ALLEEG = [];  CURRENTSET=[]; EEG=[]; EEG_interp_avref = [];
	
	input_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	
	try
		% load completely processed file
		EEG_single_subject_final = pop_loadset('filename', [ bemobil_config.filename_prefix num2str(subject)...
			'_' bemobil_config.single_subject_cleaned_ICA_filename], 'filepath', output_filepath);
	end
	
	if ~force_recomp && exist('EEG_single_subject_final','var')
		clear EEG_single_subject_final
		disp('Subject is completely preprocessed already.')
		continue
    end
	
	disp(['Subject #' num2str(subject)]);
    
    % get all xdf filename in subject folder
    filenames = dir(fullfile(bemobil_config.study_folder, bemobil_config.raw_data_folder, [bemobil_config.filename_prefix num2str(subject)]));
    xdf_ix = find(contains({filenames.name}, 'xdf'));
    filenames = {filenames(xdf_ix).name};
    % remove prefix and suffix to keep compatible with below function...
    for i = 1:numel(filenames)
        % index of first underscore: participant id _ filename
        u_ix = find(ismember(filenames{i}, '_'), 1, 'first');
        bemobil_config.filenames{i} = filenames{i}(u_ix+1:end-4);
    end
	
% 	load xdf files and process them with mobilab, export to eeglab, split MoBI and merge all conditions for EEG, do
% 	basic preprocessing and interpolation
	[ALLEEG, EEG_interp_avref, CURRENTSET] = bemobil_process_all_mobilab(subject, bemobil_config, ALLEEG, CURRENTSET, mobilab, force_recomp);

    % get rid of memory mapped object storage by mobilab
	try 
        pop_editoptions( 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0);
    catch
        warning('Could NOT edit EEGLAB memory options!!'); 
    end
    
% 	start the processing pipeline for AMICA
	[ALLEEG, EEG, CURRENTSET] = bemobil_process_all_AMICA(ALLEEG, EEG_interp_avref, CURRENTSET, subject, bemobil_config, force_recomp);

	STUDY = []; CURRENTSTUDY = 0; ALLEEG = [];  CURRENTSET=[]; EEG_interp_avref = []; bemobil_config.filenames = [];


	%% clean with IClabel
	
    disp('Cleaning data with ICLabel')
    
	EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.copy_weights_interpolate_avRef_filename], 'filepath', input_filepath);
	
	% clean now, save files and figs
	[ALLEEG, ~, CURRENTSET, ICs_keep, ICs_throw,fig_clean] = bemobil_clean_with_iclabel( EEG , ALLEEG, CURRENTSET, 'lite',...
        [1], bemobil_config.brain_threshold,...
		[ bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.single_subject_cleaned_ICA_filename],output_filepath);
	close(fig_clean)
	
    
	% plot dipoles
	pop_dipplot( EEG, ICs_keep,...
		'mri',[fileparts(which('dipfitdefs')) '\standard_BEM\standard_mri.mat'],'normlen','on');

	% save fig
	savefig(fullfile(output_filepath,'brain_dipoles'))
	close
    
    disp('...done.')

end

subjects
subject

close all
clear
disp('PROCESSING DONE! YOU CAN CLOSE THE WINDOW NOW!')