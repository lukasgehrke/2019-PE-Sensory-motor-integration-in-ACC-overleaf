%% clear all and load params
close all; clear
eeglab
runmobilab;

PE_config;

%% processing loop

if ~exist('ALLEEG','var')
	eeglab;
	runmobilab;
end

for subject = subjects
	
	disp(['Subject #' num2str(subject)]);
	
	% load xdf files and process them with mobilab, export to eeglab, split MoBI and merge all conditions for EEG
	[ALLEEG, EEG_merged, CURRENTSET] = bemobil_process_all_mobilab(subject, bemobil_config, ALLEEG, CURRENTSET, mobilab);
    
	% finally start the complete processing pipeline including AMICA
	[ALLEEG, EEG_AMICA_final, CURRENTSET] = bemobil_process_all_AMICA(ALLEEG, EEG_merged, CURRENTSET, subject, bemobil_config);
	
end

%% plot brain ICs

if ~exist('ALLEEG','var'); eeglab; end
pop_editoptions( 'option_storedisk', 0, 'option_savetwofiles', 1, 'option_saveversion6', 0, 'option_single', 0, 'option_memmapdata', 0, 'option_eegobject', 0, 'option_computeica', 1, 'option_scaleicarms', 1, 'option_rememberfolder', 1, 'option_donotusetoolboxes', 0, 'option_checkversion', 1, 'option_chat', 1);

for subject = subjects
	
	disp(['Subject #' num2str(subject)]);
	
	input_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	output_filepath = [bemobil_config.study_folder bemobil_config.single_subject_analysis_folder bemobil_config.filename_prefix num2str(subject)];
	
	EEG = pop_loadset('filename',[ bemobil_config.filename_prefix num2str(subject) '_'...
		bemobil_config.copy_weights_interpolate_avRef_filename], 'filepath', input_filepath);
	[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'study',0);
	
	% clean now, save files and figs
	[ALLEEG, EEG_cleaned, CURRENTSET, ICs_keep, ICs_throw] = bemobil_clean_with_iclabel( EEG , ALLEEG, CURRENTSET, [1], bemobil_config.brain_threshold,...
		[ bemobil_config.filename_prefix num2str(subject) '_' bemobil_config.single_subject_cleaned_ICA_filename],output_filepath);
	
end

% plot dipoles
pop_dipplot( EEG, ICs_keep,...
	'mri','P:\\Marius\\toolboxes\\eeglab14_1_0b\\plugins\\dipfit2.3\\standard_BEM\\standard_mri.mat','normlen','on');

% save fig
savefig(fullfile(output_filepath,'brain_dipoles'))

%% epoch cleaning

%% ERP pipeline

% difference ERP mismatch - congruent
% - compute mean of all clean congruent trials
% - subtract mean of all clean congruent trials from n clean mismatch
% single trials
% - extract velocity at box touched for all the n clean difference trials
% - run 1st level analysis fitting lm at each point of the ERP post event
% - 










%% Clustering?

