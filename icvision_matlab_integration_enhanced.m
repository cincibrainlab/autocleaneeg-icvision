function [EEG, results] = icvision_matlab_integration_enhanced(EEG, varargin)
% ICVISION_MATLAB_INTEGRATION_ENHANCED  Enhanced MATLAB integration with cleaning options
%
% This function provides a MATLAB -> Python -> MATLAB workflow for ICVision
% with options for classification-only or cleaning + classification.
%
% Usage:
%   [EEG, results] = icvision_matlab_integration_enhanced(EEG)
%   [EEG, results] = icvision_matlab_integration_enhanced(EEG, 'Mode', 'classify')
%   [EEG, results] = icvision_matlab_integration_enhanced(EEG, 'Mode', 'clean')
%   [EEG, results] = icvision_matlab_integration_enhanced(EEG, 'TempDir', temp_dir, 'ApiKey', api_key)
%
% Inputs:
%   EEG     - EEGLAB EEG structure with ICA decomposition
%
% Name-Value Parameters:
%   'Mode'     - 'classify' (default) or 'clean'
%                'classify': Only add classification to EEG.etc, don't modify data
%                'clean': Apply artifact removal AND add classification
%   'TempDir'  - Temporary directory for intermediate files (default: system temp)
%   'ApiKey'   - OpenAI API key string (default: get from environment)
%
% Outputs:
%   EEG     - Updated EEG structure
%             Mode 'classify': Original data + classification in EEG.etc
%             Mode 'clean': Cleaned data + classification in EEG.etc
%   results - Structure with classification details:
%             .mode: 'classify' or 'clean'
%             .n_components: Number of components classified
%             .n_artifacts: Number of artifact components found
%             .artifacts_removed: Indices of removed components (clean mode only)
%             .classification_summary: Summary of component classes
%
% Examples:
%   % Classification only (adds metadata, keeps original data)
%   [EEG, results] = icvision_matlab_integration_enhanced(EEG, 'Mode', 'classify');
%   
%   % Cleaning + classification (removes artifacts, adds metadata)
%   [EEG, results] = icvision_matlab_integration_enhanced(EEG, 'Mode', 'clean');

% Parse input arguments
p = inputParser;
addRequired(p, 'EEG', @isstruct);
addParameter(p, 'Mode', 'classify', @(x) ismember(lower(x), {'classify', 'clean'}));
addParameter(p, 'TempDir', tempdir, @ischar);
addParameter(p, 'ApiKey', '', @ischar);
parse(p, EEG, varargin{:});

mode = lower(p.Results.Mode);
temp_dir = p.Results.TempDir;
api_key = p.Results.ApiKey;

% Get API key if not provided
if isempty(api_key)
    api_key = getenv('OPENAI_API_KEY');
    if isempty(api_key)
        fprintf('Warning: No API key provided and OPENAI_API_KEY not found in environment.\n');
        fprintf('You may need to provide the API key:\n');
        fprintf('  [...] = icvision_matlab_integration_enhanced(EEG, ''ApiKey'', ''your-key'')\n');
        fprintf('Attempting to proceed anyway...\n');
    end
end

% Validate EEG structure
if ~isstruct(EEG)
    error('EEG must be a structure');
end

if ~isfield(EEG, 'icaweights') || isempty(EEG.icaweights)
    error('EEG structure must contain ICA decomposition (run ICA first)');
end

fprintf('=== ICVision MATLAB Integration (Enhanced) ===\n');
fprintf('Processing EEG dataset: %s\n', EEG.setname);
fprintf('Mode: %s\n', upper(mode));

n_components = size(EEG.icaweights, 1);
fprintf('Number of ICA components: %d\n', n_components);

try
    % Save temporary .set file for Python processing
    temp_filename = sprintf('icvision_temp_%s.set', datestr(now, 'yyyymmdd_HHMMSS'));
    temp_filepath = fullfile(temp_dir, temp_filename);
    
    fprintf('Saving temporary .set file: %s\n', temp_filepath);
    pop_saveset(EEG, 'filename', temp_filename, 'filepath', temp_dir);
    
    % Prepare file paths
    temp_filepath_py = strrep(temp_filepath, '\', '/');
    icalabel_filepath = [temp_filepath(1:end-4) '_icalabel.mat'];
    icalabel_filepath_py = strrep(icalabel_filepath, '\', '/');
    
    % Call Python ICVision classification
    fprintf('Running ICVision classification via Python...\n');
    
    if strcmp(mode, 'clean')
        % Clean mode: also export cleaned data
        cleaned_filepath = [temp_filepath(1:end-4) '_cleaned.set'];
        cleaned_filepath_py = strrep(cleaned_filepath, '\', '/');
        
        if ~isempty(api_key)
            python_cmd = sprintf(['python -c "import os; os.environ[''OPENAI_API_KEY''] = ''%s''; ' ...
                'from icvision.eeglab import classify_components_for_matlab; import scipy.io; ' ...
                'result = classify_components_for_matlab(''%s'', mode=''%s''); ' ...
                'scipy.io.savemat(''%s'', {''ICLabel'': result})"'], ...
                api_key, temp_filepath_py, mode, icalabel_filepath_py);
        else
            python_cmd = sprintf(['python -c "from icvision.eeglab import classify_components_for_matlab; import scipy.io; ' ...
                'result = classify_components_for_matlab(''%s'', mode=''%s''); ' ...
                'scipy.io.savemat(''%s'', {''ICLabel'': result})"'], ...
                temp_filepath_py, mode, icalabel_filepath_py);
        end
    else
        % Classify mode: only classification
        if ~isempty(api_key)
            python_cmd = sprintf(['python -c "import os; os.environ[''OPENAI_API_KEY''] = ''%s''; ' ...
                'from icvision.eeglab import classify_components_for_matlab; import scipy.io; ' ...
                'result = classify_components_for_matlab(''%s'', mode=''%s''); ' ...
                'scipy.io.savemat(''%s'', {''ICLabel'': result})"'], ...
                api_key, temp_filepath_py, mode, icalabel_filepath_py);
        else
            python_cmd = sprintf(['python -c "from icvision.eeglab import classify_components_for_matlab; import scipy.io; ' ...
                'result = classify_components_for_matlab(''%s'', mode=''%s''); ' ...
                'scipy.io.savemat(''%s'', {''ICLabel'': result})"'], ...
                temp_filepath_py, mode, icalabel_filepath_py);
        end
    end
    
    % Execute Python command
    [status, cmdout] = system(python_cmd);
    
    if status ~= 0
        error('Python ICVision classification failed:\n%s', cmdout);
    end
    
    fprintf('ICVision classification completed successfully\n');
    
    % Load classification results
    if ~exist(icalabel_filepath, 'file')
        error('Classification results file not found: %s', icalabel_filepath);
    end
    
    fprintf('Loading classification results...\n');
    icalabel_data = load(icalabel_filepath);
    
    % Apply cleaning if requested
    if strcmp(mode, 'clean')
        if exist(cleaned_filepath, 'file')
            fprintf('Loading cleaned EEG data...\n');
            EEG_cleaned = pop_loadset(cleaned_filepath);
            
            % Replace EEG data with cleaned version
            EEG.data = EEG_cleaned.data;
            EEG.setname = [EEG.setname '_ICVision_cleaned'];
            
            % Update processing history
            if ~isfield(EEG, 'history')
                EEG.history = '';
            end
            EEG.history = [EEG.history sprintf('EEG = icvision_matlab_integration_enhanced(EEG, ''Mode'', ''clean''); %% %s\n', datestr(now))];
            
            fprintf('Applied artifact removal to EEG data\n');
        else
            warning('Cleaned data file not found, proceeding with classification only');
        end
    end
    
    % Integrate classification into EEG structure
    fprintf('Integrating ICLabel classification into EEG structure...\n');
    
    % Ensure etc field exists
    if ~isfield(EEG, 'etc')
        EEG.etc = struct();
    end
    
    % Ensure ic_classification field exists
    if ~isfield(EEG.etc, 'ic_classification')
        EEG.etc.ic_classification = struct();
    end
    
    % Add ICLabel classification data
    EEG.etc.ic_classification.ICLabel = icalabel_data.ICLabel;
    
    % Verify integration
    n_classes = length(EEG.etc.ic_classification.ICLabel.classes);
    n_components_classified = size(EEG.etc.ic_classification.ICLabel.classification, 1);
    
    fprintf('ICLabel integration successful:\n');
    fprintf('  Classes: %d\n', n_classes);
    fprintf('  Components: %d\n', n_components_classified);
    fprintf('  Classification matrix: [%d x %d]\n', ...
        size(EEG.etc.ic_classification.ICLabel.classification));
    fprintf('  Version: %s\n', EEG.etc.ic_classification.ICLabel.version);
    
    % Create results structure
    results = struct();
    results.mode = mode;
    results.n_components = n_components_classified;
    
    % Process artifact information
    if isfield(icalabel_data.ICLabel, 'artifacts_removed')
        results.artifacts_removed = icalabel_data.ICLabel.artifacts_removed;
        results.n_artifacts = length(results.artifacts_removed);
    else
        results.artifacts_removed = [];
        results.n_artifacts = 0;
    end
    
    % Display classification summary
    fprintf('\nClassification Summary:\n');
    [~, predicted_classes] = max(EEG.etc.ic_classification.ICLabel.classification, [], 2);
    
    classification_summary = struct();
    for i = 1:length(EEG.etc.ic_classification.ICLabel.classes)
        class_name = EEG.etc.ic_classification.ICLabel.classes{i};
        count = sum(predicted_classes == i);
        fprintf('  %s: %d components\n', class_name, count);
        
        % Store in results
        field_name = strrep(strrep(lower(class_name), ' ', '_'), '-', '_');
        classification_summary.(field_name) = count;
    end
    results.classification_summary = classification_summary;
    
    % Display mode-specific information
    if strcmp(mode, 'clean') && results.n_artifacts > 0
        fprintf('\nArtifact Removal Summary:\n');
        fprintf('  Components removed: %d\n', results.n_artifacts);
        fprintf('  Removed component indices: %s\n', mat2str(results.artifacts_removed));
        fprintf('  EEG data has been cleaned\n');
    elseif strcmp(mode, 'classify')
        fprintf('\nClassification-only mode: EEG data unchanged\n');
        if results.n_artifacts > 0
            fprintf('  Found %d artifact components (not removed)\n', results.n_artifacts);
            fprintf('  Use ''Mode'', ''clean'' to remove artifacts\n');
        end
    end
    
    % Clean up temporary files
    fprintf('\nCleaning up temporary files...\n');
    cleanup_files = {temp_filepath, [temp_filepath(1:end-4) '.fdt'], icalabel_filepath};
    if strcmp(mode, 'clean')
        cleanup_files{end+1} = cleaned_filepath;
        cleanup_files{end+1} = [cleaned_filepath(1:end-4) '.fdt'];
    end
    
    for i = 1:length(cleanup_files)
        if exist(cleanup_files{i}, 'file')
            delete(cleanup_files{i});
        end
    end
    
    fprintf('=== Integration Complete ===\n');
    fprintf('Mode: %s\n', upper(mode));
    fprintf('EEG structure updated with ICVision classification\n');
    if strcmp(mode, 'clean')
        fprintf('EEG data cleaned (artifacts removed)\n');
    end
    fprintf('Access results via: EEG.etc.ic_classification.ICLabel\n\n');
    
catch ME
    % Clean up temporary files on error
    cleanup_files = {temp_filepath, [temp_filepath(1:end-4) '.fdt'], icalabel_filepath};
    if exist('cleaned_filepath', 'var')
        cleanup_files{end+1} = cleaned_filepath;
        cleanup_files{end+1} = [cleaned_filepath(1:end-4) '.fdt'];
    end
    
    for i = 1:length(cleanup_files)
        if exist('cleanup_files', 'var') && exist(cleanup_files{i}, 'file')
            delete(cleanup_files{i});
        end
    end
    
    % Re-throw the error
    rethrow(ME);
end

end