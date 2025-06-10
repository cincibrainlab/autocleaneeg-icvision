function [EEG, results] = icvision_matlab_integration_enhanced_mock(EEG, varargin)
% ICVISION_MATLAB_INTEGRATION_ENHANCED_MOCK  Mock version for testing both modes
%
% This function provides the same interface as icvision_matlab_integration_enhanced
% but uses fake classification data to test both modes without API calls.
%
% Usage:
%   [EEG, results] = icvision_matlab_integration_enhanced_mock(EEG)
%   [EEG, results] = icvision_matlab_integration_enhanced_mock(EEG, 'Mode', 'classify')
%   [EEG, results] = icvision_matlab_integration_enhanced_mock(EEG, 'Mode', 'clean')

% Parse input arguments
p = inputParser;
addRequired(p, 'EEG', @isstruct);
addParameter(p, 'Mode', 'classify', @(x) ismember(lower(x), {'classify', 'clean'}));
addParameter(p, 'TempDir', tempdir, @ischar);
parse(p, EEG, varargin{:});

mode = lower(p.Results.Mode);
temp_dir = p.Results.TempDir;

% Validate EEG structure
if ~isstruct(EEG)
    error('EEG must be a structure');
end

if ~isfield(EEG, 'icaweights') || isempty(EEG.icaweights)
    error('EEG structure must contain ICA decomposition (run ICA first)');
end

fprintf('=== ICVision MATLAB Integration (Enhanced Mock) ===\n');
fprintf('Processing EEG dataset: %s\n', EEG.setname);
fprintf('Mode: %s\n', upper(mode));

n_components = size(EEG.icaweights, 1);
fprintf('Number of ICA components: %d\n', n_components);

try
    % Create mock classification data
    fprintf('Generating mock ICVision classification data...\n');
    
    % ICLabel classes (exactly 7 classes as expected by EEGLAB)
    icalabel_classes = {'Brain'; 'Muscle'; 'Eye'; 'Heart'; 'Line Noise'; 'Channel Noise'; 'Other'};
    
    % Generate realistic mock classification
    rng(42); % For reproducible results
    
    % Create probability matrix (n_components x 7)
    classification_matrix = rand(n_components, 7);
    
    % Normalize each row to sum to 1.0
    row_sums = sum(classification_matrix, 2);
    classification_matrix = classification_matrix ./ row_sums;
    
    % Make realistic classifications with some artifacts
    artifact_components = [];
    for i = 1:n_components
        % Assign realistic dominant classes
        if i <= round(n_components * 0.6)
            % 60% brain components
            dominant_class = 1; % Brain
        elseif i <= round(n_components * 0.75)
            % 15% muscle artifacts
            dominant_class = 2; % Muscle
            artifact_components = [artifact_components, i-1]; % 0-indexed for Python compatibility
        elseif i <= round(n_components * 0.85)
            % 10% eye artifacts
            dominant_class = 3; % Eye
            artifact_components = [artifact_components, i-1];
        elseif i <= round(n_components * 0.9)
            % 5% heart artifacts
            dominant_class = 4; % Heart
            artifact_components = [artifact_components, i-1];
        else
            % 10% other
            dominant_class = 7; % Other
            artifact_components = [artifact_components, i-1];
        end
        
        % Set probabilities
        classification_matrix(i, :) = classification_matrix(i, :) * 0.2;
        classification_matrix(i, dominant_class) = 0.6 + rand() * 0.3;
        
        % Renormalize
        classification_matrix(i, :) = classification_matrix(i, :) / sum(classification_matrix(i, :));
    end
    
    % Create mock ICLabel structure
    mock_icalabel = struct();
    mock_icalabel.classes = icalabel_classes;
    mock_icalabel.classification = classification_matrix;
    mock_icalabel.version = 'ICVision1.0-Enhanced-Mock';
    mock_icalabel.n_components = n_components;
    mock_icalabel.artifacts_removed = artifact_components;
    mock_icalabel.cleaning_applied = strcmp(mode, 'clean');
    
    % Apply cleaning if requested
    if strcmp(mode, 'clean')
        fprintf('Applying mock artifact removal...\n');
        fprintf('Mock artifacts to remove: %d components\n', length(artifact_components));
        
        % Simulate artifact removal by actually modifying the EEG data
        if length(artifact_components) > 0 && isfield(EEG, 'icaact')
            fprintf('Simulating artifact removal from EEG data...\n');
            
            % Create a copy of the original data for reconstruction
            if ~isfield(EEG, 'data') || isempty(EEG.data)
                fprintf('Warning: No EEG.data found, cannot simulate cleaning\n');
            else
                % Simulate removing artifacts by reconstructing signal without artifact components
                % This is a simplified simulation - real ICVision does proper ICA reconstruction
                
                % Get ICA activations if available, otherwise simulate
                if isfield(EEG, 'icaact') && ~isempty(EEG.icaact)
                    ica_activations = EEG.icaact;
                else
                    % Simulate ICA activations if not available
                    fprintf('Simulating ICA activations for mock cleaning...\n');
                    ica_activations = randn(n_components, size(EEG.data, 2)) * 0.1;
                end
                
                % Zero out artifact components in activations (simulate removal)
                ica_activations_cleaned = ica_activations;
                for i = 1:length(artifact_components)
                    comp_idx = artifact_components(i) + 1; % Convert to 1-indexed
                    if comp_idx <= size(ica_activations_cleaned, 1)
                        ica_activations_cleaned(comp_idx, :) = 0;
                        fprintf('  Zeroed component %d (artifact type: %s)\n', comp_idx, get_component_type(comp_idx, n_components));
                    end
                end
                
                % Simulate mixing matrix effect (simplified)
                % In real ICA: data = mixing_matrix * ica_activations
                % Here we just add small random changes to simulate the effect
                original_std = std(EEG.data(:));
                noise_reduction = 0.1; % Simulate 10% noise reduction
                EEG.data = EEG.data + randn(size(EEG.data)) * original_std * noise_reduction;
                
                fprintf('Mock artifact removal applied to EEG.data\n');
            end
        else
            fprintf('No ICA activations available, simulating basic noise reduction...\n');
            % Just add small changes to simulate cleaning effect
            if isfield(EEG, 'data') && ~isempty(EEG.data)
                original_std = std(EEG.data(:));
                EEG.data = EEG.data + randn(size(EEG.data)) * original_std * 0.05;
                fprintf('Applied mock noise reduction to EEG.data\n');
            end
        end
        
        % Update processing history
        if ~isfield(EEG, 'history')
            EEG.history = '';
        end
        EEG.history = [EEG.history sprintf('EEG = icvision_matlab_integration_enhanced_mock(EEG, ''Mode'', ''clean''); %% %s (MOCK)\n', datestr(now))];
        EEG.setname = [EEG.setname '_ICVision_cleaned_mock'];
        
        fprintf('Mock cleaning completed - EEG.data has been modified\n');
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
    EEG.etc.ic_classification.ICLabel = mock_icalabel;
    
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
    results.artifacts_removed = artifact_components;
    results.n_artifacts = length(artifact_components);
    
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
        fprintf('\nMock Artifact Removal Summary:\n');
        fprintf('  Components that would be removed: %d\n', results.n_artifacts);
        fprintf('  Component indices: %s\n', mat2str(results.artifacts_removed));
        fprintf('  Mock cleaning applied\n');
    elseif strcmp(mode, 'classify')
        fprintf('\nClassification-only mode: EEG data unchanged\n');
        if results.n_artifacts > 0
            fprintf('  Found %d artifact components (not removed)\n', results.n_artifacts);
            fprintf('  Use ''Mode'', ''clean'' to remove artifacts\n');
        end
    end
    
    fprintf('=== Mock Integration Complete ===\n');
    fprintf('Mode: %s\n', upper(mode));
    fprintf('EEG structure updated with mock ICVision classification\n');
    if strcmp(mode, 'clean')
        fprintf('Mock cleaning applied\n');
    end
    fprintf('Access results via: EEG.etc.ic_classification.ICLabel\n\n');
    
    % Test structure access
    fprintf('Testing structure access:\n');
    fprintf('  Number of classes: %d\n', length(EEG.etc.ic_classification.ICLabel.classes));
    fprintf('  First class: %s\n', EEG.etc.ic_classification.ICLabel.classes{1});
    fprintf('  Classification shape: [%d x %d]\n', size(EEG.etc.ic_classification.ICLabel.classification));
    
catch ME
    % Re-throw the error
    rethrow(ME);
end

end

function comp_type = get_component_type(comp_idx, n_components)
    % Helper function to determine mock component type based on index
    if comp_idx <= round(n_components * 0.6)
        comp_type = 'Brain';
    elseif comp_idx <= round(n_components * 0.75)
        comp_type = 'Muscle';
    elseif comp_idx <= round(n_components * 0.85)
        comp_type = 'Eye';
    elseif comp_idx <= round(n_components * 0.9)
        comp_type = 'Heart';
    else
        comp_type = 'Other';
    end
end