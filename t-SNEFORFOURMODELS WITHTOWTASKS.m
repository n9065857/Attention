%% Load and Prepare Data for Wrist and Humerus
% Wrist Data
parentDirWrist = 'C:\Users\n9065\Desktop\Wrist';
dataDirWrist = 'Test';
TestWrist = imageDatastore(fullfile(parentDirWrist, dataDirWrist), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
TestWrist.ReadFcn = @customReadDatastoreImage;

% Humerus Data
parentDirHumerus = 'C:\Users\n9065\Desktop\Humerus';
dataDirHumerus = 'Test';
TestHumerus = imageDatastore(fullfile(parentDirHumerus, dataDirHumerus), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
TestHumerus.ReadFcn = @customReadDatastoreImage;

%% Feature Extraction from Four Models
% Wrist Features
FeatureTestXCEPTIONWrist = activations(net1_wrist, TestWrist, layer1_wrist, 'OutputAs', 'rows');
FeatureTestInceptionResWrist = activations(net2_wrist, TestWrist, layer2_wrist, 'OutputAs', 'rows');

% Humerus Features
FeatureTestXCEPTIONHumerus = activations(net1_humerus, TestHumerus, layer1_humerus, 'OutputAs', 'rows');
FeatureTestInceptionResHumerus = activations(net2_humerus, TestHumerus, layer2_humerus, 'OutputAs', 'rows');

% Combine features from four models (Before Feature Selection)
FeatureTestCombinedAll = [
    FeatureTestXCEPTIONWrist; 
    FeatureTestInceptionResWrist;
    FeatureTestXCEPTIONHumerus; 
    FeatureTestInceptionResHumerus
];

% Assign labels for visualization
labelsAll = [
    repmat("XceptionWrist", size(FeatureTestXCEPTIONWrist, 1), 1);
    repmat("InceptionResWrist", size(FeatureTestInceptionResWrist, 1), 1);
    repmat("XceptionHumerus", size(FeatureTestXCEPTIONHumerus, 1), 1);
    repmat("InceptionResHumerus", size(FeatureTestInceptionResHumerus, 1), 1)
];

%% t-SNE for Combined Features (Before Feature Selection)
rng default; % For reproducibility
nAll = size(FeatureTestCombinedAll, 1);
idxAll = randsample(nAll, nAll); % Shuffle indices
X_all_before = FeatureTestCombinedAll(idxAll, :);
labels_all_before = labelsAll(idxAll);

% Determine PCA components
numFeaturesAllBefore = size(X_all_before, 2);
numPCAComponentsAllBefore = min(numFeaturesAllBefore, 50);

% Apply t-SNE
disp('Applying 2D t-SNE for Combined Features (Before Feature Selection)...');
Y_all_before = tsne(X_all_before, 'Algorithm', 'barneshut', ...
                     'NumPCAComponents', numPCAComponentsAllBefore, 'NumDimensions', 2);

% Define custom colors for models
clrAllBefore = [
    1, 0, 0; % Red for XceptionWrist
    0, 1, 0; % Green for InceptionResWrist
    0, 0, 1; % Blue for XceptionHumerus
    1, 1, 0  % Yellow for InceptionResHumerus
];

% Plot t-SNE Results for Combined Features (Before Feature Selection)
figure;
gscatter(Y_all_before(:,1), Y_all_before(:,2), labels_all_before, clrAllBefore);
title('t-SNE Visualization for Combined Features (Before Feature Selection)');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
grid on;

%% Feature Selection for Combined Features
% Dynamically adjust the target dimension for PCA
targetDimAll = min(size(FeatureTestCombinedAll, 2), 50); % Ensure targetDimAll <= number of features

% Apply PCA for feature selection
[~, FeatureTestSelectedAll] = pca(FeatureTestCombinedAll, 'NumComponents', targetDimAll);

%% t-SNE for Combined Features (After Feature Selection)
rng default; % For reproducibility
X_all_after = FeatureTestSelectedAll(idxAll, :); % Shuffle after feature selection

% Apply t-SNE
disp('Applying 2D t-SNE for Combined Features (After Feature Selection)...');
Y_all_after = tsne(X_all_after, 'Algorithm', 'barneshut', ...
                    'NumPCAComponents', targetDimAll, 'NumDimensions', 2);

% Plot t-SNE Results for Combined Features (After Feature Selection)
figure;
gscatter(Y_all_after(:,1), Y_all_after(:,2), labels_all_before, clrAllBefore);
title('t-SNE Visualization for Combined Features (After Feature Selection)');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
grid on;

%% Custom Function: customReadDatastoreImage
function data = customReadDatastoreImage(filename)
    % Read the image file
    data = imread(filename); 
    % Ensure the image has at least 3 channels
    data = data(:, :, min(1:3, end)); 
    % Resize the image to 299x299 pixels
    data = imresize(data, [299 299]);
end
