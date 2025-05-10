%% Load Training and Testing Data for Humerus
parentDir = 'C:\Users\n9065\Desktop\New folder (2)ChestXrayG1\Group2';
dataDir = 'Train';

TrainGroup2 = imageDatastore(fullfile(parentDir, dataDir), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
TrainGroup2.ReadFcn = @customReadDatastoreImage;

dataDir = 'Test';
TestGroup2 = imageDatastore(fullfile(parentDir, dataDir), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
TestGroup2.ReadFcn = @customReadDatastoreImage;
%%
%% Feature Extraction for Humerus
% Load models for Humerus
net11 = InceptionResNetv2ChestXrayGroup2WithAttention;
net22 = XceptionChestXrayGroup2WithAttention;

layer11 = 'newChestInceptionG2Attention1';
layer22 = 'newChestXceptionG2Attention1';

FeatureTrainXCEPTIONGroup2 = activations(net11, TrainGroup2, layer11, 'outputAs', 'rows');
FeatureTestXCEPTIONGroup2 = activations(net11, TestGroup2, layer11, 'outputAs', 'rows');

FeatureTrainInceptionResGroup2 = activations(net22, TrainGroup2, layer22, 'outputAs', 'rows');
FeatureTestInceptionResGroup2 = activations(net22, TestGroup2, layer22, 'outputAs', 'rows');

FeatureTrainAllGroup2 = [FeatureTrainXCEPTIONGroup2, FeatureTrainInceptionResGroup2];
FeatureTestAllGroup2 = [FeatureTestXCEPTIONGroup2, FeatureTestInceptionResGroup2];
% Combine features from four models (Before Feature Selection)
FeatureTestCombinedAll = [
   
    FeatureTestXCEPTIONGroup2; 
    FeatureTestInceptionResGroup2
];

labelsAll = [
    
    repmat("XceptionGroup2", size(FeatureTestXCEPTIONGroup2, 1), 1);
    repmat("InceptionResGroup2", size(FeatureTestInceptionResGroup2, 1), 1)
];
%%
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
%%
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

%% Combine Features from Two Models for Humerus

% Load features from the two models
FeatureTestCombinedGroup2 = [FeatureTestXCEPTIONGroup2, FeatureTestInceptionResGroup2]; % Combined features

% Shuffle the data and labels for randomness
rng default; % For reproducibility
n = size(FeatureTestCombinedGroup2, 1); % Number of test data points
idx = randsample(n, n); % Shuffle indices

% Extract features and labels for Humerus (Before SIFT)
X_before = FeatureTestCombinedGroup2(idx, :); % Shuffled data
labels_before = TestGroup2.Labels(idx); % Shuffled labels

% Map activity labels
activities_before = unique(labels_before); % Unique activity labels from dataset
activity_before = activities_before(labels_before); % Map labels to activity names

%% t-SNE Before SIFT (Two Models)
% Determine the number of features
numFeatures_before = size(X_before, 2);

% Adjust NumPCAComponents to be within valid range
numPCAComponents_before = min(numFeatures_before, 50); % Ensure it doesn't exceed the number of columns

% Apply 2D t-SNE
disp('Applying 2D t-SNE for combined features (Before SIFT)...');
Y_before = tsne(X_before, 'Algorithm', 'barneshut', ...
                'NumPCAComponents', numPCAComponents_before, 'NumDimensions', 2);

% Visualize t-SNE Results (Before SIFT)
figure;
numGroups_before = length(activities_before);
clr_before = hsv(numGroups_before); % Generate colors for groups
gscatter(Y_before(:,1), Y_before(:,2), activity_before, clr_before);
title('t-SNE Visualization of Combined Features (Before SIFT)');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
grid on;

%% SIFT Feature Extraction for Humerus
[descriptorsGroup2, siftLabels] = extractSIFTFeatures(TestGroup2);

% Combine with features from models (After SIFT)
maxDim = min(size(FeatureTestCombinedGroup2, 2), size(descriptorsGroup2, 2));
targetDim = min(50, maxDim);

% Apply PCA to align dimensions
[~, reducedGroup2Features] = pca(FeatureTestCombinedGroup2, 'NumComponents', targetDim);
[~, reducedSIFTFeatures] = pca(descriptorsGroup2, 'NumComponents', targetDim);

% Combine reduced features
combinedFeaturesAfterSIFT = [reducedGroup1Features; reducedSIFTFeatures];

% Shuffle data and labels for After SIFT
n_after = size(combinedFeaturesAfterSIFT, 1);
idx_after = randsample(n_after, n_after);
X_after = combinedFeaturesAfterSIFT(idx_after, :);
labels_after = [labels_before; siftLabels]; % Combine original and SIFT labels
labels_after = labels_after(idx_after); % Shuffle labels

% Map activity labels for After SIFT
activities_after = unique(labels_after); % Unique activity labels from dataset
activity_after = activities_after(labels_after); % Map labels to activity names

%% t-SNE After SIFT (Two Models + SIFT)
% Determine the number of features
numFeatures_after = size(X_after, 2);

% Adjust NumPCAComponents to be within valid range
numPCAComponents_after = min(numFeatures_after, 50);

% Apply 2D t-SNE
disp('Applying 2D t-SNE for combined features (After SIFT)...');
Y_after = tsne(X_after, 'Algorithm', 'barneshut', ...
               'NumPCAComponents', numPCAComponents_after, 'NumDimensions', 2);

% Visualize t-SNE Results (After SIFT)
figure;
numGroups_after = length(activities_after);
clr_after = hsv(numGroups_after); % Generate colors for groups
gscatter(Y_after(:,1), Y_after(:,2), activity_after, clr_after);
title('t-SNE Visualization of Combined Features (After SIFT)');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
grid on;
%%
%% Custom Functions
function data = customReadDatastoreImage(filename)
    data = imread(filename); 
    data = data(:, :, min(1:3, end)); 
    data = imresize(data, [299 299]);
end

function [descriptors, labels] = extractSIFTFeatures(datastore)
    % Ensure the input is an imageDatastore
    if ~isa(datastore, 'matlab.io.datastore.ImageDatastore')
        error('Input must be an imageDatastore.');
    end
    
    % Initialize outputs
    descriptors = [];
    labels = [];
    
    % Process each image in the datastore
    numImages = numel(datastore.Files);
    for i = 1:numImages
        img = readimage(datastore, i);  % Read each image
        if size(img, 3) == 3
            img = rgb2gray(img);  % Convert to grayscale if RGB
        end
        img = imresize(img, [256, 256]); % Resize for consistency
        
        try
            % Use detectSURFFeatures and extractFeatures as SIFT alternative
            points = detectSURFFeatures(img);  % Detect keypoints
            [desc, ~] = extractFeatures(img, points);  % Extract descriptors
            
            % Accumulate descriptors and labels
            descriptors = [descriptors; desc];
            labels = [labels; repmat(datastore.Labels(i), size(desc, 1), 1)];
        catch ME
            disp(['Error processing image at index ', num2str(i), ': ', ME.message]);
        end
    end
end
