%% Load and Prepare Data for Wrist and Humerus
% Wrist Data
parentDirGroup1 = 'C:\Users\n9065\Desktop\New folder (2)ChestXrayG1\Group1';
dataDirGroup1 = 'Test';
TestGroup1 = imageDatastore(fullfile(parentDirGroup1, dataDirGroup1), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
TestGroup1.ReadFcn = @customReadDatastoreImage;

% Humerus Data
parentDirGroup2 = 'C:\Users\n9065\Desktop\New folder (2)ChestXrayG1\Group2';
dataDirGroup2 = 'Test';
TestGroup2 = imageDatastore(fullfile(parentDirGroup2, dataDirGroup2), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
TestGroup2.ReadFcn = @customReadDatastoreImage;
%%
net1_Group1=inceptionresnetv2ChestXrayWithAttention
net2_Group1=XceptionChestXrayWithAttention
layer1_Group1='newChestInceptionG1Attention1'
layer2_Group1='newChestXceptionG1Attention1'
net1_Group2=InceptionResNetv2ChestXrayGroup2WithAttention
net2_Group2=XceptionChestXrayGroup2WithAttention
layer1_Group2='newChestInceptionG2Attention1'
layer2_Group2='newChestXceptionG2Attention1'
%% Feature Extraction from Four Models
% Wrist Features
FeatureTestXCEPTIONGroup1 = activations(net1_Group1, TestGroup1, layer1_Group1, 'OutputAs', 'rows');
FeatureTestInceptionResGroup1 = activations(net2_Group1, TestGroup1, layer2_Group1, 'OutputAs', 'rows');

% Humerus Features
FeatureTestXCEPTIONGroup2 = activations(net1_Group2, TestGroup2, layer1_Group2, 'OutputAs', 'rows');
FeatureTestInceptionResGroup2 = activations(net2_Group2, TestGroup2, layer2_Group2, 'OutputAs', 'rows');

% Combine features from four models (Before Feature Selection)
FeatureTestCombinedAll = [
    FeatureTestXCEPTIONGroup1; 
    FeatureTestInceptionResGroup1;
    FeatureTestXCEPTIONGroup2; 
    FeatureTestInceptionResGroup2
];

% Assign labels for visualization
labelsAll = [
    repmat("XceptionGroup1", size(FeatureTestXCEPTIONGroup1, 1), 1);
    repmat("InceptionResGroup1", size(FeatureTestInceptionResGroup1, 1), 1);
    repmat("XceptionGroup2", size(FeatureTestXCEPTIONGroup2, 1), 1);
    repmat("InceptionResGroup2", size(FeatureTestInceptionResGroup2, 1), 1)
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
