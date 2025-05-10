%% Load Training and Testing Data for Humerus
parentDir = 'C:\Users\n9065\Desktop\Humerus';
dataDir = 'Training';

TrainHumerus = imageDatastore(fullfile(parentDir, dataDir), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
TrainHumerus.ReadFcn = @customReadDatastoreImage;

dataDir = 'Test';
TestHumerus = imageDatastore(fullfile(parentDir, dataDir), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
TestHumerus.ReadFcn = @customReadDatastoreImage;

%% Feature Extraction for Humerus
% Load models for Humerus
net1 = inceptionresnetigHumerousV3WithAttentionVVVV2222;
net2 = xceptionTLbigHumerousV3withAttention;

layer1 = 'newHumerousAttention1';
layer2 = 'HumerusAttention1';

FeatureTrainXCEPTIONHumerus = activations(net1, TrainHumerus, layer1, 'outputAs', 'rows');
FeatureTestXCEPTIONHumerus = activations(net1, TestHumerus, layer1, 'outputAs', 'rows');

FeatureTrainInceptionResHumerus = activations(net2, TrainHumerus, layer2, 'outputAs', 'rows');
FeatureTestInceptionResHumerus = activations(net2, TestHumerus, layer2, 'outputAs', 'rows');

% Combine features from both models
FeatureTrainCombinedHumerus = [FeatureTrainXCEPTIONHumerus, FeatureTrainInceptionResHumerus];
FeatureTestCombinedHumerus = [FeatureTestXCEPTIONHumerus, FeatureTestInceptionResHumerus];

%% t-SNE Before SIFT (Combined Model Features)
numRows = size(FeatureTrainCombinedHumerus, 1);
perplexityValue = min(30, max(5, floor(numRows / 3)));

if numRows < 2
    error('Not enough data points for t-SNE.');
else
    tsneBeforeSIFT = tsne(FeatureTrainCombinedHumerus, 'Perplexity', perplexityValue);
end

% Visualization: t-SNE Before SIFT
figure;
gscatter(tsneBeforeSIFT(:,1), tsneBeforeSIFT(:,2), TrainHumerus.Labels);
title('t-SNE of Combined Model Features for Humerus (Before SIFT)');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');

%% SIFT Feature Extraction
[trainDescriptors, trainLabels] = extractSIFTFeatures(TrainHumerus);

% Debugging Output
if isempty(trainDescriptors)
    error('trainDescriptors is empty. Check the SIFT extraction function.');
else
    disp(['Number of descriptors extracted: ', num2str(size(trainDescriptors, 1))]);
    disp(['Descriptor dimensionality: ', num2str(size(trainDescriptors, 2))]);
end

% Determine the maximum allowed PCA components
maxDimHumerus = min(size(FeatureTrainCombinedHumerus));
maxDimSIFT = min(size(trainDescriptors));
%%
% Set targetDim to the smallest value among 50, maxDimHumerus, and maxDimSIFT
targetDim = min([50, maxDimHumerus, maxDimSIFT]);
% PCA on FeatureTrainCombinedHumerus
[~, reducedHumerus] = pca(FeatureTrainCombinedHumerus, 'NumComponents', targetDim);

% PCA on trainDescriptors
[~, reducedSIFT] = pca(trainDescriptors, 'NumComponents', targetDim);

% Combine reduced features
combinedFeaturesWithSIFT = [reducedHumerus; reducedSIFT];

% Debug combined size
disp(['Size of combinedFeaturesWithSIFT: ', num2str(size(combinedFeaturesWithSIFT))]);

%% t-SNE After SIFT
numRowsAfterSIFT = size(combinedFeaturesWithSIFT, 1);
perplexityValueAfterSIFT = min(30, max(5, floor(numRowsAfterSIFT / 3)));

if numRowsAfterSIFT < 2
    error('Not enough data points for t-SNE.');
else
    tsneAfterSIFT = tsne(combinedFeaturesWithSIFT, 'Perplexity', perplexityValueAfterSIFT);
end

% Visualization: t-SNE After SIFT
figure;
gscatter(tsneAfterSIFT(:,1), tsneAfterSIFT(:,2), [TrainHumerus.Labels; trainLabels]);
title('t-SNE of Combined Model and SIFT Features for Humerus (After SIFT)');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');

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
        img = readimage(datastore, i);  % Read the image
        if size(img, 3) == 3
            img = rgb2gray(img);  % Convert to grayscale if RGB
        elseif size(img, 3) ~= 1
            disp(['Skipping unsupported image format at index ', num2str(i)]);
            continue;
        end
        
        try
            % Resize image for consistent processing
            img = imresize(img, [256, 256]);
            
            % Extract keypoints and descriptors using SIFT
            [keypoints, desc] = detectAndDescribe(img);
            
            % Accumulate descriptors and labels
            descriptors = [descriptors; desc];
            labels = [labels; repmat(datastore.Labels(i), size(desc, 1), 1)];
        catch ME
            disp(['Error processing image at index ', num2str(i), ': ', ME.message]);
        end
    end
end

function [keypoints, descriptors] = detectAndDescribe(img)
    % Use MATLAB's detectSURFFeatures and extractFeatures as SIFT alternative
    points = detectSURFFeatures(img);
    [descriptors, keypoints] = extractFeatures(img, points);
end