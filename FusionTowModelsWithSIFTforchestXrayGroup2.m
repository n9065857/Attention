%% Setup Paths
% Group 1 - Wrist
parentDir1 = 'C:\Users\n9065\Desktop\New folder (2)ChestXrayG1\Group2';



%% Load Train/Test Data for Group1
Train2 = imageDatastore(fullfile(parentDir1, 'Train'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
Train2.ReadFcn = @customReadDatastoreImage;

Test2 = imageDatastore(fullfile(parentDir1, 'Test'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
Test2.ReadFcn = @customReadDatastoreImage;

fprintf('Train2 images: %d\n', numel(Train2.Files));
fprintf('Test2 images: %d\n', numel(Test2.Files));

%% Load Deep Networks with Attention
net1 = InceptionResNetv2ChestXrayGroup2WithAttention;
net2 = XceptionChestXrayGroup2WithAttention;

% Final fully connected layers from each network
layer1 = 'newChestInceptionG2Attention1';
layer2 = 'newChestXceptionG2Attention1';

%% Extract Deep Features from Train1
FeatureTrain2_net1 = activations(net1, Train2, layer1, 'OutputAs', 'rows');
FeatureTrain2_net2 = activations(net2, Train2, layer2, 'OutputAs', 'rows');

%% Extract Deep Features from Test1
FeatureTest2_net1 = activations(net1, Test2, layer1, 'OutputAs', 'rows');
FeatureTest2_net2 = activations(net2, Test2, layer2, 'OutputAs', 'rows');

% Extract SIFT features (new version returns valid indices)
[trainSIFT, trainLabels, validTrainIdx] = extractSIFTFeatures(Train2);
[testSIFT, testLabels, validTestIdx] = extractSIFTFeatures(Test2);

% Apply same valid index filtering to deep features
FeatureTrain2_net1 = FeatureTrain2_net1(validTrainIdx, :);
FeatureTrain2_net2 = FeatureTrain2_net2(validTrainIdx, :);

FeatureTest2_net1 = FeatureTest2_net1(validTestIdx, :);
FeatureTest2_net2 = FeatureTest2_net2(validTestIdx, :);

% Now they all match!
assert(size(FeatureTrain2_net1,1) == size(trainSIFT,1), 'Still mismatch after filtering');

% Fuse features
FeatureTrainAllGroup2 = [FeatureTrain2_net1, FeatureTrain2_net2, trainSIFT];
FeatureTestAllGroup2  = [FeatureTest2_net1, FeatureTest2_net2, testSIFT];

LabelTrainGroup2 = trainLabels;
LabelTestGroup2 = testLabels;



%% Normalize Features (Recommended for ML Classifiers)
%FeatureTrainAllGroup1 = vertcat(FeatureTrainAllGroup1);
%FeatureTestAllGroup1  = vertcat(FeatureTestAllGroup1);

%% Train a Classifier (Example: SVM)

predictedLabels =NeuralNetwork.predictFcn(FeatureTestAllGroup2); 


%%
%%
% Custom image read function
function data = customReadDatastoreImage(filename)
    data = imread(filename); 
    data = data(:, :, min(1:3, end)); 
    data = imresize(data, [299 299]);
end
%%
function [features, labels, validIdx] = extractSIFTFeatures(imds)
    % Robust SIFT-like feature extractor using bag of features

    % Build bag of features dictionary
    bag = bagOfFeatures(imds, ...
        'VocabularySize', 500, ...
        'StrongestFeatures', 0.8, ...
        'PointSelection', 'Detector');

    % Initialize
    numImages = numel(imds.Files);
    features = [];
    labels = [];
    validIdx = false(numImages, 1);

    for i = 1:numImages
        try
            img = readimage(imds, i);
            feat = encode(bag, img);  % get 1x500 feature vector
            features(end+1, :) = feat;
            labels(end+1, 1) = imds.Labels(i);
            validIdx(i) = true;
        catch ME
            warning("Skipping image %d (%s): %s", i, imds.Files{i}, ME.message);
        end
    end
end
