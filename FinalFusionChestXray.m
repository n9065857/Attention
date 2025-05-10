%% 
parentDir1 = 'C:\Users\n9065\Desktop\New folder (2)ChestXrayG1\Group1';  % Wrist
parentDir2 = 'C:\Users\n9065\Desktop\New folder (2)ChestXrayG1\Group2';  % Humerus

Train1 = imageDatastore(fullfile(parentDir1, 'Train'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
Train1.ReadFcn = @customReadDatastoreImage;

Test1 = imageDatastore(fullfile(parentDir1, 'Test'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
Test1.ReadFcn = @customReadDatastoreImage;

Train2 = imageDatastore(fullfile(parentDir1, 'Train'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
Train2.ReadFcn = @customReadDatastoreImage;

Test2 = imageDatastore(fullfile(parentDir1, 'Test'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
Test2.ReadFcn = @customReadDatastoreImage;

fprintf('Train2 images: %d\n', numel(Train2.Files));
fprintf('Test2 images: %d\n', numel(Test2.Files));
fprintf('TrainGroup1: %d\n', numel(Train1.Files));
fprintf('TestGroup1:  %d\n', numel(Test1.Files));
fprintf('TrainGroup2: %d\n', numel(Train2.Files));
fprintf('TestGroup2:  %d\n', numel(Test2.Files));

%%  Load Models with Attention (Once)
net1 = inceptionresnetv2ChestXrayWithAttention;
net2 = XceptionChestXrayWithAttention;
net11 = InceptionResNetv2ChestXrayGroup2WithAttention;
net22 = XceptionChestXrayGroup2WithAttention;

% Final layer names
layer1 = 'newChestInceptionG1Attention1';
layer2 = 'newChestXceptionG1Attention1';
layer11 = 'newChestInceptionG2Attention1';
layer22 = 'newChestXceptionG2Attention1';



%% Extract Deep Features from Test1
FeatureTrain1_net1 = activations(net1, Train1, layer1, 'OutputAs', 'rows');
FeatureTrain1_net2 = activations(net2, Train1, layer2, 'OutputAs', 'rows');
FeatureTest1_net1 = activations(net1, Test1, layer1, 'OutputAs', 'rows');
FeatureTest1_net2 = activations(net2, Test1, layer2, 'OutputAs', 'rows');

% Extract SIFT features (new version returns valid indices)
[trainSIFT, trainLabels, validTrainIdx] = extractSIFTFeatures(Train1);
[testSIFT, testLabels, validTestIdx] = extractSIFTFeatures(Test1);

% Apply same valid index filtering to deep features
FeatureTrain1_net1 = FeatureTrain1_net1(validTrainIdx, :);
FeatureTrain1_net2 = FeatureTrain1_net2(validTrainIdx, :);

FeatureTest1_net1 = FeatureTest1_net1(validTestIdx, :);
FeatureTest1_net2 = FeatureTest1_net2(validTestIdx, :);

% Now they all match!
assert(size(FeatureTrain1_net1,1) == size(trainSIFT,1), 'Still mismatch after filtering');

% Fuse features
FeatureTrainAllGroup1 = [FeatureTrain1_net1, FeatureTrain1_net2, trainSIFT];
FeatureTestAllGroup1  = [FeatureTest1_net1, FeatureTest1_net2, testSIFT];

LabelTrainGroup1 = trainLabels;
LabelTestGroup1  = testLabels;



%%
%% Extract Deep Features from Train1
FeatureTrain2_net11 = activations(net11, Train2, layer11, 'OutputAs', 'rows');
FeatureTrain2_net22 = activations(net22, Train2, layer22, 'OutputAs', 'rows');
FeatureTest2_net11 = activations(net11, Test2, layer11, 'OutputAs', 'rows');
FeatureTest2_net22 = activations(net22, Test2, layer22, 'OutputAs', 'rows');

% Extract SIFT features (new version returns valid indices)
[trainSIFT, trainLabels, validTrainIdx] = extractSIFTFeatures(Train2);
[testSIFT, testLabels, validTestIdx] = extractSIFTFeatures(Test2);

% Apply same valid index filtering to deep features
FeatureTrain2_net11 = FeatureTrain2_net11(validTrainIdx, :);
FeatureTrain2_net22 = FeatureTrain2_net22(validTrainIdx, :);

FeatureTest2_net11 = FeatureTest2_net11(validTestIdx, :);
FeatureTest2_net22 = FeatureTest2_net22(validTestIdx, :);

% Now they all match!
assert(size(FeatureTrain2_net11,1) == size(trainSIFT,1), 'Still mismatch after filtering');

% Fuse features
FeatureTrainAllGroup2 = [FeatureTrain2_net11, FeatureTrain2_net22, trainSIFT];
FeatureTestAllGroup2  = [FeatureTest2_net11, FeatureTest2_net22, testSIFT];

LabelTrainGroup2 = trainLabels;
LabelTestGroup2 = testLabels;



%% 
combinedFeaturesTrain = vertcat(FeatureTrainAllGroup1, FeatureTrainAllGroup2);
combinedFeaturesTest  = vertcat(FeatureTestAllGroup1, FeatureTestAllGroup2);
combinedLabelsTrain   = vertcat(LabelTrainGroup1,LabelTrainGroup2);
combinedLabelsTest    = vertcat(LabelTestGroup1, LabelTestGroup2);

%% 🧽 Normalize
combinedFeaturesTrain = normalize(combinedFeaturesTrain);
combinedFeaturesTest  = normalize(combinedFeaturesTest);

%% 🧪 Train & Predict
model = fitcecoc(combinedFeaturesTrain, combinedLabelsTrain);
predictedLabels = predict(model, combinedFeaturesTest);
accuracy = mean(predictedLabels == combinedLabelsTest);

fprintf('\n✅ Final Accuracy (Fused Groups): %.2f%%\n', accuracy * 100);
% Custom image read function
%%
function data = customReadDatastoreImage(filename)
    data = imread(filename); 
    data = data(:, :, min(1:3, end)); 
    data = imresize(data, [299 299]);
end
%%

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

