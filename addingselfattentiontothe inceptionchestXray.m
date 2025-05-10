% Load Pretrained Network
net = InceptionChestXrayGroup1;
lgraph = layerGraph(net);
%%
net.Layers
%%
% Specify Layers to Attach Self-Attention Mechanism
layerName = 'conv2d_98';
nextLayerName = 'batch_normalization_98';
numChannels = 128;
%%
% Add Self-Attention Layer
lgraph = addSelfAttentionLayer(lgraph, layerName, nextLayerName, numChannels);

%%
% Modify Final Layers
numClasses = 2;
newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'newChestInceptionG1Attention1', ...
    'WeightLearnRateFactor', 5, 'BiasLearnRateFactor', 5);
lgraph = replaceLayer(lgraph, 'newChestInceptionG1', newFCLayer);
newClassLayer = classificationLayer('Name', 'newChestInceptionG12Attention11');
lgraph = replaceLayer(lgraph, 'newChestInceptionG12', newClassLayer);
%%
%% Analyze and Visualize the Modified Network
analyzeNetwork(lgraph);
plot(lgraph);
%%
% Define the Parent Directory and Data Directory
parentDir = 'C:\Users\n9065\Desktop\New folder (2)ChestXrayG1\Group1';
dataDir = 'Train'; % Adjust as needed
%%
% Create an ImageDatastore with a custom read function
allImages = imageDatastore(fullfile(parentDir, dataDir), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @customReadDatastoreImage);  % Explicitly assign the ReadFcn

% Display the number of images in the datastore
disp(['Number of allImages: ', num2str(numel(allImages.Files))]);
%%
% Show an example image from the imageDatastore
img = readimage(allImages, 16);
imshow(img);
%%
% Split Data into Training and Validation Sets
rng default
[imgsTrain, imgsValidation] = splitEachLabel(allImages, 0.80, 'randomized');
disp(['Number of training images: ', num2str(numel(imgsTrain.Files))]);
disp(['Number of validation images: ', num2str(numel(imgsValidation.Files))]);
%%
% Data Augmentation
%imageAugmenter = imageDataAugmenter('RandRotation', [-10, 10], ...
    %'RandXTranslation', [-5, 5], 'RandYTranslation', [-5, 5], ...
   % 'RandXReflection', true);
%imgsTrain = augmentedImageDatastore([299 299], imgsTrain, 'DataAugmentation', imageAugmenter);
%%
%% Set Training Options and Train the Network
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 100, ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', imgsValidation, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');

rng default
%%
inceptionresnetv2ChestXrayWithAttention = trainNetwork(imgsTrain, lgraph, options);
%%
% Save the Trained Model
save('inceptionresnetv2ChestXrayWithAttention', '-v7.3');
%%
% Custom Read Function for ImageDatastore
function data=customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
%   data = normalize (data, 'range');
data = imresize(data,[299 299]);

end

