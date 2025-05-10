% Load the trained network if not already loaded
load('inceptionresnetigHumerousV3WithAttentionVVVV2222.mat');

% Define the layer from which to extract features
featureLayer = 'avg_pool';  % Choose an appropriate layer, such as a pooling or fully connected layer

% Initialize arrays to hold features and labels
numImages = numel(imgsValidation.Files);
features = [];
labels = [];

% Loop through the images and extract features
for i = 1:numImages
    % Read and preprocess the image
    img = readimage(imgsValidation, i);
    img = imresize(img, [299, 299]);  % Adjust the size to match the network's input requirements

    % Extract features from the specified layer using the activations function
    feature = activations(inceptionresnetigHumerousV3WithAttentionVVVV2222, img, featureLayer, ...
                          'OutputAs', 'rows');  % Extract features as rows for compatibility with t-SNE

    % Append features and corresponding labels
    features = [features; feature];  % Collect features row-wise
    labels = [labels; imgsValidation.Labels(i)];  % Append the label
end

% Apply t-SNE on the extracted features
% t-SNE parameters can be tuned based on the dataset size and desired visualization
Y = tsne(features, 'NumDimensions', 2, 'Perplexity', 30, 'Exaggeration', 12);

% Visualize the t-SNE results
figure;
gscatter(Y(:,1), Y(:,2), labels);
title('t-SNE Visualization of Extracted Features');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
legend('show');
grid on;
