% Load the pre-trained model
load('XceptionChestXrayWithAttention.mat'); % Replace with your model file
net = XceptionChestXrayWithAttention;

% Define class names manually
classNames = ["PNEUMONIA", "TURBERCULOSIS"]; % Replace with your actual class names

% Load the image (assumed to be PNG)
imagePath = 'NORMAL2-IM-0357-0001.png'; % Replace with your image file
inputImage = imread(imagePath);

% Classify the image
[YPred, scores] = classify(net, inputImage);

% Get the top predicted classes and scores
[~, topIdx] = maxk(scores, min(3, numel(scores))); % Top predictions (limit to number of classes)
topScores = scores(topIdx); % Top scores
topClasses = classNames(topIdx); % Map indices to class names

% Display the image with classification results
imshow(imagePath);
titleString = compose("%s (%.2f)", topClasses(:), topScores(:)); % Ensure inputs are column vectors
title(sprintf(join(titleString, "; "))); % Combine into a single title string
map = occlusionSensitivity(net,inputImage,YPred);
imshow(inputImage,'InitialMagnification', 150)
hold on
imagesc(map,'AlphaData',0.5)
colormap jet
colorbar

title(sprintf("Occlusion sensitivity (%s)", ...
    YPred))