%% Step 1: Load Wrist Test Data
testDataDir = 'C:\Users\n9065\Desktop\Wrist\Test';
TestWrist = imageDatastore(testDataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @customReadDatastoreImage);

% Combine the ground truth labels
groundTruthLabels = TestWrist.Labels;

disp('Loaded Wrist Test Data');
disp(['Number of samples: ', num2str(numel(TestWrist.Files))]);
disp('Ground truth labels:');
tabulate(groundTruthLabels);

%% Step 2: Feature Extraction (Deep + SIFT Features)
% Replace these variables with your actual models and layers
net1 = Frominceptionresnetv2TLwristV3WithAttentionVVVV2222;
net2 = xceptionTLbigWristV3WithAttentionVVVV2222;
layer1 = 'NewWristAttention1';
layer2 = 'newWrist1Attention1';

% Extract deep features
FeatureTestXCEPTIONWrist = activations(net1, TestWrist, layer1, 'OutputAs', 'rows');
FeatureTestInceptionResWrist = activations(net2, TestWrist, layer2, 'OutputAs', 'rows');

% Extract SIFT features
[siftDescriptorsWrist, ~] = extractSIFTFeatures(TestWrist);

% Pad SIFT descriptors to match deep features
siftDescriptorsWrist = padSIFTDescriptors(siftDescriptorsWrist, size(FeatureTestXCEPTIONWrist, 1));

% Combine features
FeatureTestCombined = [FeatureTestXCEPTIONWrist, FeatureTestInceptionResWrist, siftDescriptorsWrist];
TestLabelsCombined = groundTruthLabels;

%% Step 3: Train Classifiers
% Predictor names for the feature table
numFeatures = size(FeatureTestCombined, 2);
predictorNames = arrayfun(@(x) ['Var', num2str(x)], 1:numFeatures, 'UniformOutput', false);

% Convert to table
FeatureTestTable = array2table(FeatureTestCombined, 'VariableNames', predictorNames);

% Train SVM classifier (example only, adjust Train data for proper training)
FeatureTrain = FeatureTestCombined; % Replace with actual train data
TrainLabels = TestLabelsCombined;   % Replace with actual train labels
FeatureTrainTable = array2table(FeatureTrain, 'VariableNames', predictorNames);
SVM = fitcsvm(FeatureTrainTable, TrainLabels, 'KernelFunction', 'linear');

% Train KNN classifier
KNN = fitcknn(FeatureTrainTable, TrainLabels, 'NumNeighbors', 5);
Tree = fitctree(FeatureTrainTable, TrainLabels);

% Save models
save('SVM.mat', 'SVM');
save('KNN.mat', 'KNN');
save('Tree.mat', 'Tree');

%% Step 4: Predict and Evaluate
% Predict using classifiers
predSVM = predict(SVM, FeatureTestTable);
predKNN = predict(KNN, FeatureTestTable);
%%
predTree = predict(Tree, FeatureTestTable);
%%
% Combine predictions for majority voting
predictions = [categorical(predSVM), categorical(predKNN), categorical(predTree)];
finalPredictions = arrayfun(@(i) majorityvote(predictions(i, :)), 1:size(predictions, 1))';

% Calculate accuracy
accuracy = mean(finalPredictions == TestLabelsCombined);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
confMatrix = confusionmat(TestLabelsCombined, finalPredictions);
disp('Confusion Matrix:');
disp(confMatrix);
%%
%% confusion metrix 
RE= TestWrist.Labels;                      
cm = confusionmat(RE, predSVM);
[cm,order] = confusionmat(RE, predSVM,'Order',{'Negative','Positive'});
cm1 = bsxfun (@rdivide, cm, sum(cm,2));
mean(diag(cm1))
cm2 = confusionchart(RE, predSVM);
%%

% Calculate confusion matrix
conf_mat = confusionmat(RE, predSVM);
% Calculate overall metrics
num_classes = size(conf_mat, 1);
tp = zeros(num_classes, 1);
tn = zeros(num_classes, 1);
fp = zeros(num_classes, 1);
fn = zeros(num_classes, 1);

for i = 1:num_classes
    tp(i) = conf_mat(i,i);
    tn(i) = sum(sum(conf_mat))-tp(i)-sum(conf_mat(i,:))-sum(conf_mat(:,i))+2*conf_mat(i,i);
    fp(i) = sum(conf_mat(:,i))-tp(i);
    fn(i) = sum(conf_mat(i,:))-tp(i);
end
Ko=(tp(i)+tn(i))/(tp(i)+tn(i)+fp(i)+fn(i));
  
    kpositive= (tp(i)+fp(i))*(tp(i)+fn(i)) /(tp(i)+tn(i)+fp(i)+fn(i))^2;
    knegtive= (fn(i)+tn(i))*(fp(i)+tn(i))/(tp(i)+tn(i)+fp(i)+fn(i))^2;
    Ke= kpositive+knegtive; 
accuracy = sum(tp)/sum(sum(conf_mat));
specificity = sum(tn./(tn+fp))/num_classes;
recall = sum(tp./(tp+fn))/num_classes;
precision = sum(tp./(tp+fp))/num_classes;
f1_score = (2*(precision * recall)) / (precision+recall);
K= (Ko- Ke)./ (1 - Ke);
% Display results
fprintf('Confusion matrix:\n');
disp(conf_mat);
fprintf('Accuracy: %.3f\n', accuracy);
fprintf('Specificity: %.3f\n', specificity);
fprintf('Recall: %.3f\n', recall);
fprintf('Precision: %.3f\n', precision);
fprintf('F1-score: %.3f\n', f1_score);
fprintf('K = %2.2f.\n',100*K)

%% Helper Functions

% Majority Voting
function out = majorityvote(in)
    [values, ~, idx] = unique(in);
    count = histc(idx, 1:numel(values));
    [~, maxIdx] = max(count);
    out = values(maxIdx);
end

% Custom Image Read Function
function data = customReadDatastoreImage(filename)
    data = imread(filename);
    data = data(:, :, min(1:3, end)); % Ensure 3 channels
    data = imresize(data, [299 299]); % Resize to required dimensions
end

% Extract SIFT Features
function [descriptors, labels] = extractSIFTFeatures(datastore)
    descriptors = [];
    labels = [];
    for i = 1:numel(datastore.Files)
        img = readimage(datastore, i);
        img = rgb2gray(img); % Convert to grayscale
        points = detectSURFFeatures(img); % Detect keypoints
        [features, ~] = extractFeatures(img, points); % Extract descriptors
        
        % Aggregate descriptors (mean pooling)
        if ~isempty(features)
            descriptors = [descriptors; mean(features, 1)];
        else
            descriptors = [descriptors; zeros(1, 64)]; % Handle empty features
        end
        labels = [labels; datastore.Labels(i)];
    end
end

% Pad SIFT Descriptors
function paddedDescriptors = padSIFTDescriptors(descriptors, targetRows)
    if size(descriptors, 1) < targetRows
        paddedDescriptors = [descriptors; zeros(targetRows - size(descriptors, 1), size(descriptors, 2))];
    else
        paddedDescriptors = descriptors(1:targetRows, :);
    end
end