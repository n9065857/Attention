%% Load
A= 'C:\Users\n9065\Desktop\New folder (2)ChestXrayG1\Group1';
parentDir = A;
 dataDir = 'Test';
% dataDir = 'Test';

%%  Divide into Training and Validation Data
Test = imageDatastore(fullfile(parentDir,dataDir),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
 Test.ReadFcn = @customReadDatastoreImage;
 disp(['Number of training images: ',num2str(numel(Test.Files))]);
%  allImages.ReadFcn = @customReadDatastoreImage1;
%% Measure network accuracy
TreeModel = classify(XceptionChestXrayGroup2WithAttention,TestGroup2);

accuracy = mean(TreeModel == TestGroup2.Labels);
%% confusion metrix 
RE= TestGroup2.Labels;                      
cm = confusionmat(RE, KNNModel);
[cm,order] = confusionmat(RE,KNNModel,'Order',{'PNEUMONIA','TURBERCULOSIS'})
cm1= bsxfun (@rdivide, cm, sum(cm,2));
mean(diag(cm1))
cm2 = confusionchart(RE, KNNModel);
%%
% Calculate confusion matrix
conf_mat = confusionmat(RE, KNNModel);
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
fprintf('K = %2.2f.\n',100*K);

%%

function data=customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[299 299]);
end